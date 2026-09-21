from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from ..config import load_settings
from .news_features import aggregate_news
from .context import add_calendar_and_market_context
from ..utils.io import atomic_write_json, atomic_write_parquet, read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range without inheriting pandas DataFrame attrs.

    Pandas >=2.x compares ``attrs`` during concat/finalization. The project
    previously attached a full news DataFrame to ``price.attrs``; concatenating
    Series derived from that DataFrame could therefore attempt to compare two
    DataFrames and raise ``ValueError: The truth value of a DataFrame is
    ambiguous``. Build plain Series from numpy arrays and keep metadata out of
    pandas ``attrs`` entirely.
    """
    index = df.index
    high = pd.Series(pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float), index=index)
    low = pd.Series(pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float), index=index)
    close = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float), index=index)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


def _fractional_difference(series: pd.Series, d: float, threshold: float = 1e-5, max_lags: int = 200) -> pd.Series:
    # Fixed-width binomial weights; the result is shifted only by the available lag window.
    weights = [1.0]
    for k in range(1, max_lags):
        weights.append(-weights[-1] * (d - k + 1) / k)
        if abs(weights[-1]) < threshold:
            break
    weights = np.array(weights[::-1])
    out = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.astype(float).values
    width = len(weights)
    for i in range(width - 1, len(values)):
        window = values[i - width + 1 : i + 1]
        if np.isfinite(window).all():
            out.iloc[i] = float(np.dot(weights, window))
    return out


def build_features_for_symbol(
    price: pd.DataFrame,
    macro: pd.DataFrame,
    settings,
    *,
    symbol: str = "",
    news_frame: pd.DataFrame | None = None,
    bhav_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = price.copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    close = df["close"]
    df["ret_1"] = close.pct_change()
    for w in settings.features.get("return_windows", [1, 3, 5, 10, 20, 60]):
        df[f"ret_{w}"] = close.pct_change(w)
        df[f"momentum_{w}"] = close / close.shift(w) - 1

    for w in settings.features.get("sma_windows", [5, 10, 20, 50, 100, 200]):
        sma = close.rolling(w, min_periods=w).mean()
        df[f"sma_ratio_{w}"] = close / sma - 1
    for w in settings.features.get("ema_windows", [10, 20, 50]):
        ema = close.ewm(span=w, adjust=False).mean()
        df[f"ema_ratio_{w}"] = close / ema - 1

    for w in settings.features.get("volatility_windows", [5, 10, 20, 60]):
        df[f"volatility_{w}"] = df["ret_1"].rolling(w).std() * np.sqrt(252)

    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["gap_pct"] = df["open"] / df["close"].shift(1) - 1

    rsi_window = int(settings.features.get("rsi_window", 14))
    df[f"rsi_{rsi_window}"] = _rsi(close, rsi_window)
    macd_line, macd_signal, macd_hist = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    atr_window = int(settings.features.get("atr_window", 14))
    df[f"atr_pct_{atr_window}"] = _atr(df, atr_window) / close.replace(0, np.nan)

    bb_window = int(settings.features.get("bollinger_window", 20))
    bb_mid = close.rolling(bb_window).mean()
    bb_std = close.rolling(bb_window).std()
    df["bb_position"] = (close - bb_mid) / (2 * bb_std).replace(0, np.nan)
    df["bb_width"] = (4 * bb_std) / bb_mid.replace(0, np.nan)

    for w in settings.features.get("volume_windows", [5, 20, 60]):
        vmean = df["volume"].rolling(w).mean()
        vstd = df["volume"].rolling(w).std()
        df[f"relative_volume_{w}"] = df["volume"] / vmean.replace(0, np.nan)
        df[f"volume_z_{w}"] = (df["volume"] - vmean) / vstd.replace(0, np.nan)
    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_z20"] = (df["obv"] - df["obv"].rolling(20).mean()) / df["obv"].rolling(20).std()

    # G1-Step1: turnover proxy covers FULL history (yfinance-derived, known at
    # 15:30 close — no shift, same rule as other price features §7.3).
    df["turnover_cr_proxy"] = close * df["volume"] / 1e7
    _tp = df["turnover_cr_proxy"]
    df["turnover_proxy_z20"] = (_tp - _tp.rolling(20).mean()) / _tp.rolling(20).std()
    # Intraday lock proxy: close pinned at both extremes ~ circuit-locked day.
    # Exact historical price bands are not published free day-wise, so this is
    # a proxy, named as such — never a fabricated band.
    df["lock_proxy"] = ((df["close"] == df["high"]) & (df["close"] == df["low"])).astype(float)

    # G1-Step1: NSE bhavcopy delivery/turnover (2020+ on the free archive).
    # Bhavcopy publishes in the evening AFTER close, so shift(1) — same
    # conservatism as macro. Missing stays NaN (LightGBM native missing;
    # B-07 rule: never 0-fill) with delivery_available flag.
    if bhav_frame is not None and not bhav_frame.empty:
        b = bhav_frame.reindex(df.index)
        df["delivery_per"] = pd.to_numeric(b["delivery_per"], errors="coerce").shift(1)
        df["turnover_lacs"] = pd.to_numeric(b["turnover_lacs"], errors="coerce").shift(1)
        df["no_trades"] = pd.to_numeric(b["no_trades"], errors="coerce").shift(1)
        _dp = df["delivery_per"]
        df["delivery_z20"] = (_dp - _dp.rolling(20).mean()) / _dp.rolling(20).std()
        _tl = df["turnover_lacs"]
        df["turnover_z20"] = (_tl - _tl.rolling(20).mean()) / _tl.rolling(20).std()
        _nt = df["no_trades"]
        df["trades_z20"] = (_nt - _nt.rolling(20).mean()) / _nt.rolling(20).std()
    else:
        df["delivery_per"] = np.nan
        df["turnover_lacs"] = np.nan
        df["no_trades"] = np.nan
        df["delivery_z20"] = np.nan
        df["turnover_z20"] = np.nan
        df["trades_z20"] = np.nan
    df["delivery_available"] = df["delivery_per"].notna().astype(float)

    # Price/technical features are information available after the close at timestamp t.
    # The target enters on t+1 open, so there is no global one-row shift here. This keeps
    # the feature timestamp and trade timestamp consistent. External/macro series are
    # shifted conservatively by one session below to account for publication timing.

    macro = macro.reindex(df.index).ffill()
    if settings.features.get("include_macro", True) and not macro.empty:
        for col in macro.columns:
            series = pd.to_numeric(macro[col], errors="coerce")
            df[f"macro_{col}"] = series.shift(1)
            df[f"macro_{col}_ret5"] = series.pct_change(5).shift(1)
            df[f"macro_{col}_ret20"] = series.pct_change(20).shift(1)
            rm = series.rolling(60, min_periods=20).mean()
            rs = series.rolling(60, min_periods=20).std().replace(0, np.nan)
            df[f"macro_{col}_z60"] = ((series - rm) / rs).shift(1)
            df[f"macro_{col}_vol20"] = series.pct_change().rolling(20).std().shift(1)
        if "nifty" in macro.columns:
            df["asset_minus_nifty_ret20"] = (close.pct_change(20) - macro["nifty"].pct_change(20)).shift(1)
        if "vix" in macro.columns:
            vix = pd.to_numeric(macro["vix"], errors="coerce")
            df["vix_regime_high"] = (vix > vix.rolling(252, min_periods=60).quantile(0.75)).shift(1).astype(float)
            df["vix_change_5d"] = vix.pct_change(5).shift(1)

    if settings.features.get("include_market_relative", True):
        market_ret = close.pct_change(20).shift(1)
        df["asset_ret20_relative_to_self"] = market_ret

    if settings.features.get("include_calendar_context", True):
        df = add_calendar_and_market_context(df)

    # G1-Step3 (F-13): regime features. All point-in-time safe.
    # regime_bull uses macro_nifty which is already shift(1)-lagged above.
    if "macro_nifty" in df.columns:
        _nifty = pd.to_numeric(df["macro_nifty"], errors="coerce")
        _dma200 = _nifty.rolling(200, min_periods=60).mean()
        df["regime_bull"] = (_nifty > _dma200).astype(float)
    else:
        df["regime_bull"] = np.nan
    # Expiry-week flag: NSE monthly expiry week (last Thursday of month;
    # deterministic calendar, known in advance — no shift).
    _idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    _expiries: list[pd.Timestamp] = []
    for _per in _idx.to_period("M").unique():
        _m = _idx[_idx.to_period("M") == _per]
        _thus = _m[_m.dayofweek == 3]
        if len(_thus):
            _expiries.append(_thus.max())
    _expiries = sorted(_expiries)
    if _expiries:
        _exp64 = np.array([np.datetime64(e) for e in _expiries])
        _idx64 = _idx.to_numpy(dtype="datetime64[ns]")
        _pos = np.clip(np.searchsorted(_exp64, _idx64, side="left"), 0, len(_expiries) - 1)
        _next_exp = pd.DatetimeIndex(_exp64[_pos])
        _dte = (_next_exp - _idx).days
        df["expiry_week_flag"] = ((_dte >= 0) & (_dte <= 4)).astype(float)
    else:
        df["expiry_week_flag"] = 0.0

    for d in settings.features.get("fractional_d_windows", [0.3, 0.5]):
        fd = _fractional_difference(np.log(close.replace(0, np.nan)), float(d))
        df[f"fracdiff_{d}"] = fd.shift(1)

    if settings.features.get("include_news", True):
        if news_frame is not None and not news_frame.empty:
            nf = aggregate_news(news_frame, pd.DatetimeIndex(df.index), str(symbol))
            # Defensive column handling: a news provider must never be able to
            # overwrite market/technical features.
            overlap = [c for c in nf.columns if c in df.columns]
            if overlap:
                nf = nf.drop(columns=overlap)
            df = df.join(nf, how="left")
        else:
            # Keep the feature schema stable even when no news is available.
            empty_news = aggregate_news(
                pd.DataFrame(), pd.DatetimeIndex(df.index), str(symbol)
            )
            df = df.join(empty_news, how="left")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _mtime(path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _load_frame(path):
    try:
        return pd.read_parquet(path) if path.exists() else None
    except Exception:
        return None


def run_feature_engineering() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "features.log")
    for d in (paths.data_features, paths.data_features_stocks,
              paths.data_features_news, paths.data_features_market):
        d.mkdir(parents=True, exist_ok=True)

    macro_path = paths.data_raw / "macro.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()
    manifest = read_json(paths.state / "data_manifest.json", default={}) or {}
    from ..data.trading_calendar import load_calendar as _load_calendar
    from .join import join_symbol_features
    from .news_context import aggregate_market, build_news_features_for_symbol
    calendar = _load_calendar()
    ref_dir = paths.data_raw.parent / "reference"
    ref_mtime = max([_mtime(ref_dir / r) for r in ("sector_map.json", "group_map.json")] + [0.0])
    join_log_path = paths.data_features / "_join_log.json"
    join_log = read_json(join_log_path, default={}) or {}

    # Market-wide rows once (all announcements + all text news).
    market_out = paths.data_features_market / "market_features.parquet"
    ann_all, news_all = [], []
    for _sym in settings.symbols:
        _safe = _sym.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        _a = _load_frame(paths.data_raw / "announcements" / f"{_safe}.parquet")
        if _a is not None and not _a.empty:
            ann_all.append(_a)
        _n = _load_frame(paths.data_raw / "news" / f"{_safe}.parquet")
        if _n is not None and not _n.empty:
            news_all.append(_n)
    ann_all = pd.concat(ann_all, ignore_index=True) if ann_all else pd.DataFrame()
    news_all = pd.concat(news_all, ignore_index=True) if news_all else pd.DataFrame()
    sessions = pd.DatetimeIndex([])
    for _sym in settings.symbols:
        _safe = _sym.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        sessions = sessions.union(pd.DatetimeIndex(pd.read_parquet(paths.data_raw / f"{_safe}.parquet").index))
    sessions = sessions.sort_values()
    _mtimes = [0.0]
    for _sym in settings.symbols:
        _safe = _sym.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        _mtimes.append(_mtime(paths.data_raw / "announcements" / f"{_safe}.parquet"))
        _mtimes.append(_mtime(paths.data_raw / "news" / f"{_safe}.parquet"))
    if not market_out.exists() or _mtime(market_out) < max(_mtimes):
        logger.info("Building market features over %d sessions.", len(sessions))
        atomic_write_parquet(aggregate_market(ann_all, news_all, sessions, calendar), market_out)
    market_feat = _load_frame(market_out)

    completed = []
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        src = paths.data_raw / f"{safe}.parquet"
        stocks_out = paths.data_features_stocks / f"{safe}_features.parquet"
        news_out = paths.data_features_news / f"{safe}_news_features.parquet"
        joined_out = paths.data_features / f"{safe}_features.parquet"
        if stocks_out.exists() and symbol in manifest:
            try:
                stocks_feat = pd.read_parquet(stocks_out)
                raw_last = pd.to_datetime(manifest[symbol]["last_date"])
                price_current = (not stocks_feat.empty) and (pd.Timestamp(stocks_feat.index.max()) >= raw_last)
                inputs_current = _mtime(stocks_out) >= max(
                    _mtime(src), _mtime(macro_path),
                    _mtime(paths.data_raw / "bhavcopy" / f"{safe}.parquet"), ref_mtime)
                rebuild_stocks = not (price_current and inputs_current)
            except Exception as exc:
                logger.warning("Could not validate stock features for %s; rebuilding: %s", symbol, exc)
                rebuild_stocks = True
        else:
            rebuild_stocks = True
        if rebuild_stocks:
            logger.info("Engineering stock features for %s.", symbol)
            price = pd.read_parquet(src)
            news_frame = _load_frame(paths.data_raw / "news" / f"{safe}.parquet")
            bhav_frame = _load_frame(paths.data_raw / "bhavcopy" / f"{safe}.parquet")
            stocks_feat = build_features_for_symbol(
                price, macro, settings, symbol=symbol,
                news_frame=news_frame if settings.enable_news else None,
                bhav_frame=bhav_frame)
            atomic_write_parquet(stocks_feat, stocks_out)
        else:
            logger.info("Stock features for %s are current; skipping.", symbol)
        # ---- news layer (decision-date context; company+market scope) ----
        news_inputs_mtime = max(
            _mtime(paths.data_raw / "news" / f"{safe}.parquet"),
            _mtime(paths.data_raw / "announcements" / f"{safe}.parquet"),
            _mtime(paths.data_raw / "gdelt_bq" / f"{safe}.parquet"),
            _mtime(ref_dir / "trading_calendar.parquet"))
        if not news_out.exists() or _mtime(news_out) < news_inputs_mtime:
            logger.info("Engineering news features for %s.", symbol)
            atomic_write_parquet(build_news_features_for_symbol(
                _load_frame(paths.data_raw / "news" / f"{safe}.parquet"),
                _load_frame(paths.data_raw / "announcements" / f"{safe}.parquet"),
                _load_frame(paths.data_raw / "gdelt_bq" / f"{safe}.parquet"),
                pd.DatetimeIndex(stocks_feat.index), symbol, calendar), news_out)
        news_feat = _load_frame(news_out)
        # ---- materialized join (readers unchanged) ----
        joined = join_symbol_features(stocks_feat, news_feat, market_feat)
        atomic_write_parquet(joined, joined_out)
        join_log[symbol] = {
            "stocks_mtime": _mtime(stocks_out), "news_mtime": _mtime(news_out),
            "market_mtime": _mtime(market_out), "rows": int(len(joined)),
            "news_cols": int(news_feat.shape[1]) if news_feat is not None else 0,
        }
        completed.append(symbol)

    atomic_write_json(join_log_path, join_log)
    update_state(paths.state, "features", status="complete", symbols=completed)
    logger.info("Feature engineering complete (split store + joined view).")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run_feature_engineering()

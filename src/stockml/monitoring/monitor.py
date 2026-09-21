from __future__ import annotations

import json

import pandas as pd

from ..config import load_settings
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .drift import population_stability_index


# B-09: raw price/volume LEVELS trend by nature — PSI on them always fires.
# Monitor only stationary transforms (returns, ratios, z-scores, ranks).
# G1: bhav raw levels (turnover value, trade counts) trend like volume —
# their z-scores stay monitored instead.
LEVEL_COLUMNS = frozenset({
    "open", "high", "low", "close", "adj_close", "adjclose",
    "volume", "obv",
    "turnover_cr_proxy", "turnover_lacs", "no_trades", "ttl_qty",
    "delivery_qty",
})

TOP_SHAP_ALERT_CAP = 20


# Deterministic calendar columns carry zero drift information: a 20-session
# window in Aug-Sep can never match a 16y month/week distribution (PSI ~11).
# Found by the G1-R(d) experiment, which also showed the reference-window
# choice barely matters (93 vs 91 alerts) — the monitored SET was the bug.
CALENDAR_COLUMNS = frozenset({
    "calendar_month", "calendar_quarter", "calendar_weekofyear", "calendar_dayofweek",
    "calendar_month_sin", "calendar_month_cos", "calendar_quarter_sin", "calendar_quarter_cos",
    "fiscal_quarter", "fiscal_year",
    "days_to_month_end", "is_quarter_end", "is_month_end", "is_month_start",
    "earnings_season_proxy",
})

# Macro point-in-time transforms that ARE stationary and stay monitored.
MACRO_TRANSFORM_MARKERS = ("_ret", "_z", "_vol", "_regime", "_change", "asset_minus")


def _is_calendar(col: str) -> bool:
    c = col.lower()
    return c in CALENDAR_COLUMNS or c.startswith("calendar_") or c.startswith("fiscal_")


def _stationary_columns(panel: pd.DataFrame) -> list[str]:
    numeric = list(panel.select_dtypes(include="number").columns)
    stationary = []
    for c in numeric:
        if c.lower() in LEVEL_COLUMNS:
            continue
        if _is_calendar(c):
            continue  # deterministic calendar — zero drift information
        if c.startswith("macro_") and not any(m in c for m in MACRO_TRANSFORM_MARKERS):
            continue  # raw macro level (e.g. macro_nifty) — trends by nature
        stationary.append(c)
    return stationary


def _top_shap_features(paths, limit: int = TOP_SHAP_ALERT_CAP) -> list[str] | None:
    try:
        shap_path = paths.results / "shap_feature_importance.csv"
        if not shap_path.exists():
            return None
        fi = pd.read_csv(shap_path)
        if "feature" not in fi.columns:
            return None
        return [str(f) for f in fi["feature"].head(limit).tolist()]
    except Exception:
        return None


def run_monitor() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "monitor.log")
    datasets = []
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = paths.data_features / f"{safe}_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["symbol"] = symbol
            datasets.append(df)
    if not datasets:
        raise RuntimeError("No feature datasets available.")
    panel = pd.concat(datasets).sort_index()
    dates = pd.DatetimeIndex(panel.index).unique().sort_values()
    latest = dates[-settings.drift_lookback :]
    reference_end = dates[max(0, len(dates) - settings.drift_lookback * 5)]
    reference = panel.loc[panel.index <= reference_end]
    current = panel.loc[panel.index.isin(latest)]

    numeric_cols = _stationary_columns(panel)
    top_features = _top_shap_features(paths)
    alert_universe = set(top_features) if top_features else set(numeric_cols)
    psi = {}
    alerts = []
    for col in numeric_cols:
        score = population_stability_index(reference[col], current[col], settings.psi_buckets)
        psi[col] = score
        if score >= settings.drift_psi_threshold and col in alert_universe:
            alerts.append({"feature": col, "psi": score})
    quality = {
        "latest_data": str(dates[-1].date()),
        "rows": int(len(panel)),
        "missing_rate": float(panel[numeric_cols].isna().mean().mean()) if len(numeric_cols) else 0.0,
        "feature_psi": psi,
        "excluded_level_columns": sorted(LEVEL_COLUMNS),
        "alerts": alerts,
        "threshold": settings.drift_psi_threshold,
    }
    atomic_write_json(paths.results / "monitoring.json", quality)
    logger.info("Monitoring complete. %d drift alerts.", len(alerts))
    update_state(paths.state, "monitor", status="complete", output=str(paths.results / "monitoring.json"))


if __name__ == "__main__":
    run_monitor()

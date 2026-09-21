from __future__ import annotations

import numpy as np
import pandas as pd

from stockml.evaluation.metrics import apply_temperature
from stockml.monitoring.drift import population_stability_index
from stockml.validation.time_splits import three_way_date_split


def test_universe_tier_is_nifty50_with_50_symbols():
    from stockml.config import load_settings
    settings = load_settings()
    assert settings.universe_tier == "nifty50"
    assert len(settings.symbols) == 50
    assert all(s.endswith(".NS") for s in settings.symbols)
    assert len(set(settings.symbols)) == 50


def test_xs_ranks_are_bounded_and_non_constant():
    from stockml.features.cross_sectional import compute_xs_ranks, DEFAULT_XS_RANK_SOURCES
    assert len(DEFAULT_XS_RANK_SOURCES) >= 12
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    base_ret = np.linspace(-0.05, 0.05, 10)
    panel = pd.DataFrame({
        "timestamp": list(idx) * 3,
        "symbol": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        # Values must differ ACROSS symbols per session — identical cross-sections
        # rank to a constant (the B-01 single-symbol trap this test guards against).
        "ret_20": list(base_ret) + list(base_ret + 0.02) + list(base_ret - 0.02),
        "volatility_20": list(np.linspace(0.1, 0.4, 10)) + list(np.linspace(0.4, 0.1, 10)) + list(np.linspace(0.2, 0.3, 10)),
        "rsi_14": list(np.linspace(30, 70, 10)) + list(np.linspace(70, 30, 10)) + list(np.linspace(40, 60, 10)),
    })
    out = compute_xs_ranks(panel)
    for col in ("ret_20_xs_rank", "volatility_20_xs_rank", "rsi_14_xs_rank"):
        assert col in out.columns
        assert out[col].between(0, 1).all()
        assert out[col].nunique() > 1


def _synthetic_price(n=300, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1_000_000,
    }, index=idx)


def test_flat_share_grows_with_flat_band():
    from stockml.evaluation.flat_band_ablation import scan_flat_bands, pick_flat_band
    frames = {"TEST.NS": _synthetic_price()}
    scan = scan_flat_bands(frames, bands=[0.25, 0.5, 0.75, 1.0])
    shares = scan["flat_share"].tolist()
    assert all(shares[i] <= shares[i + 1] + 1e-9 for i in range(len(shares) - 1))
    assert abs(scan[["down_share", "flat_share", "up_share"]].sum(axis=1) - 1.0).max() < 1e-9
    winner = pick_flat_band(scan)
    assert winner in [0.25, 0.5, 0.75, 1.0]


def test_segment_aware_costs_land_in_realistic_band():
    from types import SimpleNamespace
    from stockml.backtest.backtest import round_trip_cost_rate
    base = dict(commission_bps=1.0, slippage_bps=3.0, other_cost_bps=1.0)
    delivery, info = round_trip_cost_rate(SimpleNamespace(backtest_segment="delivery", **base))
    assert info["segment"] == "delivery"
    assert 25.0 <= info["round_trip_bps"] <= 35.0
    futures, finfo = round_trip_cost_rate(SimpleNamespace(backtest_segment="futures", **base))
    assert finfo["segment"] == "futures"
    assert abs(finfo["round_trip_bps"] - 15.0) < 1e-9
    assert futures < delivery


def test_beats_majority_flag_logic():
    acc, majority = 0.53, 0.5104
    assert bool((acc - majority) > 0) is True
    assert bool((0.4910 - majority) > 0) is False


def test_temperature_probabilities_sum_to_one():
    p = np.array([[0.2, 0.3, 0.5]])
    out = apply_temperature(p, 2.0)
    assert np.isclose(out.sum(axis=1), 1.0).all()


def test_psi_identical_is_near_zero():
    x = pd.Series(np.linspace(0, 1, 100))
    assert population_stability_index(x, x) < 1e-8


def test_date_split_has_three_partitions():
    idx = pd.date_range("2010-01-01", periods=100, freq="D")
    train, val, test, _, _ = three_way_date_split(idx)
    assert train.sum() > 0 and val.sum() > 0 and test.sum() > 0
    assert not (train & val).any()
    assert not (val & test).any()


def test_directional_metrics_are_reported():
    from stockml.evaluation.metrics import classification_metrics
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 0, 2])
    result = classification_metrics(y_true, y_pred, np.full((4, 3), 1/3))
    assert "directional_accuracy_nonflat_actual" in result
    assert "directional_accuracy_when_model_takes_direction" in result
    assert result["directional_samples_nonflat_actual"] == 3


def test_atr_does_not_fail_with_dataframe_attrs():
    from stockml.features.feature_engineering import _atr
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    frame = pd.DataFrame({
        "open": np.linspace(100, 110, 40),
        "high": np.linspace(101, 111, 40),
        "low": np.linspace(99, 109, 40),
        "close": np.linspace(100.5, 110.5, 40),
        "volume": np.arange(40) + 1000,
    }, index=idx)
    frame.attrs["news_frame"] = pd.DataFrame({"x": [1, 2, 3]})
    result = _atr(frame, 14)
    assert len(result) == len(frame)
    assert result.notna().sum() > 0


def test_macro_symbols_are_india_only():
    from stockml.data.macro import DEFAULT_MACRO_SYMBOLS
    assert "sp500" not in DEFAULT_MACRO_SYMBOLS
    assert "nasdaq" not in DEFAULT_MACRO_SYMBOLS
    assert "us10y" not in DEFAULT_MACRO_SYMBOLS
    assert DEFAULT_MACRO_SYMBOLS["vix"] == "^INDIAVIX"
    assert DEFAULT_MACRO_SYMBOLS["banknifty"] == "^NSEBANK"


def test_drift_monitor_excludes_level_columns():
    from stockml.monitoring.monitor import _stationary_columns
    frame = pd.DataFrame({
        "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
        "close": [1.0, 2.0], "volume": [1.0, 2.0], "obv": [1.0, 2.0],
        "ret_1": [0.01, -0.02], "rsi_14": [55.0, 45.0],
    })
    cols = _stationary_columns(frame)
    assert "ret_1" in cols and "rsi_14" in cols
    for raw in ("open", "high", "low", "close", "volume", "obv"):
        assert raw not in cols


def test_drift_monitor_excludes_macro_levels_but_keeps_transforms():
    from stockml.monitoring.monitor import _stationary_columns
    frame = pd.DataFrame({
        "macro_nifty": [1.0, 2.0], "macro_banknifty": [1.0, 2.0],
        "macro_vix": [1.0, 2.0], "macro_usd_inr": [1.0, 2.0],
        "macro_nifty_ret5": [0.01, -0.01], "macro_gold_z60": [0.5, -0.5],
        "vix_regime_high": [0.0, 1.0], "ret_20": [0.01, 0.02],
    })
    cols = _stationary_columns(frame)
    for raw in ("macro_nifty", "macro_banknifty", "macro_vix", "macro_usd_inr"):
        assert raw not in cols
    for t in ("macro_nifty_ret5", "macro_gold_z60", "vix_regime_high", "ret_20"):
        assert t in cols


def test_drift_monitor_excludes_calendar_columns():
    from stockml.monitoring.monitor import _stationary_columns
    frame = pd.DataFrame({
        "calendar_month": [8.0, 9.0], "fiscal_year": [2026.0, 2026.0],
        "calendar_month_sin": [0.1, 0.2], "earnings_season_proxy": [0.0, 1.0],
        "is_month_end": [0.0, 0.0], "ret_5": [0.01, -0.01],
    })
    cols = _stationary_columns(frame)
    assert cols == ["ret_5"]


def test_final_report_block_shows_edge_over_majority():
    from stockml.evaluation.final_report import format_final_report
    report = {
        "period": "2023-12-20 → 2026-09-15", "n_test": 670,
        "n_correct": 329, "n_wrong": 341, "accuracy": 0.4910,
        "majority": 0.5104, "linear": 0.4522, "edge_pp": -1.94,
        "beats_majority_baseline": False,
        "sig_total": 141, "sig_correct": 73, "sig_acc": 0.5177,
        "sig_coverage": 0.2104, "threshold": 0.55,
        "roc_auc": 0.5219, "mcc": 0.0442, "balanced_accuracy": 0.3466,
        "ece": None, "sharpe": -1.78, "total_return": -0.0506,
        "gross": 0.0077, "costs": 0.0594,
    }
    text = format_final_report(report)
    assert "EDGE OVER MAJORITY" in text
    assert "BELOW BASELINE" in text
    assert "Majority-class baseline" in text


def test_feature_builder_accepts_news_without_pandas_attrs():
    from stockml.features.feature_engineering import build_features_for_symbol
    from types import SimpleNamespace
    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    price = pd.DataFrame({
        "open": np.linspace(100, 130, 80),
        "high": np.linspace(101, 131, 80),
        "low": np.linspace(99, 129, 80),
        "close": np.linspace(100.5, 130.5, 80),
        "volume": np.arange(80) + 1000,
    }, index=idx)
    price.attrs["news_frame"] = pd.DataFrame({"x": [1, 2, 3]})
    settings = SimpleNamespace(features={
        "return_windows": [1, 3],
        "sma_windows": [5],
        "ema_windows": [5],
        "volatility_windows": [5],
        "volume_windows": [5],
        "rsi_window": 14,
        "atr_window": 14,
        "bollinger_window": 20,
        "fractional_d_windows": [0.3],
        "include_market_relative": True,
        "include_macro": False,
        "include_news": True,
        "include_calendar_context": False,
    })
    out = build_features_for_symbol(price, pd.DataFrame(index=idx), settings, symbol="^NSEI", news_frame=pd.DataFrame())
    assert "atr_pct_14" in out.columns
    assert "news_count" in out.columns


def test_ece_is_bounded_and_perfect_model_is_zero():
    from stockml.evaluation.metrics import expected_calibration_error
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    proba = np.full((500, 3), 1 / 3)
    out = expected_calibration_error(y, proba, n_bins=10)
    assert 0.0 <= out["ece"] <= 1.0
    assert out["n_bins"] == 10
    assert abs(sum(b["count"] for b in out["bins"]) - 500) < 1e-9
    # Perfect confident model → ECE 0.
    y2 = np.array([0, 1, 2, 0])
    p2 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    assert expected_calibration_error(y2, p2)["ece"] < 1e-9


def test_bhav_features_flag_availability_and_keep_nan():
    from stockml.features.feature_engineering import build_features_for_symbol
    from types import SimpleNamespace
    idx = pd.date_range("2021-01-01", periods=60, freq="B")
    price = pd.DataFrame({
        "open": np.linspace(100, 130, 60), "high": np.linspace(101, 131, 60),
        "low": np.linspace(99, 129, 60), "close": np.linspace(100.5, 130.5, 60),
        "volume": np.arange(60) + 100000,
    }, index=idx)
    settings = SimpleNamespace(features={
        "return_windows": [5], "sma_windows": [5], "ema_windows": [5],
        "volatility_windows": [5], "volume_windows": [5], "rsi_window": 14,
        "atr_window": 14, "bollinger_window": 20, "fractional_d_windows": [],
        "include_market_relative": False, "include_macro": False,
        "include_news": False, "include_calendar_context": True,
    })
    # No bhav → all NaN + flag 0 (B-07: missing distinct from neutral).
    out0 = build_features_for_symbol(price, pd.DataFrame(index=idx), settings, symbol="X.NS", bhav_frame=None)
    assert out0["delivery_available"].eq(0.0).all()
    assert out0["delivery_per"].isna().all()
    assert "turnover_proxy_z20" in out0.columns and "lock_proxy" in out0.columns
    assert "regime_bull" in out0.columns and "expiry_week_flag" in out0.columns
    assert set(out0["expiry_week_flag"].unique()) <= {0.0, 1.0}
    # With bhav → shift(1) point-in-time: first row NaN, later rows populated, flag 1.
    bhav = pd.DataFrame({
        "delivery_per": np.linspace(30, 60, 60), "turnover_lacs": 1000.0,
        "no_trades": 5000.0, "ttl_qty": 100000.0,
    }, index=idx)
    out1 = build_features_for_symbol(price, pd.DataFrame(index=idx), settings, symbol="X.NS", bhav_frame=bhav)
    assert out1["delivery_per"].iloc[0] != out1["delivery_per"].iloc[0]  # NaN from shift(1)
    assert out1["delivery_available"].iloc[-1] == 1.0
    assert out1["delivery_z20"].notna().sum() > 0


def test_group_sector_fanout_is_identical_within_bucket():
    from stockml.features.cross_sectional import add_group_sector_features
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    panel = pd.DataFrame({
        "timestamp": list(idx) * 3,
        "symbol": ["ADANIENT.NS"] * 5 + ["ADANIPORTS.NS"] * 5 + ["INFY.NS"] * 5,
        "ret_5": [0.01] * 15,
        "ret_20": [0.02] * 5 + [0.04] * 5 + [0.10] * 5,
    })
    sectors = pd.DataFrame([
        {"symbol": "ADANIENT.NS", "bucket": "Metals & Mining", "from": pd.Timestamp("2010-01-01"), "to": pd.Timestamp.max},
        {"symbol": "ADANIPORTS.NS", "bucket": "Services", "from": pd.Timestamp("2010-01-01"), "to": pd.Timestamp.max},
        {"symbol": "INFY.NS", "bucket": "Information Technology", "from": pd.Timestamp("2010-01-01"), "to": pd.Timestamp.max},
    ])
    groups = pd.DataFrame([
        {"symbol": "ADANIENT.NS", "bucket": "ADANI", "from": pd.Timestamp("2010-01-01"), "to": pd.Timestamp.max},
        {"symbol": "ADANIPORTS.NS", "bucket": "ADANI", "from": pd.Timestamp("2010-01-01"), "to": pd.Timestamp.max},
    ])
    out = add_group_sector_features(panel, sectors, groups)
    ad = out[out["symbol"].isin(["ADANIENT.NS", "ADANIPORTS.NS"])]
    # Same group → identical dispersion on every timestamp.
    assert (ad.groupby("timestamp")["group_dispersion"].nunique() == 1).all()
    assert out["ret_20_minus_group"].notna().sum() > 0
    # INFY has no group → NaN relative (kept missing, not zero-filled).
    assert out.loc[out["symbol"] == "INFY.NS", "ret_20_minus_group"].isna().all()
    assert "ret_20_minus_sector" in out.columns and "sector_xs_rank" in out.columns


def test_prepare_xy_keeps_pre2020_rows_with_nan_delivery():
    from stockml.models.lightgbm_train import prepare_xy
    idx = pd.MultiIndex.from_arrays(
        [list(pd.date_range("2015-01-01", periods=4)) * 2, ["A"] * 4 + ["B"] * 4],
        names=["timestamp", "symbol"],
    )
    df = pd.DataFrame({
        "ret_5": np.arange(8, dtype=float), "delivery_per": [np.nan] * 8,
        "delivery_z20": [np.nan] * 8, "turnover_lacs": [np.nan] * 8,
        "turnover_z20": [np.nan] * 8, "no_trades": [np.nan] * 8,
        "trades_z20": [np.nan] * 8,
        "target": [0, 1, 2, 0, 1, 2, 0, 1], "event_end": list(pd.date_range("2015-02-01", periods=8)),
    }, index=idx)
    X, y, _, _ = prepare_xy(df)
    assert len(X) == 8  # NaN delivery must NOT drop pre-2020 rows
    assert X["delivery_per"].isna().all()


def test_monitor_excludes_bhav_levels_but_keeps_transforms():
    from stockml.monitoring.monitor import _stationary_columns
    frame = pd.DataFrame({
        "turnover_cr_proxy": [1.0, 2.0], "turnover_lacs": [1.0, 2.0],
        "no_trades": [1.0, 2.0], "turnover_proxy_z20": [0.1, -0.1],
        "delivery_per": [45.0, 55.0], "delivery_z20": [0.2, -0.2],
        "delivery_available": [1.0, 1.0], "regime_bull": [1.0, 0.0],
        "ret_20_minus_sector": [0.01, -0.01],
    })
    cols = _stationary_columns(frame)
    for raw in ("turnover_cr_proxy", "turnover_lacs", "no_trades"):
        assert raw not in cols
    for t in ("turnover_proxy_z20", "delivery_per", "delivery_z20",
              "delivery_available", "regime_bull", "ret_20_minus_sector"):
        assert t in cols


def _autopsy_frame():
    from stockml.evaluation.up_autopsy import era_split
    idx = pd.date_range("2023-01-01", periods=12, freq="ME")
    df = pd.DataFrame({
        "timestamp": list(idx) * 2,
        "actual_label": [2] * 12 + [0] * 12,
        "predicted_label": [0] * 6 + [2] * 6 + [0] * 12,
        "p_down": [0.55] * 6 + [0.2] * 6 + [0.5] * 12,
        "p_flat": [0.1] * 24,
        "p_up": [0.45] * 6 + [0.7] * 6 + [0.4] * 12,
    })
    return df


def test_autopsy_cells_and_auc():
    from stockml.evaluation.up_autopsy import confusion_cells, ovr_auc_per_class, per_class_confidence
    df = _autopsy_frame()
    cells = confusion_cells(df)
    assert cells["n_actual_up"] == 12
    assert cells["up_to_down"] == 6 and cells["up_to_up"] == 6
    auc = ovr_auc_per_class(df["actual_label"].to_numpy(), df[["p_down", "p_flat", "p_up"]].to_numpy())
    assert auc["up"] > 0.5
    conf = per_class_confidence(df["actual_label"].to_numpy(), df[["p_down", "p_flat", "p_up"]].to_numpy())
    assert conf["up"] > conf["down"]


def test_autopsy_slices_require_cross_era():
    from stockml.evaluation.up_autopsy import slice_miss_rates, quintile_slices
    df = _autopsy_frame()
    miss = (df["actual_label"] == 2) & (df["predicted_label"] != 2)
    ctx = pd.DataFrame({
        "timestamp": df["timestamp"],
        "sector": ["A"] * 6 + ["B"] * 6 + ["A"] * 12,
        "val": list(np.linspace(0, 1, 12)) + list(np.linspace(0, 1, 12)),
    })
    rows = slice_miss_rates(miss, ctx, "sector")
    assert {r["bucket"] for r in rows} == {"A", "B"}
    assert all("per_era" in r and "cross_era" in r for r in rows)
    q = quintile_slices(miss, ctx, "val")
    assert len(q) > 0 and all(r["reliable"] == (r["rows"] >= 200) for r in q)


def test_worst_misses_are_confident_up_to_down():
    from stockml.evaluation.up_autopsy import worst_misses
    df = _autopsy_frame()
    ctx = pd.DataFrame({"sector": ["X"] * 24}, index=df.index)
    w = worst_misses(df, ctx, n=3)
    assert len(w) == 3
    assert ((w["actual_label"] == 2) & (w["predicted_label"] == 0)).all()
    assert "ctx_sector" in w.columns


def test_objective_logs_per_fold_diagnostics(tmp_path):
    import optuna
    from stockml.models.lightgbm_train import objective_factory
    from types import SimpleNamespace
    rng = np.random.default_rng(3)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)], index=idx)
    y = pd.Series(rng.integers(0, 3, n), index=idx)
    events = pd.Series(idx + pd.Timedelta(days=5), index=idx)
    settings = SimpleNamespace(n_splits=2, purge_days=1, embargo_days=1,
                               class_weighting=False, random_seed=42,
                               num_boost_round=20, early_stopping_rounds=5,
                               optuna_pruner="none")
    study = optuna.create_study(direction="maximize")
    factory = objective_factory(X, y, events, settings, tmp_path)
    # Drive the trial through study.optimize for proper user-attr plumbing.
    study.optimize(factory, n_trials=1)
    t = study.trials[0]
    assert t.state.name == "COMPLETE"
    assert "fold1_recall_up" in t.user_attrs and "fold1_top15" in t.user_attrs
    diag = tmp_path / "fold_diagnostics.jsonl"
    assert diag.exists()
    lines = diag.read_text().strip().split("\n")
    assert len(lines) == 2
    import json as _json
    rec = _json.loads(lines[0])
    assert set(rec["recall"]) == {"down", "flat", "up"} and len(rec["top15"]) == 5


def test_dm_flags_identical_and_clear_winner():
    from stockml.evaluation.dm import dm_statistic, up_vs_rest_dm
    import numpy as np
    same = dm_statistic(np.zeros(100))
    assert same["p_value"] == 1.0
    # Model perfect on UP, base never predicts UP → strong rejection.
    y = np.array([2] * 60 + [0] * 60)
    pm = np.array([2] * 60 + [0] * 60)
    pb = np.array([0] * 120)
    out = up_vs_rest_dm(y, pm, pb)
    assert out["stat"] > 2.0 and out["p_value"] < 0.05
    assert out["model_up_accuracy"] > out["base_up_accuracy"]


def test_announcement_taxonomy_mapping():
    from stockml.features.announcement_features import map_event_flags
    s = pd.Series(["Acquisition of stake", "Financial Results Q3", "Analyst Meet Concall",
                   "Disclosure under SEBI Takeover Regulations", "Loss of Share Certificates"])
    out = map_event_flags(s)
    assert out["ann_event_acquisition"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert out["ann_event_earnings"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert out["ann_event_analyst"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert out["ann_event_regulatory"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_announcement_decision_date_no_double_shift():
    from stockml.features.announcement_features import aggregate_announcements
    cal = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"])
    trading_index = cal
    ann = pd.DataFrame({
        # Mon 10:00 IST (04:30 UTC) → Mon; Mon 16:00 IST (10:30 UTC) → Tue.
        "published_utc": [pd.Timestamp("2024-03-04 04:30", tz="UTC"), pd.Timestamp("2024-03-04 10:30", tz="UTC")],
        "symbol": ["X.NS", "X.NS"],
        "subject": ["Acquisition", "Updates"],
        "article_id": ["a1", "a2"],
    })
    out = aggregate_announcements(ann, trading_index, "X.NS", cal)
    assert out.loc["2024-03-04", "ann_count"] == 1.0
    assert out.loc["2024-03-05", "ann_count"] == 1.0
    assert out.loc["2024-03-05", "ann_event_acquisition"] == 0.0
    assert out.loc["2024-03-04", "ann_event_acquisition"] == 1.0
    assert out.loc["2024-03-04", "days_since_last_ann"] == 0.0
    assert out.loc["2024-03-06", "days_since_last_ann"] == 1.0
    assert out["ann_available"].eq(1.0).all()
    # Empty calendar → zeros + NaN recency (safe fallback, no crash).
    out0 = aggregate_announcements(ann, trading_index, "X.NS", pd.DatetimeIndex([]))
    assert out0["ann_count"].eq(0.0).all() and out0["days_since_last_ann"].isna().all()


def test_u2_scope_activity_excludes_self_and_tags_buckets():
    from stockml.evaluation.u2_d1 import bucket_at, scope_activity, activity_for_rows, subset_report
    assert bucket_at([("G", pd.Timestamp("2020-01-01"), pd.Timestamp.max)], pd.Timestamp("2024-01-01")) == "G"
    assert bucket_at([("G", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"))], pd.Timestamp("2024-01-01")) is None
    assert bucket_at([], pd.Timestamp("2024-01-01")) is None
    ann = pd.DataFrame({
        "symbol": ["A.NS", "B.NS", "C.NS"],
        "decision_date": pd.to_datetime(["2024-03-04", "2024-03-04", "2024-03-05"]),
    })
    sectors = {"A.NS": [("S1", pd.Timestamp("2010-01-01"), pd.Timestamp.max)],
               "B.NS": [("S1", pd.Timestamp("2010-01-01"), pd.Timestamp.max)],
               "C.NS": [("S2", pd.Timestamp("2010-01-01"), pd.Timestamp.max)]}
    groups = {"A.NS": [("G1", pd.Timestamp("2010-01-01"), pd.Timestamp.max)],
              "B.NS": [("G1", pd.Timestamp("2010-01-01"), pd.Timestamp.max)]}
    tagged = scope_activity(ann, sectors, groups)
    assert tagged["group_tag"].tolist() == ["G1", "G1", None]
    pred = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-03-04", "2024-03-04", "2024-03-05"]),
        "symbol": ["A.NS", "C.NS", "A.NS"],
        "actual_label": [2, 2, 0],
        "predicted_label": [0, 2, 0],
        "p_up": [0.3, 0.7, 0.2],
    })
    act = activity_for_rows(pred, tagged, sectors, groups)
    # A on 03-04: B (same group+sector) announced → 1 each, excl. self.
    assert act.loc[0, "group_act"] == 1.0 and act.loc[0, "sector_act"] == 1.0
    # C on 03-04: no group, sector S2 alone → 0/0; market counts all that day.
    assert act.loc[1, "group_act"] == 0.0 and act.loc[1, "sector_act"] == 0.0
    assert act.loc[1, "market_act"] == 2.0
    is_up = pred["actual_label"] == 2
    rep = subset_report("t", act["group_act"] > 0, is_up, is_up & (pred["predicted_label"] != 2),
                        pred["p_up"], pd.Series(["e1"] * 3, index=pred.index))
    assert rep["t"]["rows"] == 1 and not rep["t"]["reliable"]


def test_u3_sentiment_buckets_and_context():
    from stockml.evaluation.u3_d1 import sign_bucket, sentiment_context
    assert sign_bucket(0.5) == "positive" and sign_bucket(-0.5) == "negative"
    assert sign_bucket(0.0) == "neutral" and sign_bucket(float("nan")) == "none"
    cal = pd.DatetimeIndex(["2024-03-04", "2024-03-05"])
    pred = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-03-04", "2024-03-05"]),
        "symbol": ["A.NS", "A.NS"],
    })
    ann = pd.DataFrame({
        "symbol": ["A.NS", "A.NS"],
        "published_utc": [pd.Timestamp("2024-03-04 04:30", tz="UTC"), pd.Timestamp("2024-03-04 10:30", tz="UTC")],
        "finbert_score": [0.9, -0.4],
    })
    ctx = sentiment_context(pred, ann, cal)
    # 03-04: one scored ann (0.9); 03-05: the 16:00-IST one rolls to 03-05 (-0.4).
    assert ctx.loc[0, "sent_count"] == 1 and ctx.loc[0, "sent_mean"] == 0.9
    assert ctx.loc[1, "sent_count"] == 1 and ctx.loc[1, "sent_mean"] == -0.4


def test_cap_trades_per_day_keeps_top_confidence():
    from stockml.evaluation.economics import cap_trades_per_day
    pred = pd.DataFrame({
        "event_start": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"]),
        "confidence": [0.6, 0.8, 0.7, 0.61],
        "v": [1, 2, 3, 4],
    })
    out = cap_trades_per_day(pred, 2)
    assert sorted(out["v"].tolist()) == [2, 3, 4]
    out1 = cap_trades_per_day(pred, 1)
    assert sorted(out1["v"].tolist()) == [2, 4]


def test_pbo_flags_obvious_overfit_and_clean():
    from stockml.evaluation.economics import cpcv_pbo
    import numpy as np
    idx = pd.date_range("2020-01-01", periods=800, freq="B")
    rng = np.random.default_rng(0)
    noise = {f"s{i}": pd.Series(rng.normal(0, 1, 800), index=idx) for i in range(6)}
    out = cpcv_pbo(noise, n_partitions=8, n_combos=100, seed=1)
    assert out["pbo"] is not None and 0.0 <= out["pbo"] <= 1.0
    assert out["n_strategies"] == 6
    assert cpcv_pbo({"only": pd.Series(rng.normal(size=800), index=idx)})["pbo"] is None


def test_fo_proxy_is_quartile_and_pit_shifted():
    from stockml.evaluation.economics import fo_proxy_eligible
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    turnover = pd.DataFrame({
        "BIG.NS": np.linspace(100, 200, 80),
        "MID.NS": np.linspace(50, 60, 80),
        "SMALL1.NS": np.linspace(1, 2, 80),
        "SMALL2.NS": np.linspace(1, 2, 80),
    }, index=idx)
    pred = pd.DataFrame({
        "timestamp": list(idx[-5:]) * 4,
        "symbol": ["BIG.NS"] * 5 + ["MID.NS"] * 5 + ["SMALL1.NS"] * 5 + ["SMALL2.NS"] * 5,
    })
    mask = fo_proxy_eligible(pred, turnover, window=60, quartile=0.75)
    # Top quartile of 4 = top 1 → BIG only (PIT trailing median preserves order).
    assert mask[pred["symbol"] == "BIG.NS"].all()
    assert not mask[pred["symbol"] != "BIG.NS"].any()


def test_touch_ledger_backfill_and_log(tmp_path):
    from stockml.evaluation.touch_ledger import backfill, log_touch, BACKFILL
    from types import SimpleNamespace
    paths = SimpleNamespace(results=tmp_path)
    es = backfill(paths)
    assert len(es) == len(BACKFILL)
    assert all(all(k in e for k in ("date", "track", "phase", "description", "rows_touched", "artifact_ref", "eligible")) for e in es)
    n0 = sum(1 for e in es if e["eligible"])
    n1 = log_touch(paths, "I0", "D1", "test touch", 100, "x.json")
    assert n1 == n0 + 1
    # Backfill is idempotent and preserves logged touches.
    es2 = backfill(paths)
    assert len(es2) == len(BACKFILL) + 1


def _grid_price(n=300, drift=0.001, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(drift, 1.0, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1_000_000,
    }, index=idx)


def test_label_grid_cells_deterministic_and_mirror_drift():
    from stockml.evaluation.label_grid import cell_stats
    frames = {"A.NS": _grid_price(drift=0.05), "B.NS": _grid_price(drift=0.05, seed=9)}
    c1 = cell_stats(frames, 5, 1.0)
    c2 = cell_stats(frames, 5, 1.0)
    assert c1 == c2  # deterministic
    shares = c1["down_share"] + c1["flat_share"] + c1["up_share"]
    assert abs(shares - 1.0) < 1e-9
    assert c1["up_share"] > c1["down_share"]  # upward drift → more UPs
    assert c1["up_ret_med"] > 0 > c1["down_ret_med"]  # mirror-image medians


def test_announcement_beyond_calendar_dropped_not_crash():
    from stockml.features.announcement_features import aggregate_announcements
    cal = pd.DatetimeIndex(["2024-03-04", "2024-03-05"])
    ann = pd.DataFrame({
        "published_utc": [pd.Timestamp("2024-03-04 04:30", tz="UTC"), pd.Timestamp("2026-09-17 08:00", tz="UTC")],
        "symbol": ["X.NS", "X.NS"],
        "subject": ["Updates", "Updates"],
        "article_id": ["a1", "a2"],
    })
    out = aggregate_announcements(ann, cal, "X.NS", cal)
    assert out.loc["2024-03-04", "ann_count"] == 1.0  # future row dropped, past kept
    assert out["ann_count_20d"].max() == 1.0


def test_trading_calendar_uses_nifty_sessions():
    from stockml.data.trading_calendar import build_calendar, load_calendar
    idx = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"])
    macro = pd.DataFrame({"nifty": [22000.0, 22100.0, float("nan"), 22200.0]}, index=idx)
    cal = build_calendar(macro)
    assert list(cal.date.astype(str)) == ["2024-03-04", "2024-03-05", "2024-03-07"]
    assert load_calendar(ref_dir="nonexistent_dir_xyz").empty


def test_parse_announcements_converts_ist_to_utc_and_skips_bad_rows():
    from stockml.data.announcements import parse_announcements
    payload = [
        {"seq_id": "1", "an_dt": "17-Sep-2026 19:50:37", "desc": "Updates",
         "attchmntText": "Investor meeting", "attchmntFile": "http://x/y.pdf"},
        {"seq_id": "", "an_dt": "garbage", "desc": "X"},  # skipped: no id
        {"seq_id": "2", "an_dt": "not-a-date", "desc": "Y"},  # skipped: bad ts
    ]
    out = parse_announcements(payload, "RELIANCE.NS")
    assert len(out) == 1
    assert out.iloc[0]["article_id"] == "nse_ann:1"
    # 19:50 IST = 14:20 UTC same day.
    assert str(out.iloc[0]["published_utc"]) == "2026-09-17 14:20:37"
    assert out.iloc[0]["symbol"] == "RELIANCE.NS"


def test_decision_date_cutoff_weekend_and_holiday():
    import pandas as pd
    from stockml.features.decision_date import decision_date
    # Calendar: Mar 2024 weekdays minus Mon 2026-01-01-like holiday case below.
    cal = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"])
    # B-05 vectors: 09:50 UTC = 15:20 IST → same day; 10:10 UTC = 15:40 IST → next.
    assert decision_date(pd.Timestamp("2024-03-05 09:50", tz="UTC"), cal) == pd.Timestamp("2024-03-05")
    assert decision_date(pd.Timestamp("2024-03-05 10:10", tz="UTC"), cal) == pd.Timestamp("2024-03-06")
    # Saturday → next Monday (calendar extended); Friday 16:00 IST (=10:30
    # UTC) before a Monday holiday → Tuesday.
    cal2 = pd.DatetimeIndex(["2024-03-08", "2024-03-11", "2024-03-12"])
    assert decision_date(pd.Timestamp("2024-03-09 05:00", tz="UTC"), cal2) == pd.Timestamp("2024-03-11")
    hol_cal = pd.DatetimeIndex(["2024-03-01", "2024-03-05", "2024-03-06"])
    assert decision_date(pd.Timestamp("2024-03-01 10:30", tz="UTC"), hol_cal) == pd.Timestamp("2024-03-05")


def test_flat_threshold_rule_and_sweep():
    from stockml.evaluation.flat_threshold import apply_flat_threshold, sweep_flat_threshold
    import numpy as np
    proba = np.array([[0.5, 0.3, 0.2], [0.3, 0.35, 0.35], [0.2, 0.5, 0.3]])
    pred = apply_flat_threshold(proba, 0.40)
    assert pred.tolist() == [0, 2, 1]  # FLAT iff p_flat>=tau else argmax(DOWN,UP)
    # tau > 1 disables FLAT (fallback to DOWN/UP); differs from argmax
    # only where argmax itself chose FLAT (tie rows).
    assert apply_flat_threshold(proba, 1.01).tolist() == [0, 2, 2]
    y = np.array([0, 2, 1, 1, 0, 2])
    p = np.array([[0.6, 0.2, 0.2], [0.3, 0.3, 0.4], [0.3, 0.5, 0.2],
                  [0.4, 0.45, 0.15], [0.35, 0.4, 0.25], [0.2, 0.3, 0.5]])
    out = sweep_flat_threshold(y, p, grid=[0.30, 0.45])[0]
    assert out["winner"]["tau_flat"] in (0.30, 0.45)
    assert out["winner"]["accuracy"] >= out["base_accuracy"] - 0.005 - 1e-9


def test_config_loads_flat_threshold():
    from stockml.config import load_settings
    settings = load_settings()
    assert settings.flat_threshold == 0.40


def test_i01_bars_and_buckets():
    from stockml.evaluation.i01_d1 import se_2sample, bucket_delta_report, earnings_bucket, days_since_last
    assert se_2sample(0.5, 1000, 0.5, 1000) < 0.05
    assert se_2sample(0.5, 5, 0.5, 5) == float("inf")
    good = {"a": {"up_rows": 500, "miss_rate": 0.8}, "b": {"up_rows": 500, "miss_rate": 0.5}}
    v = bucket_delta_report(good, "b", "a")
    assert v["pass"] and abs(v["delta"] - 0.3) < 1e-9
    thin = {"a": {"up_rows": 10, "miss_rate": 0.8}, "b": {"up_rows": 500, "miss_rate": 0.5}}
    assert not bucket_delta_report(thin, "b", "a")["pass"]
    assert earnings_bucket(__import__("numpy").array([0, 5, 6, 15, 30, float("nan")])).tolist() == ["0-5", "0-5", "6-10", "11-20", "21+", "none"]
    d = days_since_last(pd.DatetimeIndex(["2024-03-05", "2024-03-04"]), pd.DatetimeIndex(["2024-03-01"]))
    assert d.tolist() == [4.0, 3.0]
    d0 = days_since_last(pd.DatetimeIndex(["2024-03-01"]), pd.DatetimeIndex([]))
    assert d0[0] != d0[0]


def test_i1_leadlag_assigns_prior_session_regime():
    from stockml.evaluation.i01_d1 import leadlag_buckets
    idx = pd.date_range("2024-03-04", periods=6, freq="B")
    price_ret = pd.DataFrame({"LEAD.NS": [0.05, 0.05, -0.05, 0.0, 0.0, 0.0], "LAG.NS": [0.0] * 6}, index=idx)
    members = pd.DataFrame([{"symbol": "LEAD.NS", "sector": "S", "leader": True},
                            {"symbol": "LAG.NS", "sector": "S", "leader": False}])
    pred = pd.DataFrame({"timestamp": list(idx) * 2, "symbol": ["LEAD.NS"] * 6 + ["LAG.NS"] * 6})
    lab = leadlag_buckets(pred, price_ret, members)
    lag_lab = lab[pred["symbol"] == "LAG.NS"]
    assert set(lag_lab.dropna().unique()) <= {"leader-up", "leader-down", "leader-flat"}
    assert lab[pred["symbol"] == "LEAD.NS"].isna().all()


def test_parse_bulkdeals_keeps_valid_sides_only():
    from stockml.data.bulkdeals import parse_deals
    text = ("Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,Trade Price / Wght. Avg. Price\n"
            "18-SEP-2026,RELIANCE,Rel Corp,X,BUY,100,1500.0\n"
            "18-SEP-2026,INFY,Inf Corp,Y,SELL,50,1800.0\n"
            "18-SEP-2026,BAD,Bad Corp,Z,HOLD,10,5.0\n"
            "not-a-date,NOPE,No Corp,W,BUY,5,1.0\n")
    out = parse_deals(text, "bulk")
    assert len(out) == 2 and set(out["side"]) == {"BUY", "SELL"}
    assert out["source"].eq("bulk").all()
    assert parse_deals("garbage,,,", "block").empty


def test_gdelt_provider_retries_429_then_parses():
    from stockml.data.news import GDELTDocProvider
    import datetime as _dt
    calls = {"n": 0}

    class FakeResp:
        status_code = 200
        content = b'{"articles": [{"title": "T", "url": "u", "seendate": "20260919T100000", "domain": "d"}]}'

        def raise_for_status(self):
            pass

        def json(self):
            import json as _j
            return _j.loads(self.content)

    class Flaky429:
        status_code = 429
        content = b"limit requests"

        def raise_for_status(self):
            pass

    class Sess:
        def get(self, *a, **k):
            calls["n"] += 1
            return Flaky429() if calls["n"] < 3 else FakeResp()

    import stockml.data.news as _news
    orig_sleep = _news.time.sleep
    _news.time.sleep = lambda s: None
    try:
        p = GDELTDocProvider(session=Sess())
        assert p.pause_seconds >= 6.0
        out = p.fetch("X.NS", "X", _dt.datetime(2026, 9, 1), _dt.datetime(2026, 9, 19), max_records=10)
        assert len(out) == 1 and out[0].title == "T"
        assert calls["n"] == 3
    finally:
        _news.time.sleep = orig_sleep


def test_down_flag_cascade_and_economics():
    from stockml.evaluation.down_flag import apply_down_flag, sweep_down_threshold, avoided_loss_economics
    import numpy as np
    proba = np.array([[0.5, 0.3, 0.2], [0.3, 0.35, 0.35], [0.2, 0.5, 0.3], [0.65, 0.2, 0.15]])
    pred = apply_down_flag(proba, tau_flat=0.40, tau_down=0.60)
    assert pred.tolist() == [0, 2, 1, 0]  # flat-first, then down, else argmax(D,U)
    # Operative region: tau_down < 0.5 overrules UP-leaning rows.
    assert apply_down_flag(np.array([[0.35, 0.2, 0.45]]), tau_flat=0.40, tau_down=0.30).tolist() == [0]
    assert apply_down_flag(np.array([[0.35, 0.2, 0.45]]), tau_flat=0.40, tau_down=0.40).tolist() == [2]
    y = np.array([0, 0, 1, 2])
    out = sweep_down_threshold(y, proba, tau_flat=0.40, grid=[0.55, 0.70])
    assert out["winner"]["tau_down"] in (0.55, 0.70)
    econ = avoided_loss_economics(y, np.array([0, 2, 1, 0]), np.array([-0.02, 0.01, 0.0, -0.03]), exit_cost=0.00125)
    assert econ["n_flags"] == 2
    assert abs(econ["mean_net"] - ((0.02 - 0.00125) + (0.03 - 0.00125)) / 2) < 1e-9
    empty = avoided_loss_economics(y, np.array([2, 2, 1, 2]), np.array([0.0] * 4))
    assert empty["n_flags"] == 0 and empty["mean_net"] is None


def test_gdelt_bq_match_and_frame():
    from stockml.data.gdelt_bq import match_symbol, frame_from_rows, _aliases
    assert match_symbol("RELIANCE INDUSTRIES LTD", "X", ["RELIANCE INDUSTRIES", "RELIANCE"])
    assert not match_symbol("INFOSYS LTD", "Y", ["RELIANCE"])
    assert "RELIANCE" in _aliases("RELIANCE.NS", "Reliance Industries")
    rows = [
        {"Actor1Name": "RELIANCE INDUSTRIES", "Actor2Name": "X", "SQLDATE": 20240115,
         "EventCode": "093", "EventRootCode": "09", "AvgTone": -2.5, "GoldsteinScale": -4.0,
         "NumMentions": 10, "SOURCEURL": "http://x"},
        {"Actor1Name": "INFOSYS", "Actor2Name": "Y", "SQLDATE": 20240115,
         "EventCode": "010", "EventRootCode": "01", "AvgTone": 3.0, "GoldsteinScale": 4.0,
         "NumMentions": 5, "SOURCEURL": "http://y"},
        {"Actor1Name": "RELIANCE INDUSTRIES", "Actor2Name": "X", "SQLDATE": "bad",
         "EventCode": "093", "AvgTone": 0.0, "GoldsteinScale": 0.0, "NumMentions": 1, "SOURCEURL": ""},
    ]
    out = frame_from_rows(rows, "RELIANCE.NS", ["RELIANCE INDUSTRIES", "RELIANCE"])
    assert len(out) == 1  # Infosys filtered, bad date dropped
    assert out.iloc[0]["tone"] == -2.5 and out.iloc[0]["source"] == "gdelt_bq"


def test_av_quota_guard_stops_at_budget(tmp_path):
    from stockml.data.news import AlphaVantageNewsProvider, QuotaExhausted, _quota_check_and_take
    qp = tmp_path / "av_quota.json"
    _quota_check_and_take(qp, 2)
    _quota_check_and_take(qp, 2)
    try:
        _quota_check_and_take(qp, 2)
        assert False, "should have raised"
    except QuotaExhausted:
        pass
    # Rollover: yesterday's state does not block today.
    import json as _j, datetime as _d
    qp.write_text(_j.dumps({"date": "2000-01-01", "used": 99}))
    _quota_check_and_take(qp, 2)  # no raise
    p = AlphaVantageNewsProvider("k", quota_path=qp, daily_budget=500)
    assert p.daily_budget == 500
    try:
        AlphaVantageNewsProvider("")
        assert False, "should have raised"
    except ValueError:
        pass


def test_rss_normalize_keeps_valid_entries_only():
    from stockml.data.rss import normalize_entries
    import time as _t
    entries = [
        {"title": "Sensex rallies", "link": "http://x/1", "summary": "s",
         "published_parsed": _t.gmtime(1726700000), "updated_parsed": None,
         "published": None, "updated": None},
        {"title": "", "link": "http://x/2", "summary": "no title"},
        {"title": "No date item", "link": "http://x/3", "summary": "s"},
    ]
    out = normalize_entries(entries, "ET-Markets")
    assert len(out) == 1 and out.iloc[0]["title"] == "Sensex rallies"
    assert out.iloc[0]["provider"] == "rss" and out.iloc[0]["symbol"] == ""
    assert normalize_entries([], "X").empty


def test_join_keeps_all_stock_rows_and_no_overlap():
    from stockml.features.join import join_symbol_features
    idx = pd.date_range("2024-03-04", periods=4, freq="B")
    stocks = pd.DataFrame({"ret_5": [0.1, 0.2, 0.3, 0.4], "shared": [1, 1, 1, 1]}, index=idx)
    news = pd.DataFrame({"news_co_count": [0, 2, 0, 1], "shared": [9, 9, 9, 9]}, index=idx)
    market = pd.DataFrame({"news_mkt_count": [5, 5, 5, 5]}, index=idx)
    out = join_symbol_features(stocks, news, market)
    assert len(out) == 4 and list(out.index) == list(idx)
    assert out["shared"].eq(1.0).all()  # stocks win overlaps
    assert out["news_co_count"].tolist() == [0, 2, 0, 1]
    assert out["news_mkt_count"].eq(5.0).all()
    assert join_symbol_features(stocks, None, pd.DataFrame()).equals(stocks)


def test_n6_text_aggregation_by_decision_date():
    from stockml.features.news_context import aggregate_scored_text
    cal = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"])
    frame = pd.DataFrame({
        "symbol": ["X.NS", "X.NS", "Y.NS"],
        # Mon 10:00 IST → Mon; Mon 16:00 IST → Tue.
        "published_utc": [pd.Timestamp("2024-03-04 04:30", tz="UTC"), pd.Timestamp("2024-03-04 10:30", tz="UTC"), pd.Timestamp("2024-03-04 04:30", tz="UTC")],
        "finbert_score": [0.8, -0.6, 0.1],
    })
    out = aggregate_scored_text(frame, cal, "X.NS", cal, "news_co")
    assert out.loc["2024-03-04", "news_co_count"] == 1.0
    assert out.loc["2024-03-04", "news_co_sent_mean"] == 0.8
    assert out.loc["2024-03-05", "news_co_count"] == 1.0
    assert out.loc["2024-03-06", "news_co_count"] == 0.0
    assert out.loc["2024-03-06", "news_co_sent_mean"] != out.loc["2024-03-06", "news_co_sent_mean"]  # NaN, not 0
    assert out["news_co_available"].eq(1.0).all()
    empty = aggregate_scored_text(pd.DataFrame(), cal, "X.NS", cal, "news_co")
    assert empty["news_co_count"].eq(0.0).all() and empty["news_co_available"].eq(0.0).all()


def test_n6_event_and_market_aggregates():
    from stockml.features.news_context import aggregate_event_tone, aggregate_market
    cal = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06"])
    ev = pd.DataFrame({
        "symbol": ["X.NS", "X.NS"],
        "published_utc": pd.to_datetime(["2024-03-04", "2024-03-05"]),
        "tone": [-3.0, 2.0],
    })
    out = aggregate_event_tone(ev, cal, "X.NS", cal, "ev_co")
    assert out.loc["2024-03-04", "ev_co_count"] == 1.0
    assert out.loc["2024-03-04", "ev_co_tone_mean"] == -3.0
    assert out.loc["2024-03-06", "ev_co_count"] == 0.0
    ann = pd.DataFrame({
        "symbol": ["A.NS", "B.NS"],
        "published_utc": [pd.Timestamp("2024-03-04 04:30", tz="UTC"), pd.Timestamp("2024-03-04 04:30", tz="UTC")],
        "subject": ["Updates", "Updates"],
        "article_id": ["a1", "a2"],
    })
    mkt = aggregate_market(ann, pd.DataFrame(), cal, cal)
    assert mkt.loc["2024-03-04", "ann_mkt_count"] == 2.0
    assert mkt.loc["2024-03-05", "ann_mkt_count"] == 0.0
    assert list(mkt.columns) == ["news_mkt_count", "news_mkt_sent_mean", "news_mkt_count_z20", "ann_mkt_count"]

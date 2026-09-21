from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import load_settings
from ..risk.portfolio import confidence_position_size, enforce_gross_exposure
from ..utils.io import atomic_write_json, atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state


def round_trip_cost_rate(settings) -> tuple[float, dict]:
    """B-04/F-09: segment-aware Indian round-trip cost per unit weight.

    Delivery: 2×(commission+slippage+other) + 2×10 STT (both sides, §6.3)
              + 1.5 stamp duty (buy side). With the default 1+3+1 inputs this is
              31.5 bps — inside the realistic 25–35 bps delivery band.
    Futures:  2×(commission+slippage+other) + 5 STT (sell side only).
    """
    base = (settings.commission_bps + settings.slippage_bps + settings.other_cost_bps) / 10000.0
    segment = str(getattr(settings, "backtest_segment", "delivery")).lower()
    if segment == "futures":
        total = 2.0 * base + 0.0005
    else:
        segment = "delivery"
        total = 2.0 * base + 2 * 0.0010 + 0.00015
    return total, {"segment": segment, "round_trip_bps": total * 10000.0}


def _event_backtest(pred: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    p = pred.copy()
    if not isinstance(p.index, pd.MultiIndex):
        p.index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(p["timestamp"]), p["symbol"]], names=["timestamp", "symbol"]
        )

    required = {"event_start", "event_end", "entry_price", "exit_price", "p_up", "p_down"}
    missing = required.difference(p.columns)
    if missing:
        raise ValueError(f"Predictions are missing backtest fields: {sorted(missing)}")

    p = p.reset_index(drop=True)
    p["timestamp"] = pd.to_datetime(p["timestamp"])
    p["event_start"] = pd.to_datetime(p["event_start"])
    p["event_end"] = pd.to_datetime(p["event_end"])
    p["position"] = [
        confidence_position_size(
            float(up), float(down),
            threshold=settings.backtest_confidence_threshold,
            max_position=1.0,
        )
        for up, down in zip(p["p_up"], p["p_down"])
    ]
    p["entry_price"] = pd.to_numeric(p["entry_price"], errors="coerce")
    p["exit_price"] = pd.to_numeric(p["exit_price"], errors="coerce")
    p["realized_return"] = p["exit_price"] / p["entry_price"] - 1.0

    # Do not allow overlapping positions for the same symbol. This makes the
    # event backtest consistent with the triple-barrier holding period and avoids
    # counting several copies of the same capital at once.
    accepted = np.zeros(len(p), dtype=bool)
    active_until: dict[str, pd.Timestamp] = {}
    for i, row in p.sort_values(["symbol", "timestamp"]).iterrows():
        symbol = str(row["symbol"])
        if abs(float(row["position"])) <= 0:
            continue
        start = row["event_start"]
        end = row["event_end"]
        if pd.isna(start) or pd.isna(end) or pd.isna(row["realized_return"]):
            continue
        if symbol not in active_until or start > active_until[symbol]:
            accepted[i] = True
            active_until[symbol] = end

    p["accepted"] = accepted
    p["gross_signal"] = np.where(p["accepted"], p["position"], 0.0)

    # Normalize each entry day's gross exposure across accepted signals.
    entry_weights = []
    for _, group in p[p["accepted"]].groupby("event_start"):
        weights = enforce_gross_exposure(group["gross_signal"].copy(), settings.max_gross_exposure)
        entry_weights.extend([(idx, float(weight)) for idx, weight in weights.items()])
    p["weight"] = 0.0
    for idx, value in entry_weights:
        p.loc[idx, "weight"] = value

    total_cost_rate, cost_info = round_trip_cost_rate(settings)
    p["gross_return"] = p["weight"] * p["realized_return"].fillna(0.0)
    # Round-trip transaction cost, applied once per accepted (non-overlapping) event.
    p["cost"] = np.where(p["accepted"], p["weight"].abs() * total_cost_rate, 0.0)
    p["net_return"] = p["gross_return"] - p["cost"]

    daily = p.groupby("event_end", as_index=True)["net_return"].sum().sort_index()
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    metrics = {
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe": sharpe,
        "trade_count": int(p["accepted"].sum()),
        "gross_return_sum": float(p["gross_return"].sum()),
        "cost_paid": float(p["cost"].sum()),
        "net_return_sum": float(p["net_return"].sum()),
        "cost_bps_round_trip": float(cost_info["round_trip_bps"]),
        "segment": cost_info["segment"],
        "overlap_policy": "No overlapping event positions per symbol",
    }
    return p, {"metrics": metrics, "equity": equity, "drawdown": drawdown}


def run_backtest() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "backtest.log")
    pred_path = paths.results / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError("predictions.parquet not found. Run evaluation first.")

    pred = pd.read_parquet(pred_path).copy()
    pred, out = _event_backtest(pred, settings)
    atomic_write_parquet(pred, paths.results / "backtest_timeseries.parquet")
    atomic_write_json(paths.results / "backtest_metrics.json", out["metrics"])

    plt.figure(figsize=(16, 5))
    plt.plot(out["equity"].index, out["equity"].to_numpy())
    plt.title("Net event backtest equity curve")
    plt.xlabel("Exit date")
    plt.ylabel("Equity (start=1.0)")
    plt.tight_layout()
    plt.savefig(paths.results / "equity_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(16, 4))
    plt.plot(out["drawdown"].index, out["drawdown"].to_numpy())
    plt.title("Backtest drawdown")
    plt.xlabel("Exit date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(paths.results / "drawdown.png", dpi=160)
    plt.close()

    logger.info("Backtest net return %.2f%%, max drawdown %.2f%%", out["metrics"]["total_return"] * 100, out["metrics"]["max_drawdown"] * 100)
    update_state(paths.state, "backtest", status="complete", metrics=str(paths.results / "backtest_metrics.json"))
    # F-22: full pipeline ends with the complete §12.2 block (now with economics).
    from ..evaluation.final_report import print_final_report as _print_final
    _print_final()


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run_backtest()

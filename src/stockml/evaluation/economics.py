from __future__ import annotations

"""G3 economics variants + campaign PBO + holdout-touch ledger.

All variants are backtest-side recomputations on FROZEN predictions — no
retraining, no feature changes. Each variant is one variable:
  V0        baseline (thr 0.60, delivery 31.5bps, no cap)
  thr065/070 confidence threshold override (tuned on CALIBRATION backtest)
  cap10/cap5  max accepted trades per event_start day by confidence
  fo_*      F&O subset + futures costs (needs official list; separate step)
  best      winner composed once, judged once

Kill inputs: net Sharpe, DM vs V0 (paired daily-net, Newey-West), campaign
PBO (F-15), holdout touch ledger (results/holdout_touches.jsonl).
"""

import argparse
import copy
import datetime as _dt
import json

import numpy as np
import pandas as pd

from ..backtest.backtest import _event_backtest
from ..config import load_settings
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .dm import dm_statistic

LEDGER = "holdout_touches.jsonl"

# Historical sealed-test touches, seeded from logs/evaluate.log EDGE lines.
# Ineligible touches stay listed (visible, never silently dropped).
HISTORICAL_TOUCHES = [
    {"date": "2026-09-15", "stage": "week2", "edge_pp": -1.45, "eligible": True},
    {"date": "2026-09-15", "stage": "horizon10", "edge_pp": -2.49, "eligible": True},
    {"date": "2026-09-16", "stage": "horizon10-revert", "edge_pp": -1.45, "eligible": True},
    {"date": "2026-09-18", "stage": "buggy-mask-16k", "edge_pp": -3.07, "eligible": False, "note": "wrong test slice"},
    {"date": "2026-09-18", "stage": "attempt2", "edge_pp": -1.24, "eligible": True},
    {"date": "2026-09-19", "stage": "H1", "edge_pp": -1.29, "eligible": True},
    {"date": "2026-09-19", "stage": "H1b-flagged", "edge_pp": -0.80, "eligible": False, "note": "memorization flag"},
    {"date": "2026-09-19", "stage": "attempt2-restored", "edge_pp": -1.24, "eligible": True},
    {"date": "2026-09-19", "stage": "F-adopted", "edge_pp": -0.71, "eligible": True},
    {"date": "2026-09-19", "stage": "U-ablation", "edge_pp": -0.97, "eligible": True},
]


def append_touch(paths, stage: str, edge_pp: float | None, eligible: bool = True, note: str = "") -> int:
    """Append one holdout touch. Returns total eligible count (campaign N)."""
    line = {"date": _dt.date.today().isoformat(), "stage": stage,
            "edge_pp": edge_pp, "eligible": eligible, "note": note}
    path = paths.results / LEDGER
    existing = []
    if path.exists():
        try:
            existing = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            existing = []
    if not existing:
        existing = list(HISTORICAL_TOUCHES)
    existing.append(line)
    path.write_text("\n".join(json.dumps(e) for e in existing) + "\n", encoding="utf-8")
    return sum(1 for e in existing if e.get("eligible"))


def cap_trades_per_day(pred: pd.DataFrame, cap: int, conf_col: str = "confidence") -> pd.DataFrame:
    """Keep top-`cap` candidate signals per event_start day by confidence. Pure."""
    p = pred.copy()
    if conf_col not in p.columns and {"p_down", "p_flat", "p_up"} <= set(p.columns):
        p[conf_col] = p[["p_down", "p_flat", "p_up"]].max(axis=1)
    keep = set()
    for _, group in p.groupby("event_start"):
        keep.update(group.nlargest(min(cap, len(group)), conf_col).index)
    return p.loc[sorted(keep)].copy()


def fo_proxy_eligible(pred: pd.DataFrame, turnover: pd.DataFrame,
                      window: int = 60, quartile: float = 0.75) -> pd.Series:
    """Turnover-quartile liquid-subset proxy (PROXY, not F&O membership).

    Eligible iff the symbol's trailing-`window` median turnover_cr_proxy
    (strictly before decision day t — PIT shift) ranks at/above the
    universe `quartile` that day. Pure. Pre-registered use: a FAIL here is
    kill-confirming; a PASS is flagged non-authoritative pending a real
    F&O list, never a go signal.
    """
    med = turnover.rolling(window, min_periods=20).median().shift(1)
    cutoff = med.quantile(quartile, axis=1)
    flag = med.ge(cutoff, axis=0)
    ts = pd.to_datetime(pred["timestamp"]).dt.normalize()
    syms = pred["symbol"].astype(str)
    out = []
    for t, s in zip(ts, syms):
        try:
            out.append(bool(flag.loc[t, s]))
        except KeyError:
            out.append(False)
    return pd.Series(out, index=pred.index)


def load_turnover_panel(paths, symbols: list[str]) -> pd.DataFrame:
    """turnover_cr_proxy panel (sessions × symbols) from feature parquets."""
    cols = {}
    for symbol in symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = paths.data_features / f"{safe}_features.parquet"
        if not path.exists():
            continue
        feat = pd.read_parquet(path)
        if "turnover_cr_proxy" in feat.columns:
            cols[symbol] = pd.to_numeric(feat["turnover_cr_proxy"], errors="coerce")
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols).sort_index()
    panel.index = pd.to_datetime(panel.index).tz_localize(None).normalize()
    return panel


def run_variant(pred: pd.DataFrame, settings, name: str, **overrides) -> dict:
    """One backtest variant: settings overrides and/or row pre-filter. Pure core."""
    import dataclasses
    row_mask = overrides.pop("row_mask", None)
    proxy_note = overrides.pop("proxy_note", "")
    variant_settings = dataclasses.replace(settings, **{k: v for k, v in overrides.items()
                                                        if k in settings.__dataclass_fields__})
    frame = pred
    if row_mask is not None:
        frame = pred.loc[pd.Series(row_mask, index=pred.index).fillna(False).astype(bool)].copy()
    cap = overrides.get("trades_cap")
    if cap:
        frame = cap_trades_per_day(frame, int(cap))
    _, out = _event_backtest(frame, variant_settings)
    m = out["metrics"]
    if proxy_note:
        m["proxy_note"] = proxy_note
    daily = pd.Series(dtype=float)
    if len(out["equity"]):
        daily = out["equity"].pct_change().fillna(0.0)
    return {"name": name, "metrics": m, "daily": daily}


def dm_vs_baseline(daily_var: pd.Series, daily_base: pd.Series, lags: int = 5) -> dict:
    """Paired DM on daily net returns (variant minus baseline). Pure."""
    idx = daily_var.index.union(daily_base.index).sort_values()
    d = (daily_var.reindex(idx).fillna(0.0) - daily_base.reindex(idx).fillna(0.0)).to_numpy()
    return dm_statistic(d, lags=lags)


def cpcv_pbo(strategy_daily: dict[str, pd.Series], n_partitions: int = 16,
             n_combos: int = 500, seed: int = 42) -> dict:
    """F-15: CPCV + PBO over strategy daily-net series. Pure computation.

    Partitions the common timeline into S contiguous blocks; samples
    train/test block splits; IS-Sharpe ranks strategies per split; PBO =
    fraction of splits where the IS-best strategy's OOS rank is below median.
    """
    rng = np.random.default_rng(seed)
    names = list(strategy_daily.keys())
    idx = None
    for s in strategy_daily.values():
        idx = s.index.union(idx) if idx is not None else s.index
    idx = idx.sort_values()
    mat = np.column_stack([strategy_daily[n].reindex(idx).fillna(0.0).to_numpy() for n in names])
    S = min(n_partitions, len(idx) // 20)
    if S < 4 or len(names) < 2:
        return {"pbo": None, "note": "insufficient data", "n_strategies": len(names), "partitions": S}
    bounds = np.linspace(0, len(idx), S + 1, dtype=int)
    # Sharpe per strategy per partition.
    part_sharpe = np.zeros((S, len(names)))
    for s in range(S):
        block = mat[bounds[s]:bounds[s + 1]]
        mu, sd = block.mean(axis=0), block.std(axis=0)
        part_sharpe[s] = np.where(sd > 0, mu / sd, 0.0)
    n_half = S // 2
    oos_ranks = []
    for _ in range(n_combos):
        test_blocks = np.sort(rng.choice(S, n_half, replace=False))
        train_blocks = np.array([b for b in range(S) if b not in set(test_blocks)])
        is_rank = np.argsort(np.argsort(part_sharpe[train_blocks].mean(axis=0)))
        oos = part_sharpe[test_blocks].mean(axis=0)
        oos_rank = np.argsort(np.argsort(oos))
        best_is = int(np.argmax(is_rank))
        oos_ranks.append(float(oos_rank[best_is]) / max(len(names) - 1, 1))
    pbo = float(np.mean([r < 0.5 for r in oos_ranks]))
    return {"pbo": pbo, "n_strategies": len(names), "partitions": S,
            "combos": n_combos, "median_oos_rank_of_is_best": float(np.median(oos_ranks))}


def run_economics() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "economics.log")
    pred = pd.read_parquet(paths.results / "predictions.parquet")
    if "confidence" not in pred.columns:
        pred["confidence"] = pred[["p_down", "p_flat", "p_up"]].max(axis=1)

    variants = {"V0": {}}
    for thr in (0.65, 0.70):
        variants[f"thr{int(thr * 100):d}"] = {"backtest_confidence_threshold": thr}
    for cap in (10, 5):
        variants[f"cap{cap}"] = {"trades_cap": cap}
    # G3c-proxy (kill-confirming only): liquid-subset restriction + futures
    # costs. turnover-quartile proxy, explicitly NOT F&O membership.
    try:
        _panel = load_turnover_panel(paths, list(settings.symbols))
        _elig = fo_proxy_eligible(pred, _panel)
        variants["fo_proxy"] = {"backtest_segment": "futures",
                                "row_mask": _elig,
                                "proxy_note": "turnover-quartile proxy; non-authoritative"}
        logger.info("F&O proxy eligible rows: %d/%d.", int(_elig.sum()), len(_elig))
    except Exception as exc:
        logger.warning("F&O proxy skipped: %s", exc)
    results: dict[str, dict] = {}
    for name, ov in variants.items():
        results[name] = run_variant(pred, settings, name, **ov)
        m = results[name]["metrics"]
        logger.info("%s: sharpe %.2f net %.2f%% trades %d.", name, m["sharpe"],
                    m["total_return"] * 100, m["trade_count"])
    base_daily = results["V0"]["daily"]
    for name in results:
        if name == "V0":
            continue
        results[name]["dm_vs_V0"] = dm_vs_baseline(results[name]["daily"], base_daily)
    pbo = cpcv_pbo({n: r["daily"] for n, r in results.items()})
    out = {n: {"metrics": r["metrics"], "dm_vs_V0": r.get("dm_vs_V0")} for n, r in results.items()}
    out["_pbo"] = pbo
    atomic_write_json(paths.results / "economics.json", out)
    update_state(paths.state, "economics", status="complete", report=str(paths.results / "economics.json"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="G3 economics variants + PBO (backtest-side only).")
    parser.parse_args()
    run_economics()


if __name__ == "__main__":
    main()

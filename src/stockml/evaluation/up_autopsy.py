from __future__ import annotations

"""Issue #1 diagnosis: UP→DOWN autopsy (read-only over frozen artifacts).

Reads results/predictions.parquet + per-symbol feature parquets (context
only) + data/reference maps. Writes results/up_autopsy.json and
results/worst_20.csv. Never touches the model, config, or training data.

Anti-memorization rule: a slice pattern counts as a hypothesis ONLY if it
holds across multiple time eras (see era_split) — a pattern from one era is
noise, and chasing it teaches memorization, not learning.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import load_settings
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state

UP, DOWN = 2, 0
N_WORST = 20
MIN_SLICE_ROWS = 200


def confusion_cells(df: pd.DataFrame) -> dict:
    """Counts for the UP row: UP→DOWN / UP→FLAT / UP→UP. Pure."""
    up = df[df["actual_label"] == UP]
    return {
        "n_actual_up": int(len(up)),
        "up_to_down": int((up["predicted_label"] == DOWN).sum()),
        "up_to_flat": int((up["predicted_label"] == 1).sum()),
        "up_to_up": int((up["predicted_label"] == UP).sum()),
        "mean_p_up_on_missed": float(up.loc[up["predicted_label"] != UP, "p_up"].mean()) if (up["predicted_label"] != UP).any() else None,
        "mean_conf_on_missed": float(up.loc[up["predicted_label"] != UP, ["p_down", "p_flat", "p_up"]].max(axis=1).mean()) if (up["predicted_label"] != UP).any() else None,
    }


def era_split(df: pd.DataFrame, n_eras: int = 4) -> pd.Series:
    """Era labels by timestamp quartile. Pure. Patterns must repeat across eras."""
    ts = pd.to_datetime(df["timestamp"])
    return pd.qcut(ts.astype("int64"), n_eras, labels=[f"era{i + 1}" for i in range(n_eras)])


def slice_miss_rates(
    misses: pd.DataFrame,
    context: pd.DataFrame,
    col: str,
    by_era: bool = True,
) -> list[dict]:
    """UP-miss rate per bucket of a context column, overall + per era. Pure.

    misses: boolean-indexable frame aligned with context (same index).
    Returns buckets with miss rate, rows, and per-era miss rates so reviewers
    can reject single-era patterns (memorization bait).
    """
    work = pd.DataFrame({"miss": misses.astype(bool), "bucket": context[col], "era": era_split(context)})
    rows: list[dict] = []
    for bucket, g in work.groupby("bucket", observed=False, dropna=False):
        entry: dict = {
            "bucket": None if pd.isna(bucket) else str(bucket),
            "rows": int(len(g)),
            "miss_rate": float(g["miss"].mean()),
            "reliable": bool(len(g) >= MIN_SLICE_ROWS),
        }
        if by_era:
            entry["per_era"] = {
                str(e): {"rows": int(len(gg)), "miss_rate": float(gg["miss"].mean())}
                for e, gg in g.groupby("era", observed=False)
            }
            era_rates = [v["miss_rate"] for v in entry["per_era"].values() if v["rows"] >= MIN_SLICE_ROWS // 4]
            entry["cross_era"] = bool(len(era_rates) >= 3 and (max(era_rates) - min(era_rates)) < 0.25)
        rows.append(entry)
    return sorted(rows, key=lambda r: r["miss_rate"], reverse=True)


def quintile_slices(misses: pd.DataFrame, context: pd.DataFrame, col: str) -> list[dict]:
    """Same as slice_miss_rates with quintile buckets for numeric columns. Pure."""
    vals = pd.to_numeric(context[col], errors="coerce")
    try:
        buckets = pd.qcut(vals, 5, labels=["q1", "q2", "q3", "q4", "q5"], duplicates="drop")
    except ValueError:
        return []
    return slice_miss_rates(misses, context.assign(**{col: buckets}), col)


def ovr_auc_per_class(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """One-vs-rest AUC per class. Pure. UP-AUC≈0.5 with DOWN-AUC fine ⇒ no UP concept."""
    out = {}
    for c, name in ((0, "down"), (1, "flat"), (2, "up")):
        try:
            out[name] = float(roc_auc_score((np.asarray(y_true) == c).astype(int), np.asarray(proba)[:, c]))
        except ValueError:
            out[name] = None
    return out


def per_class_confidence(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Mean predicted prob of each class on rows where it is actual. Pure."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    out = {}
    for c, name in ((0, "down"), (1, "flat"), (2, "up")):
        mask = y_true == c
        out[name] = float(proba[mask, c].mean()) if mask.any() else None
    return out


def worst_misses(df: pd.DataFrame, context: pd.DataFrame, n: int = N_WORST) -> pd.DataFrame:
    """Highest-confidence UP→DOWN rows with context columns attached. Pure."""
    miss = df[(df["actual_label"] == UP) & (df["predicted_label"] == DOWN)].copy()
    miss["confidence"] = miss[["p_down", "p_flat", "p_up"]].max(axis=1)
    miss = miss.nlargest(n, "confidence")
    for col in context.columns:
        miss[f"ctx_{col}"] = context.reindex(miss.index)[col].to_numpy()
    return miss


def _load_context(settings, pred: pd.DataFrame) -> pd.DataFrame:
    """Context columns for test rows: sector/group (maps, point-in-time) +
    delivery/turnover/vol/regime (feature parquets). Read-only."""
    ref_dir = settings.paths.data_raw.parent / "reference"
    sector_of: dict[str, list] = {}
    try:
        sm = json.loads((ref_dir / "sector_map.json").read_text(encoding="utf-8"))
        for sym, spec in sm.items():
            if str(sym).startswith("_") or not isinstance(spec, dict):
                continue
            sector_of[sym] = (str(spec.get("sector", "Unknown")), pd.Timestamp(spec.get("from") or "2010-01-01"))
    except FileNotFoundError:
        pass
    group_of: dict[str, list] = {}
    try:
        gm = json.loads((ref_dir / "group_map.json").read_text(encoding="utf-8"))
        for gid, gspec in gm.items():
            if str(gid).startswith("_") or not isinstance(gspec, dict):
                continue
            for m in gspec.get("members", []):
                group_of.setdefault(m.get("symbol"), []).append(
                    (str(gid), pd.Timestamp(m.get("from") or "2010-01-01"),
                     pd.Timestamp(m.get("to")) if m.get("to") else pd.Timestamp.max))
    except FileNotFoundError:
        pass

    want = ["delivery_per", "turnover_proxy_z20", "volatility_20", "regime_bull",
            "ret_20_minus_sector", "ret_20_minus_group", "delivery_z20"]
    per_symbol: dict[str, pd.DataFrame] = {}
    for sym in pred["symbol"].unique():
        safe = sym.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = settings.paths.data_features / f"{safe}_features.parquet"
        if not path.exists():
            continue
        feat = pd.read_parquet(path)
        cols = [c for c in want if c in feat.columns]
        per_symbol[sym] = feat[cols]

    ctx = pd.DataFrame(index=pred.index)
    ctx["timestamp"] = pd.to_datetime(pred["timestamp"]).to_numpy()
    sectors, groups = [], []
    for _, row in pred[["timestamp", "symbol"]].iterrows():
        ts, sym = pd.Timestamp(row["timestamp"]), row["symbol"]
        spec = sector_of.get(sym)
        sectors.append(spec[0] if spec and ts >= spec[1] else "Unknown")
        g = "—"
        for gid, fr, to in group_of.get(sym, []):
            if fr <= ts <= to:
                g = gid
                break
        groups.append(g)
    ctx["sector"] = sectors
    ctx["group"] = groups
    for col in want:
        vals = []
        for idx, row in pred[["timestamp", "symbol"]].iterrows():
            f = per_symbol.get(row["symbol"])
            v = np.nan
            if f is not None and col in f.columns:
                try:
                    v = float(f.reindex(pd.DatetimeIndex([pd.Timestamp(row["timestamp"])]))[col].iloc[0])
                except Exception:
                    v = np.nan
            vals.append(v)
        ctx[col] = vals
    return ctx


def run_up_autopsy() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "up_autopsy.log")
    pred_path = paths.results / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError("Run evaluate first: results/predictions.parquet is missing.")
    pred = pd.read_parquet(pred_path)
    y_true = pred["actual_label"].to_numpy()
    proba = pred[["p_down", "p_flat", "p_up"]].to_numpy()

    ctx = _load_context(settings, pred)
    is_up = pred["actual_label"] == UP
    miss_up = is_up & (pred["predicted_label"] != UP)

    result: dict = {
        "n_test": int(len(pred)),
        "cells": confusion_cells(pred),
        "ovr_auc": ovr_auc_per_class(y_true, proba),
        "mean_prob_on_actual": per_class_confidence(y_true, proba),
        "up_miss_rate_overall": float(miss_up[is_up].mean()) if is_up.any() else None,
        "slices": {
            "sector": slice_miss_rates(miss_up[is_up], ctx[is_up], "sector"),
            "group": slice_miss_rates(miss_up[is_up], ctx[is_up], "group"),
            "regime_bull": slice_miss_rates(miss_up[is_up], ctx[is_up], "regime_bull"),
        },
        "quintiles": {
            "delivery_per": quintile_slices(miss_up[is_up], ctx[is_up], "delivery_per"),
            "turnover_proxy_z20": quintile_slices(miss_up[is_up], ctx[is_up], "turnover_proxy_z20"),
            "volatility_20": quintile_slices(miss_up[is_up], ctx[is_up], "volatility_20"),
        },
    }
    worst = worst_misses(pred, ctx, N_WORST)
    worst_path = paths.results / "worst_20.csv"
    worst.to_csv(worst_path)

    atomic_write_json(paths.results / "up_autopsy.json", result)
    update_state(paths.state, "up_autopsy", status="complete", report=str(paths.results / "up_autopsy.json"))
    logger.info("UP autopsy: %d actual UPs, miss rate %.3f, OvR AUC up=%.4f down=%.4f.",
                result["cells"]["n_actual_up"], result["up_miss_rate_overall"] or 0.0,
                result["ovr_auc"].get("up") or 0.0, result["ovr_auc"].get("down") or 0.0)
    logger.info("Worst-20 written to %s.", worst_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #1 diagnosis: UP→DOWN autopsy over frozen predictions.")
    parser.parse_args()
    run_up_autopsy()


if __name__ == "__main__":
    main()

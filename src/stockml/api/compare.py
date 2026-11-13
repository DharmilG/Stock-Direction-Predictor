"""Paired head-to-head between two registered models.

Everything here is computed from the two ``predictions.parquet`` frames joined on
``(timestamp, symbol)``, never from a hand-written report. That is what keeps the
comparison generic over ``(a, b)`` and honest if either model is rerun: the page
can only ever show the delta the rows support.

The two test windows are **not** the same length (the incumbent scored 37,292
rows, v2 37,579). So every response carries both populations:

- ``windows``  — each model's own headline numbers, from its own frame;
- ``scorecard`` — ``a``/``b`` on those full windows, next to a ``delta`` computed
  on the **shared** rows only, with a date-clustered interval.

Fifty names on the same day share a market move, so rows are not independent.
Every interval here resamples **dates**, not rows: the per-date sufficient
statistics are built once, and a replicate is a sum over resampled dates. That
is exact for accuracy, recall, MCC, log loss and ECE, and it is the reason this
route can afford 1,000 replicates.

AUC is the one exception. It depends on the ranking, not on counts, so it cannot
be built from per-date sums; those axes are reported on each model's own window
with ``ci: null`` and ``significant: null`` rather than with an interval that
would be a fiction.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from ..evaluation.paired import mcnemar
from .dashboard_support import json_file, json_safe, now_ist, to_iso
from .models import DEFAULT_MODEL, ModelSpec, registry, resolve

router = APIRouter(prefix="/dashboard/compare", tags=["compare"])

ABSTAIN = -1
CLASSES = (0, 1, 2)
CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}
ECE_BINS = 10
SUMMARY_REPS = 1000
# Segments get a coarser interval on purpose: fifty symbols x 1,000 replicates is
# a lot of work for a column a reader scans in two seconds, and 400 already
# resolves a two-point gap at this sample size.
SEGMENT_REPS = 400
SEGMENT_KEYS = ("class", "symbol", "month", "confidence", "agreement", "volatility")
MIN_SEGMENT_ROWS = 100
MIN_SEGMENT_DATES = 20
TOP_K = 15

CONTRACT: dict[str, dict[str, str]] = {
    "a": {"denominator": "model a on every row it scored", "decision": "The reference model's own headline."},
    "b": {"denominator": "model b on every row it scored", "decision": "The challenger's own headline."},
    "delta": {
        "denominator": "pooled statistic on the rows both models scored, b minus a",
        "decision": "Whether b improved on a where the two can be compared at all.",
    },
    "ci": {
        "denominator": "date-clustered bootstrap, 1,000 replicates over dates",
        "decision": "Whether the delta survives correlation inside a trading day.",
    },
    "significant": {
        "denominator": "interval excludes zero",
        "decision": "Whether the difference may be reported as a difference.",
    },
    "paired_rows": {
        "denominator": "rows present in both prediction frames",
        "decision": "The only population on which a paired claim is legal.",
    },
}


# --------------------------------------------------------------------------- #
# Frame loading
# --------------------------------------------------------------------------- #
def _predictions(spec: ModelSpec) -> pd.DataFrame | None:
    path = spec.paths.results / "predictions.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    required = {"timestamp", "symbol", "actual_label", "predicted_label", "p_down", "p_flat", "p_up"}
    if not required.issubset(frame.columns):
        return None
    frame = frame.reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize(None)
    frame["symbol"] = frame["symbol"].astype(str)
    out = frame[["timestamp", "symbol", "actual_label", "predicted_label", "p_down", "p_flat", "p_up"]].copy()
    out["actual_label"] = out["actual_label"].astype(int)
    out["predicted_label"] = out["predicted_label"].astype(int)
    return out


def _proba(frame: pd.DataFrame) -> np.ndarray:
    return frame[["p_down", "p_flat", "p_up"]].to_numpy(dtype=float)


def _envelope(data: Any, artifacts: list[str], *, as_of: Any = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "data": json_safe(data),
        "as_of": to_iso(as_of),
        "generated_at": now_ist().isoformat(),
        "source": {"kind": "artifact", "artifacts": artifacts, "contract": CONTRACT},
        "warnings": warnings or [],
    }


# --------------------------------------------------------------------------- #
# Per-date sufficient statistics
# --------------------------------------------------------------------------- #
_I_ROWS, _I_CORRECT, _I_CALLED, _I_CORRECT_CALLED, _I_LOSS = 0, 1, 2, 3, 4
_I_CONF = 5                       # 3 x 3 over called rows, [actual, predicted]
_I_BIN_N = _I_CONF + 9            # ECE bins: count
_I_BIN_OK = _I_BIN_N + ECE_BINS   #             correct (argmax of the raw proba)
_I_BIN_CF = _I_BIN_OK + ECE_BINS  #             summed confidence
WIDTH = _I_BIN_CF + ECE_BINS


def _bin_index(conf: np.ndarray) -> np.ndarray:
    """ECE bin edges, matching ``expected_calibration_error``: (lo, hi], 0 inclusive."""
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    idx = np.searchsorted(edges, conf, side="left") - 1
    return np.clip(idx, 0, ECE_BINS - 1)


class _Daily:
    """Per-date sums of everything the paired interval needs.

    Storing a flat ``(n_dates, WIDTH)`` matrix is what makes the bootstrap cheap:
    a replicate is ``stats[idx].sum(axis=1)``, and every statistic is then a
    handful of operations on a ``(reps, WIDTH)`` slice.
    """

    __slots__ = ("dates", "stats")

    def __init__(self, dates: np.ndarray, actual: np.ndarray, predicted: np.ndarray, proba: np.ndarray) -> None:
        keys, positions = np.unique(dates, return_inverse=True)
        n = len(keys)
        stats = np.zeros((n, WIDTH), dtype=float)
        stats[:, _I_ROWS] = np.bincount(positions, minlength=n)
        correct = (predicted == actual).astype(float)
        called = (predicted != ABSTAIN).astype(float)
        stats[:, _I_CORRECT] = np.bincount(positions, weights=correct, minlength=n)
        stats[:, _I_CALLED] = np.bincount(positions, weights=called, minlength=n)
        stats[:, _I_CORRECT_CALLED] = np.bincount(positions, weights=correct * called, minlength=n)
        picked = np.take_along_axis(proba, actual.reshape(-1, 1), axis=1).reshape(-1)
        stats[:, _I_LOSS] = np.bincount(
            positions, weights=-np.log(np.clip(picked, 1e-12, 1.0)), minlength=n
        )

        # 3x3 confusion over called rows only: a matrix has no column for silence.
        conf = np.zeros((n, 3, 3), dtype=float)
        mask = called.astype(bool)
        np.add.at(conf, (positions[mask], actual[mask], predicted[mask]), 1.0)
        stats[:, _I_CONF:_I_CONF + 9] = conf.reshape(n, 9)

        # ECE bins, on the raw argmax rather than the policy's call, so the two
        # models' calibration stays comparable even when one of them abstains.
        raw = proba.argmax(axis=1)
        conf_max = proba.max(axis=1)
        bins = _bin_index(conf_max)
        bin_n = np.zeros((n, ECE_BINS), dtype=float)
        bin_ok = np.zeros((n, ECE_BINS), dtype=float)
        bin_cf = np.zeros((n, ECE_BINS), dtype=float)
        np.add.at(bin_n, (positions, bins), 1.0)
        np.add.at(bin_ok, (positions, bins), (raw == actual).astype(float))
        np.add.at(bin_cf, (positions, bins), conf_max)
        stats[:, _I_BIN_N:_I_BIN_N + ECE_BINS] = bin_n
        stats[:, _I_BIN_OK:_I_BIN_OK + ECE_BINS] = bin_ok
        stats[:, _I_BIN_CF:_I_BIN_CF + ECE_BINS] = bin_cf

        self.dates = keys
        self.stats = stats


def _safe(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros(np.shape(numerator), dtype=float), where=denominator > 0)


def _axes(s: np.ndarray) -> dict[str, np.ndarray]:
    """Every row-decomposable axis from summed per-date statistics.

    Accepts a single sum (shape ``(WIDTH,)``) or a batch (shape ``(reps, WIDTH)``)
    and returns arrays of the matching leading shape.
    """
    rows = s[..., _I_ROWS]
    total = np.maximum(rows, 1e-12)
    conf = s[..., _I_CONF:_I_CONF + 9].reshape(*s.shape[:-1], 3, 3)
    support = conf.sum(axis=-1)          # rows per true class, called rows only
    predicted_total = conf.sum(axis=-2)  # called rows per predicted class
    diag = np.diagonal(conf, axis1=-2, axis2=-1)

    per_class = np.stack([_safe(diag[..., i], support[..., i]) for i in range(3)], axis=-1)
    present = (support > 0)
    balanced = _safe((per_class * present).sum(axis=-1), present.sum(axis=-1).astype(float))

    n_called = support.sum(axis=-1)
    numerator = diag.sum(axis=-1) * n_called - (predicted_total * support).sum(axis=-1)
    denominator = np.sqrt(
        np.maximum(n_called ** 2 - (predicted_total ** 2).sum(axis=-1), 0.0)
        * np.maximum(n_called ** 2 - (support ** 2).sum(axis=-1), 0.0)
    )
    mcc = np.where(denominator > 0, _safe(numerator, denominator), 0.0)

    bin_n = s[..., _I_BIN_N:_I_BIN_N + ECE_BINS]
    bin_ok = s[..., _I_BIN_OK:_I_BIN_OK + ECE_BINS]
    bin_cf = s[..., _I_BIN_CF:_I_BIN_CF + ECE_BINS]
    gap = np.abs(_safe(bin_ok, bin_n) - _safe(bin_cf, bin_n))
    ece = _safe((bin_n * gap).sum(axis=-1), np.maximum(bin_n.sum(axis=-1), 1e-12))

    return {
        "accuracy_all": _safe(s[..., _I_CORRECT], total),
        "accuracy_called": _safe(s[..., _I_CORRECT_CALLED], np.maximum(s[..., _I_CALLED], 1e-12)),
        "coverage": _safe(s[..., _I_CALLED], total),
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "log_loss": _safe(s[..., _I_LOSS], total),
        "ece": ece,
        "recall_DOWN": per_class[..., 0],
        "recall_FLAT": per_class[..., 1],
        "recall_UP": per_class[..., 2],
    }


# Row-decomposable axes get a paired interval. AUC does not: it is a ranking
# statistic, not a sum over rows, so it is window-only and says so.
PAIRED_AXES: tuple[tuple[str, str, str, bool], ...] = (
    ("accuracy_all", "Directional accuracy (all rows)", "pct", True),
    ("accuracy_called", "Accuracy when it calls", "pct", True),
    ("balanced_accuracy", "Balanced accuracy (called)", "pct", True),
    ("mcc", "Matthews correlation (called)", "raw", True),
    ("recall_DOWN", "DOWN recall (called)", "pct", True),
    ("recall_FLAT", "FLAT recall (called)", "pct", True),
    ("recall_UP", "UP recall (called)", "pct", True),
    ("log_loss", "Log loss", "raw", False),
    ("ece", "Calibration error (ECE)", "raw", False),
)

WINDOW_AXES: tuple[tuple[str, str, str, bool], ...] = (
    ("auc_DOWN", "AUC( DOWN vs rest )", "raw", True),
    ("auc_FLAT", "AUC( FLAT vs rest )", "raw", True),
    ("auc_UP", "AUC( UP vs rest )", "raw", True),
)


def _auc(actual: np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for index, name in CLASS_NAMES.items():
        binary = (actual == index).astype(int)
        try:
            out[f"auc_{name}"] = float(roc_auc_score(binary, proba[:, index]))
        except ValueError:  # a class with no positives in this window
            out[f"auc_{name}"] = None
    return out


# --------------------------------------------------------------------------- #
# The paired join
# --------------------------------------------------------------------------- #
def _paired(a_id: str, b_id: str):
    """Load, join and align two models. Returns frames plus the accounting."""
    if a_id == b_id:
        raise HTTPException(status_code=400, detail="Compare needs two different models.")
    spec_a, spec_b = resolve(a_id), resolve(b_id)
    frame_a, frame_b = _predictions(spec_a), _predictions(spec_b)
    missing = [
        f"{name}: run its evaluation first"
        for name, frame in ((a_id, frame_a), (b_id, frame_b))
        if frame is None or frame.empty
    ]
    if missing:
        raise HTTPException(status_code=409, detail="; ".join(missing))

    left = frame_a.rename(columns={c: f"{c}__a" for c in frame_a.columns if c not in ("timestamp", "symbol")})
    right = frame_b.rename(columns={c: f"{c}__b" for c in frame_b.columns if c not in ("timestamp", "symbol")})
    overlap = left.merge(right, on=["timestamp", "symbol"], how="inner")
    if overlap.empty:
        raise HTTPException(
            status_code=409,
            detail=f"{a_id} and {b_id} share no (timestamp, symbol) rows; they are not comparable.",
        )

    # Rows the two snapshots label differently cannot be scored for either side:
    # there is no answer to "who was right" when the truth itself moved. They are
    # excluded and counted, never handed to the model whose snapshot happens to be
    # the one on disk. This is usually one evaluation predating a label rebuild.
    agree = overlap["actual_label__a"].to_numpy() == overlap["actual_label__b"].to_numpy()
    drift = (
        overlap.loc[~agree]
        .groupby(["actual_label__a", "actual_label__b"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    label_drift = [
        {"a_label": CLASS_NAMES.get(int(row.actual_label__a), str(row.actual_label__a)),
         "b_label": CLASS_NAMES.get(int(row.actual_label__b), str(row.actual_label__b)),
         "n": int(row.n)}
        for row in drift.itertuples()
    ]
    joined = overlap[agree].reset_index(drop=True)
    if joined.empty:
        raise HTTPException(
            status_code=409,
            detail=f"{a_id} and {b_id} share no rows with the same true label; one of them was "
                   "evaluated against a superseded label build.",
        )

    a_only = int(len(frame_a) - len(overlap))
    b_only = int(len(frame_b) - len(overlap))
    mismatch = int((~agree).sum())
    accounting = {
        "a_rows": int(len(frame_a)),
        "b_rows": int(len(frame_b)),
        "overlap_rows": int(len(overlap)),
        "paired_rows": int(len(joined)),
        f"{a_id}_only": a_only,
        f"{b_id}_only": b_only,
        "label_mismatch": mismatch,
        "discarded": a_only + b_only + mismatch,
        "label_drift": label_drift,
    }
    return spec_a, spec_b, frame_a, frame_b, joined, accounting


def _aligned(joined: pd.DataFrame, suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = joined[f"actual_label__{suffix}"].to_numpy(dtype=int)
    predicted = joined[f"predicted_label__{suffix}"].to_numpy(dtype=int)
    proba = joined[[f"p_down__{suffix}", f"p_flat__{suffix}", f"p_up__{suffix}"]].to_numpy(dtype=float)
    return actual, predicted, proba


def _dates(joined: pd.DataFrame) -> np.ndarray:
    return joined["timestamp"].dt.normalize().to_numpy()


def _bootstrap(a_daily: _Daily, b_daily: _Daily, *, reps: int = SUMMARY_REPS, seed: int = 42) -> dict[str, np.ndarray]:
    """One set of date draws, reused for every axis, so the intervals cohere."""
    n = len(a_daily.dates)
    if n < 2:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    deltas: dict[str, np.ndarray] = {}
    axes_a = _axes(a_daily.stats[idx].sum(axis=1))
    axes_b = _axes(b_daily.stats[idx].sum(axis=1))
    for key, *_ in PAIRED_AXES:
        deltas[key] = axes_b[key] - axes_a[key]
    return deltas


def _verdict(rows: Sequence[dict[str, Any]], a_id: str, b_id: str, gates: Sequence[dict[str, Any]] | None) -> str:
    """Name the winners per axis. There is deliberately no single winner field.

    The honest answer on this panel is split: whichever model leads on direction
    gives something up on calibration. Collapsing that into one word would be the
    one thing this page must not do.
    """
    wins = [row for row in rows if row["better"] in (a_id, b_id)]
    if not wins:
        summary = "No axis separates the two models by more than noise on the shared rows."
    else:
        counts = {a_id: sum(1 for row in wins if row["better"] == a_id),
                  b_id: sum(1 for row in wins if row["better"] == b_id)}
        summary = (
            f"{b_id} is ahead on {counts[b_id]} of {len(wins)} decided axes, {a_id} on {counts[a_id]}: "
            + "; ".join(f"{row['label']} -> {row['better']}" for row in wins) + "."
        )
    if gates:
        failed = [str(gate.get("label")) for gate in gates if not gate.get("pass")]
        if failed:
            # The gates are scored on v2's own test window by its report, not on
            # the shared rows, and the sentence has to say which one it means.
            summary += (
                f" {b_id} is still not promotable: it fails {', '.join(failed)} on its own test window, "
                "so this comparison decides what each model is good for, not what serves."
            )
        else:
            summary += f" {b_id} clears every promotion gate on its own test window; promotion remains a separate decision."
    return summary


def _gates() -> list[dict[str, Any]] | None:
    report = json_file(registry()["v2"].paths.artifacts / "v2_report.json") or {}
    return ((report.get("comparison") or {}).get("gates")) or None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.get("/summary")
def compare_summary(
    a: str = Query(default=DEFAULT_MODEL, description="reference model id"),
    b: str = Query(default="v2", description="challenger model id"),
) -> dict[str, Any]:
    """Scorecard, on the shared rows, with each model's own window as context."""
    spec_a, spec_b, frame_a, frame_b, joined, accounting = _paired(a, b)
    dates = _dates(joined)
    actual_a, pred_a, proba_a = _aligned(joined, "a")
    actual_b, pred_b, proba_b = _aligned(joined, "b")

    warnings: list[str] = []
    if accounting["label_mismatch"]:
        warnings.append(
            f"{accounting['label_mismatch']} row(s) carry a different true label in the two snapshots "
            f"and are excluded from every paired figure rather than awarded to either model "
            f"({accounting['overlap_rows']} rows overlap, {accounting['paired_rows']} are scoreable). "
            "One of the two evaluations predates the current label build; re-run it to close the gap."
        )
    if accounting["discarded"]:
        warnings.append(
            f"{accounting['discarded']} row(s) are not paired: {accounting[f'{spec_a.id}_only']} exist only "
            f"for {spec_a.id} and {accounting[f'{spec_b.id}_only']} only for {spec_b.id} (different test "
            f"windows) plus the {accounting['label_mismatch']} label mismatch(es). Each model's own window "
            "is shown separately so nothing is hidden by the cut."
        )

    daily_a = _Daily(dates, actual_a, pred_a, proba_a)
    daily_b = _Daily(dates, actual_b, pred_b, proba_b)
    point_a = _axes(daily_a.stats.sum(axis=0))
    point_b = _axes(daily_b.stats.sum(axis=0))
    deltas = _bootstrap(daily_a, daily_b)

    # Each model's own window, so the table can show "a" and "b" on the rows each
    # one actually scored while the delta stays on the shared rows.
    full_a = _axes(
        _Daily(_dates(frame_a), frame_a["actual_label"].to_numpy(int),
               frame_a["predicted_label"].to_numpy(int), _proba(frame_a)).stats.sum(axis=0)
    )
    full_b = _axes(
        _Daily(_dates(frame_b), frame_b["actual_label"].to_numpy(int),
               frame_b["predicted_label"].to_numpy(int), _proba(frame_b)).stats.sum(axis=0)
    )

    rows: list[dict[str, Any]] = []
    for key, label, unit, higher_is_better in PAIRED_AXES:
        delta = float(point_b[key] - point_a[key])
        draws = deltas.get(key)
        ci_low = float(np.percentile(draws, 2.5)) if draws is not None and len(draws) else None
        ci_high = float(np.percentile(draws, 97.5)) if draws is not None and len(draws) else None
        if ci_low is None or ci_high is None:
            significant = None
        else:
            significant = bool(ci_low > 0 or ci_high < 0)
        better = None
        if significant:
            better = b if ((delta > 0) == higher_is_better) else a
        rows.append(
            {
                "axis": key,
                "label": label,
                "unit": unit,
                "higher_is_better": higher_is_better,
                "a": float(full_a[key]),
                "b": float(full_b[key]),
                "paired_a": float(point_a[key]),
                "paired_b": float(point_b[key]),
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significant": significant,
                "better": better,
                "populations": "a, b on their own windows; delta on the shared rows",
            }
        )

    auc_a = _auc(frame_a["actual_label"].to_numpy(int), _proba(frame_a))
    auc_b = _auc(frame_b["actual_label"].to_numpy(int), _proba(frame_b))
    for key, label, unit, higher_is_better in WINDOW_AXES:
        value_a, value_b = auc_a.get(key), auc_b.get(key)
        rows.append(
            {
                "axis": key,
                "label": label,
                "unit": unit,
                "higher_is_better": higher_is_better,
                "a": value_a,
                "b": value_b,
                "paired_a": None,
                "paired_b": None,
                "delta": None if value_a is None or value_b is None else float(value_b - value_a),
                "ci_low": None,
                "ci_high": None,
                "significant": None,
                "better": None,
                "populations": "each model's own window; ranking statistic, so no paired interval",
            }
        )

    gates = _gates()
    return _envelope(
        {
            "a": spec_a.id,
            "b": spec_b.id,
            "labels": {spec_a.id: spec_a.label, spec_b.id: spec_b.label},
            "scorecard": rows,
            "verdict": _verdict(rows, spec_a.id, spec_b.id, gates),
            "gates": gates,
            "paired_rows": accounting["paired_rows"],
            "a_rows": accounting["a_rows"],
            "b_rows": accounting["b_rows"],
            "overlap_rows": accounting["overlap_rows"],
            "label_mismatch": accounting["label_mismatch"],
            "label_drift": accounting["label_drift"],
            "discarded": accounting["discarded"],
            "test_dates": int(len(daily_a.dates)),
            "reps": SUMMARY_REPS,
            "class_share": {
                name: float((actual_a == index).mean()) for index, name in CLASS_NAMES.items()
            },
        },
        [f"{spec_a.paths.results / 'predictions.parquet'}", f"{spec_b.paths.results / 'predictions.parquet'}"],
        as_of=joined["timestamp"].max(),
        warnings=warnings,
    )


# --- segments -------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _volatility_lookup(features_dir: str) -> pd.DataFrame:
    """``volatility_20`` per (date, symbol), read from the feature store.

    Cached because it opens one file per symbol. A partial store simply yields
    fewer rows, and the segment reports the coverage it got instead of guessing.
    """
    frames = []
    for path in sorted(Path(features_dir).glob("*_features.parquet")):
        try:
            data = pd.read_parquet(path, columns=["volatility_20"])
        except (OSError, ValueError, KeyError):
            continue
        if "volatility_20" not in data.columns or data.empty:
            continue
        # Feature files are named after the sanitised symbol.
        symbol = path.name.removesuffix("_features.parquet").replace("index_", "^").replace("_", ".")
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(data.index).normalize(),
                    "symbol": symbol,
                    "volatility_20": data["volatility_20"].to_numpy(dtype=float),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "volatility_20"])
    return pd.concat(frames, ignore_index=True)


def _segment_column(joined: pd.DataFrame, how: str, a_id: str, a_proba: np.ndarray, actual: np.ndarray, b_pred: np.ndarray) -> tuple[pd.Series, str, list[str]]:
    """The segment label for every shared row, plus its definition."""
    notes: list[str] = []
    if how == "class":
        labels = pd.Series([CLASS_NAMES[int(v)] for v in actual], index=joined.index)
        return labels, "true class of the row", notes
    if how == "symbol":
        return joined["symbol"].astype(str), "symbol", notes
    if how == "month":
        labels = joined["timestamp"].dt.to_period("M").astype(str)
        return labels, "calendar month of the row", notes

    if how == "confidence":
        # One reference model, deliberately: a bucket has to hold the same rows
        # for both models, or the two accuracies inside it are not comparable.
        conf = a_proba.max(axis=1)
        edges = np.quantile(conf, [0.2, 0.4, 0.6, 0.8])
        labels = pd.Series(
            [f"Q{int(np.searchsorted(edges, value, side='right')) + 1}" for value in conf],
            index=joined.index,
        )
        bounds = [float(conf.min()), *map(float, edges), float(conf.max())]
        notes.append(
            f"Confidence buckets are quintiles of {a_id}'s top-class probability "
            f"(edges {', '.join(f'{value:.3f}' for value in bounds)}), so the same rows fall in the "
            "same bucket for both models."
        )
        return labels, f"quintiles of {a_id} confidence", notes

    if how == "agreement":
        a_ok = joined["predicted_label__a"].to_numpy(int) == actual
        b_ok = b_pred == actual
        labels = pd.Series(
            np.select(
                [a_ok & b_ok, a_ok & ~b_ok, ~a_ok & b_ok],
                ["both right", f"{a_id} only", "challenger only"],
                default="both wrong",
            ),
            index=joined.index,
        )
        notes.append(
            "Agreement buckets are defined by the outcome, so they are counts of who was right, "
            "not a test: the interval and winner columns are deliberately empty here."
        )
        return labels, "who got the row right", notes

    if how == "volatility":
        lookup = _volatility_lookup(str(registry()["v1"].paths.data_features))
        if lookup.empty:
            labels = pd.Series(["unavailable"] * len(joined), index=joined.index)
            notes.append("volatility_20 is not readable from the feature store; the volatility split is unavailable.")
            return labels, "terciles of volatility_20", notes
        merged = joined[["timestamp", "symbol"]].merge(lookup, on=["timestamp", "symbol"], how="left")
        values = merged["volatility_20"].to_numpy(dtype=float)
        coverage = float(np.isfinite(values).mean())
        if coverage < 1.0:
            notes.append(
                f"volatility_20 joined on {coverage:.1%} of shared rows; rows without it are reported "
                "as 'unavailable' and excluded from the winner column."
            )
        finite = values[np.isfinite(values)]
        if finite.size < 3:
            labels = pd.Series(["unavailable"] * len(joined), index=joined.index)
            return labels, "terciles of volatility_20", notes
        edges = np.quantile(finite, [1 / 3, 2 / 3])
        labels = pd.Series(
            np.select(
                [~np.isfinite(values), values <= edges[0], values <= edges[1]],
                ["unavailable", "low volatility", "mid volatility"],
                default="high volatility",
            ),
            index=joined.index,
        )
        return labels, "terciles of volatility_20 on the shared rows", notes

    raise HTTPException(
        status_code=400,
        detail=f"Unknown segment {how!r}. Known: {', '.join(SEGMENT_KEYS)}",
    )


def _ratio_ci(rows_per_date: np.ndarray, a_correct: np.ndarray, b_correct: np.ndarray, *, reps: int = SEGMENT_REPS, seed: int = 42) -> tuple[float | None, float | None]:
    n = len(rows_per_date)
    if n < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    total = np.maximum(rows_per_date[idx].sum(axis=1), 1e-12)
    a_acc = a_correct[idx].sum(axis=1) / total
    b_acc = b_correct[idx].sum(axis=1) / total
    delta = b_acc - a_acc
    return float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))


@router.get("/segments")
def compare_segments(
    a: str = Query(default=DEFAULT_MODEL),
    b: str = Query(default="v2"),
    by: str = Query(default="class", description=f"one of {', '.join(SEGMENT_KEYS)}"),
) -> dict[str, Any]:
    """Where one model wins: the same paired delta, computed inside each slice."""
    spec_a, spec_b, _frame_a, _frame_b, joined, accounting = _paired(a, b)
    actual, pred_a, proba_a = _aligned(joined, "a")
    _, pred_b, _ = _aligned(joined, "b")

    labels, definition, notes = _segment_column(joined, by, spec_a.id, proba_a, actual, pred_b)
    dates = _dates(joined)
    a_ok = (pred_a == actual).astype(float)
    b_ok = (pred_b == actual).astype(float)

    rows: list[dict[str, Any]] = []
    for key in sorted(labels.unique(), key=str):
        mask = (labels == key).to_numpy()
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue
        seg_dates = dates[mask]
        keys, positions = np.unique(seg_dates, return_inverse=True)
        rows_per_date = np.bincount(positions, minlength=len(keys)).astype(float)
        a_per_date = np.bincount(positions, weights=a_ok[mask], minlength=len(keys))
        b_per_date = np.bincount(positions, weights=b_ok[mask], minlength=len(keys))
        n_a = float(a_per_date.sum() / max(rows_per_date.sum(), 1e-12))
        n_b = float(b_per_date.sum() / max(rows_per_date.sum(), 1e-12))
        delta = n_b - n_a

        enough = n_rows >= MIN_SEGMENT_ROWS and len(keys) >= MIN_SEGMENT_DATES
        ci_low = ci_high = None
        if enough and by != "agreement":
            ci_low, ci_high = _ratio_ci(rows_per_date, a_per_date, b_per_date)
        significant = None if ci_low is None or ci_high is None else bool(ci_low > 0 or ci_high < 0)
        winner = None
        if significant:
            winner = spec_b.id if delta > 0 else spec_a.id
        rows.append(
            {
                "key": str(key),
                "n": n_rows,
                "dates": int(len(keys)),
                "a_accuracy": n_a,
                "b_accuracy": n_b,
                "delta_pp": delta * 100.0,
                "ci_low": None if ci_low is None else ci_low * 100.0,
                "ci_high": None if ci_high is None else ci_high * 100.0,
                "winner": winner,
                "significant": significant,
                "note": (
                    None
                    if enough
                    else f"n={n_rows} over {len(keys)} date(s) is below the {MIN_SEGMENT_ROWS}-row / "
                         f"{MIN_SEGMENT_DATES}-date floor, so no interval is claimed."
                ),
            }
        )

    rows.sort(key=lambda row: row["n"], reverse=True)
    # The notes explain the definition; only a missing join is a warning, because
    # only a missing join means a row of this table is reporting less than it seems.
    return _envelope(
        {
            "a": spec_a.id,
            "b": spec_b.id,
            "by": by,
            "definition": definition,
            "rows": rows,
            "paired_rows": accounting["paired_rows"],
            "metric": "accuracy_all",
            "notes": notes,
        },
        [
            str(spec_a.paths.results / "predictions.parquet"),
            str(spec_b.paths.results / "predictions.parquet"),
        ],
        as_of=joined["timestamp"].max(),
        warnings=[note for note in notes if "unavailable" in note or "joined on" in note],
    )


@router.get("/pairing")
def compare_pairing(
    a: str = Query(default=DEFAULT_MODEL),
    b: str = Query(default="v2"),
) -> dict[str, Any]:
    """The 2x2 and the flips: the only rows that carry information about which is better."""
    spec_a, spec_b, _fa, _fb, joined, accounting = _paired(a, b)
    actual, pred_a, _ = _aligned(joined, "a")
    _, pred_b, _ = _aligned(joined, "b")
    a_ok = pred_a == actual
    b_ok = pred_b == actual

    test = mcnemar(a_ok, b_ok)
    agreement = {
        "both_right": int((a_ok & b_ok).sum()),
        "a_only": int((a_ok & ~b_ok).sum()),
        "b_only": int((~a_ok & b_ok).sum()),
        "both_wrong": int((~a_ok & ~b_ok).sum()),
    }

    flips: list[dict[str, Any]] = []
    for (pred_a_value, pred_b_value), group in pd.DataFrame(
        {"a_pred": pred_a, "b_pred": pred_b, "a_hit": a_ok, "b_hit": b_ok}
    ).groupby(["a_pred", "b_pred"]):
        if pred_a_value == pred_b_value:
            continue
        flips.append(
            {
                "a_pred": CLASS_NAMES.get(int(pred_a_value), "ABSTAIN" if pred_a_value == ABSTAIN else str(pred_a_value)),
                "b_pred": CLASS_NAMES.get(int(pred_b_value), "ABSTAIN" if pred_b_value == ABSTAIN else str(pred_b_value)),
                "n": int(len(group)),
                "a_hit_rate": float(group["a_hit"].mean()),
                "b_hit_rate": float(group["b_hit"].mean()),
            }
        )
    flips.sort(key=lambda row: row["n"], reverse=True)

    return _envelope(
        {
            "a": spec_a.id,
            "b": spec_b.id,
            "agreement": agreement,
            "mcnemar_p": test["p_value"],
            "discordant_n": test["discordant"],
            "a_only": test["a_only"],
            "b_only": test["b_only"],
            "flips": flips,
            "paired_rows": accounting["paired_rows"],
            "method": test["method"],
            "reading": (
                "Only the discordant rows say anything about which model is better. A p-value above 0.05 "
                "means the row counts cannot separate them, whatever the accuracy gap looks like."
            ),
        },
        [],
        as_of=joined["timestamp"].max(),
    )


@lru_cache(maxsize=4)
def _shap(path: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    if "feature" not in frame.columns or "mean_abs_shap" not in frame.columns:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


@router.get("/attribution")
def compare_attribution(
    a: str = Query(default=DEFAULT_MODEL),
    b: str = Query(default="v2"),
) -> dict[str, Any]:
    """Whether the two models are looking at the same thing, and where they diverge."""
    spec_a, spec_b = resolve(a), resolve(b)
    frame_a = _shap(str(spec_a.paths.results / "shap_feature_importance.csv"))
    frame_b = _shap(str(spec_b.paths.results / "shap_feature_importance.csv"))
    warnings: list[str] = []
    if frame_a.empty:
        warnings.append(f"{a} has no SHAP importance file; its side of the attribution is empty.")
    if frame_b.empty:
        warnings.append(
            f"{b} has no SHAP importance file; run `python -m stockml.v2.stages evaluate` first."
        )

    def top(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
        subset = frame.head(limit)
        total = float(frame["mean_abs_shap"].sum()) or 1.0
        return [
            {
                "rank": position + 1,
                "feature": str(row.feature),
                "share": float(row.mean_abs_shap) / total,
            }
            for position, row in enumerate(subset.itertuples())
        ]

    rank_a = {str(name): position for position, name in enumerate(frame_a["feature"])}
    rank_b = {str(name): position for position, name in enumerate(frame_b["feature"])}
    shared = sorted(set(rank_a) & set(rank_b))

    correlation = None
    if len(shared) >= 3:
        value = spearmanr([rank_a[name] for name in shared], [rank_b[name] for name in shared]).statistic
        correlation = None if value is None or math.isnan(value) else float(value)

    top_a = {row["feature"] for row in top(frame_a, TOP_K)}
    top_b = {row["feature"] for row in top(frame_b, TOP_K)}
    divergent = [
        {
            "feature": name,
            "a_rank": rank_a.get(name),
            "b_rank": rank_b.get(name),
            "only_in": a if name in top_a else b,
        }
        for name in sorted(top_a.symmetric_difference(top_b), key=lambda name: str(name))
    ]

    note = (
        "Importance is mean |SHAP| over each model's own test rows, so the two lists are not measured on "
        "identical samples. A high rank correlation means the pair is reading the same features; the "
        "divergent list is where a disagreement in behaviour would come from."
        if shared
        else "The two importance files share no feature names; check that both were fitted on the same feature store."
    )
    return _envelope(
        {
            "a": spec_a.id,
            "b": spec_b.id,
            "a_top": top(frame_a, TOP_K),
            "b_top": top(frame_b, TOP_K),
            "shared_features": len(shared),
            "a_features": int(len(frame_a)),
            "b_features": int(len(frame_b)),
            "rank_correlation": correlation,
            "divergent": divergent,
            "note": note,
        },
        [
            str(spec_a.paths.results / "shap_feature_importance.csv"),
            str(spec_b.paths.results / "shap_feature_importance.csv"),
        ],
        warnings=warnings,
    )

"""Behaviour analysis for the trained LightGBM model.

Pure, FastAPI-free functions over a scored frame
(``results/predictions_splits.parquet``). Every figure the ``#/model`` page
draws is derived here, and the functions are deliberately symmetric across
classes: "why DOWN", "why UP" and "why balanced" are the same computation
with a different dominant class. The mode + headline are *generated* from the
numbers, so the page cannot report the answer it expected to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("DOWN", "FLAT", "UP")
LABELS = {0: "DOWN", 1: "FLAT", 2: "UP"}
PROBA_COLUMNS = ("p_down", "p_flat", "p_up")

# Conviction bins for panel D. 1/3 is the uniform-prior floor: below it the
# model is not merely unsure, it is less sure than guessing.
CONVICTION_EDGES = (0.0, 0.34, 0.45, 0.55, 0.65, 1.0001)

TAU_GRID = tuple(round(float(x), 2) for x in np.arange(0.30, 0.601, 0.02))

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_REJECTED = "REJECTED"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_INFO = "INFO"


def _r(value, digits: int = 4):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return round(out, digits)


def _pct(value):
    """Share (0-1) to percentage points, 1 decimal."""
    if value is None:
        return None
    return round(float(value) * 100.0, 1)


def _dates(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        return pd.to_datetime(pd.Series(frame["date"]).to_numpy())
    if isinstance(frame.index, pd.MultiIndex) and "timestamp" in (frame.index.names or []):
        return pd.to_datetime(frame.index.get_level_values("timestamp"))
    return pd.to_datetime(frame.index)


def distribution(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Panel A: what the data contains vs what the model produces."""
    total = int(len(actual))
    if total == 0:
        return {"rows": 0, "actual": {}, "predicted": {}, "bias_pp": {}, "dominant": None, "dominant_share": None}
    actual_share = {LABELS[c]: float((actual == c).mean()) for c in range(3)}
    predicted_share = {LABELS[c]: float((predicted == c).mean()) for c in range(3)}
    bias = {name: _pct(predicted_share[name] - actual_share[name]) for name in CLASSES}
    dominant = max(CLASSES, key=lambda name: predicted_share[name])
    return {
        "rows": total,
        "actual": {k: _r(v) for k, v in actual_share.items()},
        "predicted": {k: _r(v) for k, v in predicted_share.items()},
        "bias_pp": bias,
        "dominant": dominant,
        "dominant_share": _r(predicted_share[dominant]),
        "majority_class": LABELS[int(np.bincount(actual, minlength=3).argmax())],
        "majority_share": _r(max(actual_share.values())),
    }


def confusion(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Panel B: row-normalised recall with raw counts kept alongside."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

    counts = np.zeros((3, 3), dtype=int)
    for truth, call in zip(actual.astype(int), predicted.astype(int)):
        counts[truth, call] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    col_totals = counts.sum(axis=0, keepdims=True)
    recall = np.divide(counts, row_totals, out=np.full(counts.shape, np.nan, dtype=float), where=row_totals > 0)
    precision = np.divide(counts, col_totals, out=np.full(counts.shape, np.nan, dtype=float), where=col_totals > 0)
    if len(actual):
        accuracy = float(accuracy_score(actual, predicted))
        balanced = float(balanced_accuracy_score(actual, predicted))
        mcc = float(matthews_corrcoef(actual, predicted))
    else:
        accuracy = balanced = mcc = None
    return {
        "labels": list(CLASSES),
        "counts": counts.tolist(),
        "recall": {LABELS[c]: _r(recall[c, c]) for c in range(3)},
        "precision": {LABELS[c]: _r(precision[c, c]) for c in range(3)},
        "accuracy": _r(accuracy),
        "balanced_accuracy": _r(balanced),
        "mcc": _r(mcc),
    }


def class_profile(actual: np.ndarray, proba: np.ndarray) -> dict:
    """Panel C: mean predicted probabilities per *true* class.

    ``separation`` is the mean pairwise total-variation distance between those
    three mean vectors (0 = the model says the same thing whatever is true).
    """
    rows = []
    vectors = {}
    for c in range(3):
        mask = actual == c
        n = int(mask.sum())
        if n == 0:
            rows.append({"actual": LABELS[c], "n": 0, "mean": {name: None for name in CLASSES}})
            continue
        mean = proba[mask].mean(axis=0)
        vectors[c] = mean
        rows.append({
            "actual": LABELS[c],
            "n": n,
            "mean": {CLASSES[i]: _r(mean[i]) for i in range(3)},
        })
    distances = []
    for a in range(3):
        for b in range(a + 1, 3):
            if a in vectors and b in vectors:
                distances.append(0.5 * float(np.abs(vectors[a] - vectors[b]).sum()))
    separation = float(np.mean(distances)) if distances else 0.0
    overall = proba.mean(axis=0) if len(proba) else np.array([np.nan, np.nan, np.nan])
    return {
        "rows": rows,
        "separation": _r(separation),
        "mean_probability": {CLASSES[i]: _r(overall[i]) for i in range(3)},
    }


def conviction(proba: np.ndarray, predicted: np.ndarray) -> dict:
    """Panel D: is a call a decision or a plurality?"""
    if not len(proba):
        return {"bins": [], "share_below_half": None, "mean_max_p": None, "by_class": {}}
    max_p = proba.max(axis=1)
    bins = []
    for lo, hi in zip(CONVICTION_EDGES[:-1], CONVICTION_EDGES[1:]):
        mask = (max_p >= lo) & (max_p < hi)
        bins.append({
            "lo": _r(lo, 2),
            "hi": _r(min(hi, 1.0), 2),
            "n": int(mask.sum()),
            "share": _r(mask.mean()),
        })
    by_class = {}
    for c in range(3):
        mask = predicted == c
        if not mask.any():
            by_class[LABELS[c]] = {"calls": 0, "share_below_half": None, "mean_max_p": None}
            continue
        values = max_p[mask]
        by_class[LABELS[c]] = {
            "calls": int(mask.sum()),
            "share_below_half": _r((values < 0.5).mean()),
            "mean_max_p": _r(values.mean()),
        }
    return {
        "bins": bins,
        "share_below_half": _r((max_p < 0.5).mean()),
        "share_below_uniform": _r((max_p < 1.0 / 3.0 + 0.005).mean()),
        "mean_max_p": _r(max_p.mean()),
        "by_class": by_class,
    }


def margins(proba: np.ndarray, edges=(0.0, 0.02, 0.05, 0.10, 0.20, 1.0001)) -> dict:
    """Top-1 minus top-2 probability: how close every call was to flipping."""
    if not len(proba):
        return {"bins": [], "median": None, "share_below_0_05": None, "share_below_0_10": None}
    ordered = np.sort(proba, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (gap >= lo) & (gap < hi)
        bins.append({"lo": _r(lo, 2), "hi": _r(min(hi, 1.0), 2), "n": int(mask.sum()), "share": _r(mask.mean())})
    return {
        "bins": bins,
        "median": _r(np.median(gap)),
        "mean": _r(gap.mean()),
        "share_below_0_05": _r((gap < 0.05).mean()),
        "share_below_0_10": _r((gap < 0.10).mean()),
    }


def threshold_sweep(actual: np.ndarray, proba: np.ndarray, taus=TAU_GRID) -> list[dict]:
    """τ_flat sweep on the *deployed* rule, for diagnosis only.

    This never feeds back into config: it answers "would another τ have
    behaved differently", not "change τ".
    """
    from ..evaluation.flat_threshold import apply_flat_threshold

    actual = np.asarray(actual, dtype=int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for tau in taus:
        pred = apply_flat_threshold(proba, float(tau))
        recall = {}
        for c in range(3):
            mask = actual == c
            recall[LABELS[c]] = _r((pred[mask] == c).mean()) if mask.any() else None
        rows.append({
            "tau_flat": _r(tau, 2),
            "accuracy": _r((pred == actual).mean()),
            "coverage_direction": _r(float((pred != 1).mean())),
            "flat_share": _r(float((pred == 1).mean())),
            "down_share": _r(float((pred == 0).mean())),
            "up_share": _r(float((pred == 2).mean())),
            "recall": recall,
        })
    return rows


def rolling(frame: pd.DataFrame, window: int = 30) -> list[dict]:
    """Per-session behaviour, plus a rolling mean, for the "has it drifted" line."""
    if frame.empty:
        return []
    work = pd.DataFrame({
        "date": _dates(frame).to_numpy(),
        "actual": frame["actual_label"].to_numpy(dtype=int),
        "predicted": frame["predicted_label"].to_numpy(dtype=int),
    })
    grouped = work.groupby("date", sort=True)
    rows = []
    for date, chunk in grouped:
        n = len(chunk)
        rows.append({
            "date": pd.Timestamp(date).date().isoformat(),
            "n": int(n),
            "pred_down": _r(float((chunk["predicted"] == 0).mean())),
            "pred_flat": _r(float((chunk["predicted"] == 1).mean())),
            "pred_up": _r(float((chunk["predicted"] == 2).mean())),
            "actual_down": _r(float((chunk["actual"] == 0).mean())),
            "actual_up": _r(float((chunk["actual"] == 2).mean())),
            "accuracy": _r(float((chunk["predicted"] == chunk["actual"]).mean())),
        })
    if len(rows) >= 2:
        series = pd.DataFrame(rows)
        for column in ("pred_down", "pred_up", "accuracy"):
            series[f"roll_{column}"] = series[column].rolling(window, min_periods=1).mean().round(4)
        rows = series.to_dict(orient="records")
    return rows


def mode_of(dist: dict, profile: dict, confusion_block: dict) -> dict:
    """Classify the behaviour — the same rules whatever the dominant class is."""
    bias = dist.get("bias_pp") or {}
    if not bias:
        return {"mode": "unknown", "headline": "No scored rows.", "flags": [], "evidence": []}
    separation = profile.get("separation") or 0.0
    dominant = dist.get("dominant") or max(bias, key=lambda k: bias[k])
    max_bias = max(bias.values())
    accuracy = confusion_block.get("accuracy")
    majority = dist.get("majority_share")

    if max(abs(v) for v in bias.values()) <= 5.0:
        if accuracy is not None and majority is not None and accuracy > majority + 0.02:
            mode = "balanced_discriminating"
            headline = (
                f"Balanced and informative: shares match the data and accuracy {_pct(accuracy)}% "
                f"beats the {_pct(majority)}% prior by {_pct(accuracy - majority)} pp."
            )
        else:
            mode = "balanced_undiscriminating"
            headline = (
                f"Balanced but uninformative: produced shares match the data, yet accuracy "
                f"{_pct(accuracy)}% is level with the {_pct(majority)}% prior."
            )
    elif max_bias > 15.0 and separation < 0.05:
        mode = "collapsed_prior"
        headline = (
            f"Output collapsed onto {dominant}: produced {_pct(dist['predicted'][dominant])}% of calls "
            f"vs {_pct(dist['actual'][dominant])}% in the data, while the class profiles differ by only "
            f"{separation:.3f}."
        )
    elif max_bias > 10.0 and separation >= 0.05:
        mode = "tilted"
        headline = (
            f"Genuine tilt toward {dominant} (+{bias[dominant]:.1f} pp of output) with measurable "
            f"class separation of {separation:.3f}."
        )
    else:
        mode = "mild_bias"
        headline = (
            f"Mild {dominant} bias (+{bias[dominant]:.1f} pp), separation {separation:.3f}: "
            f"between a real tilt and a collapsed prior."
        )

    flags = []
    for name in CLASSES:
        actual_share = dist["actual"].get(name) or 0.0
        predicted_share = dist["predicted"].get(name) or 0.0
        if actual_share > 0.02 and predicted_share < 0.5 * actual_share:
            flags.append(f"starved:{name}")
        if bias.get(name, 0.0) > 15.0:
            flags.append(f"over_produced:{name}")

    evidence = [
        {"label": "predicted share", "value": f"{_pct(dist['predicted'][dominant])}%", "hint": dominant},
        {"label": "actual share", "value": f"{_pct(dist['actual'][dominant])}%", "hint": dominant},
        {"label": "bias", "value": f"{bias[dominant]:+.1f} pp"},
        {"label": "class separation", "value": f"{separation:.3f}"},
        {"label": "accuracy", "value": f"{_pct(accuracy)}%" if accuracy is not None else "n/a"},
        {"label": "prior", "value": f"{_pct(majority)}%" if majority is not None else "n/a"},
    ]
    return {"mode": mode, "headline": headline, "flags": flags, "evidence": evidence}


def plate(frame: pd.DataFrame, split: str, window: int = 30) -> dict:
    """Full payload for one split: the five panels plus rule and mode."""
    from ..evaluation.metrics import apply_temperature, expected_calibration_error
    from ..evaluation.flat_threshold import apply_flat_threshold

    if frame is None or not len(frame):
        return {
            "split": split,
            "rows": 0,
            "distribution": {"actual": {}, "predicted": {}, "bias_pp": {}},
            "confusion": {"labels": list(CLASSES), "counts": [[0] * 3] * 3, "recall": {}, "accuracy": None},
            "profile": {"rows": [], "separation": None},
            "conviction": {"bins": [], "by_class": {}},
            "margins": {"bins": []},
            "rolling": [],
            "rule": {},
            "mode": {"mode": "unavailable", "headline": f"No scored rows for {split}.", "flags": [], "evidence": []},
        }

    actual = frame["actual_label"].to_numpy(dtype=int)
    predicted = frame["predicted_label"].to_numpy(dtype=int)
    proba = frame[list(PROBA_COLUMNS)].to_numpy(dtype=float)
    dist = distribution(actual, predicted)
    conf = confusion(actual, predicted)
    prof = class_profile(actual, proba)
    conv = conviction(proba, predicted)
    marg = margins(proba)

    argmax_accuracy = float((proba.argmax(axis=1) == actual).mean()) if len(actual) else None
    tau_flat = None
    try:
        from ..config import load_settings
        tau_flat = float(load_settings().flat_threshold)
    except Exception:
        tau_flat = None
    # Below τ_flat the rule ignores p_flat entirely and compares p_down vs
    # p_up. Measuring that comparison directly is what shows whether the
    # directional decision is one-sided before any class evidence is used.
    if tau_flat is not None and len(proba):
        nonflat = proba[:, 1] < tau_flat
        fallback_down = float((proba[nonflat, 0] >= proba[nonflat, 2]).mean()) if nonflat.any() else None
        fallback_share = float(nonflat.mean())
    else:
        fallback_down, fallback_share = None, None
    try:
        ece = expected_calibration_error(actual, proba, n_bins=10)
        ece_value, ece_bins = ece["ece"], ece["bins"]
    except Exception:
        ece_value, ece_bins = None, []
    _ = (apply_temperature, apply_flat_threshold)  # documented dependency, used by callers

    return {
        "split": split,
        "rows": int(len(frame)),
        "distribution": dist,
        "confusion": conf,
        "profile": prof,
        "conviction": conv,
        "margins": marg,
        "rolling": rolling(frame, window=window),
        "rule": {
            "tau_flat": _r(tau_flat, 2),
            "accuracy": _r(conf["accuracy"]),
            "accuracy_argmax": _r(argmax_accuracy),
            "flat_share": _r(float((predicted == 1).mean())),
            "below_tau_share": _r(fallback_share),
            "fallback_down_share": _r(fallback_down),
            "ece": _r(ece_value),
            "ece_bins": [
                {
                    "lo": _r(b.get("lo"), 2),
                    "hi": _r(b.get("hi"), 2),
                    "count": b.get("count"),
                    "accuracy": _r(b.get("accuracy")),
                    "confidence": _r(b.get("confidence")),
                }
                for b in ece_bins
            ],
        },
        "mode": mode_of(dist, prof, conf),
    }


def driver_rows(shap_by_class: pd.DataFrame, dominant: str | None = None, limit: int = 8) -> list[dict]:
    """Panel E: the features that push *toward* a class.

    Only positive signed contributions are kept: a feature that pushes away
    from a class does not explain why the model chose it. ``dominant=None``
    returns the top rows for every class, tagged with ``class``, so each split
    can show the drivers behind whichever class it over-produces.
    """
    if shap_by_class is None or shap_by_class.empty or "class" not in shap_by_class.columns:
        return []
    frame = shap_by_class.copy()
    if dominant is not None:
        frame = frame[frame["class"].astype(str).str.upper() == dominant.upper()]
    rows: list[dict] = []
    for name, chunk in frame.groupby(frame["class"].astype(str).str.upper(), sort=True):
        chunk = chunk[chunk["mean_signed"] > 0].sort_values("mean_signed", ascending=False).head(limit)
        rows.extend(
            {
                "class": name,
                "feature": str(row["feature"]),
                "mean_signed": _r(row["mean_signed"]),
                "mean_abs": _r(row.get("mean_abs")),
            }
            for _, row in chunk.iterrows()
        )
    return rows


def hypotheses(
    *,
    plates: dict,
    sweep: list[dict],
    importance: list[dict],
    tau_flat: float | None,
    psi_threshold: float,
    calibration: dict | None = None,
) -> list[dict]:
    """Auditable candidate causes. A REJECTED verdict is a result, not a gap."""
    test = plates.get("test") or {}
    train = plates.get("train") or {}
    out: list[dict] = []

    # 1. Class overlap: the mechanism behind collapse *or* uninformative balance.
    sep = (test.get("profile") or {}).get("separation")
    below_half = (test.get("conviction") or {}).get("share_below_half")
    overlap_ok = sep is not None and below_half is not None and sep < 0.05 and below_half > 0.40
    out.append({
        "id": "class_overlap",
        "title": "Class probabilities barely separate",
        "claim": "The mean probability vectors for DOWN / FLAT / UP are nearly identical, so the "
                 "decision is decided by the tie-break, not by evidence.",
        "verdict": VERDICT_SUPPORTED if overlap_ok else VERDICT_REJECTED,
        "split": "test",
        "chart": "profile",
        "evidence": [
            {"label": "separation", "value": f"{sep:.3f}" if sep is not None else "n/a", "threshold": "< 0.050"},
            {"label": "calls below 0.5", "value": f"{_pct(below_half)}%" if below_half is not None else "n/a", "threshold": "> 40%"},
        ],
    })

    # 2. The deployed fallback: below τ the rule compares p_down vs p_up only.
    rule = test.get("rule") or {}
    fallback_down = rule.get("fallback_down_share")
    below_tau = rule.get("below_tau_share")
    fallback_tau = rule.get("tau_flat")
    onedirectional = (
        fallback_down is not None and below_tau is not None
        and below_tau > 0.50
        and (fallback_down >= 0.75 or fallback_down <= 0.25)
    )
    out.append({
        "id": "decision_rule_tiebreak",
        "title": "Below τ the rule decides on p_down vs p_up alone",
        "claim": "apply_flat_threshold() predicts FLAT only at or above τ, then picks UP only "
                 "when p_up > p_down. Whatever p_flat says below τ is discarded, so the "
                 "directional call is decided by one comparison.",
        "verdict": VERDICT_SUPPORTED if onedirectional else VERDICT_REJECTED,
        "split": "test",
        "chart": "conviction",
        "evidence": [
            {"label": "rows below τ", "value": f"{_pct(below_tau)}%" if below_tau is not None else "n/a", "threshold": "> 50%"},
            {
                "label": "of those, fallback → DOWN",
                "value": f"{_pct(fallback_down)}%" if fallback_down is not None else "n/a",
                "threshold": "≥ 75% or ≤ 25%",
            },
            {"label": "τ used", "value": f"{fallback_tau:.2f}" if fallback_tau is not None else "n/a"},
        ],
    })

    # 3. Flat band: would another τ have behaved differently? Diagnostic only.
    tau_rows = [r for r in sweep if r.get("split") == "test"] or [r for r in sweep if r.get("accuracy") is not None]
    best = max(tau_rows, key=lambda r: r["accuracy"]) if tau_rows else None
    best_tau = best["tau_flat"] if best else None
    band_ok = (
        best_tau is not None and tau_flat is not None and abs(best_tau - tau_flat) >= 0.05
    )
    out.append({
        "id": "flat_band_misset",
        "title": "A different FLAT threshold would decide differently",
        "claim": "The τ_flat sweep is recomputed from stored probabilities. It is a diagnostic "
                 "only — nothing here changes the configured threshold.",
        "verdict": VERDICT_SUPPORTED if band_ok else VERDICT_REJECTED,
        "split": "test",
        "chart": "sweep",
        "evidence": [
            {"label": "configured τ", "value": f"{tau_flat:.2f}" if tau_flat is not None else "n/a"},
            {"label": "best τ on test", "value": f"{best_tau:.2f}" if best_tau is not None else "n/a"},
            {"label": "accuracy at best τ", "value": f"{_pct(best['accuracy'])}%" if best else "n/a"},
            {
                "label": "prior for comparison",
                "value": f"{_pct((test.get('distribution') or {}).get('majority_share'))}%",
            },
            {"label": "caution", "value": "tuning on test is not a licence to change it"},
        ],
    })

    # 4. Overfit: train vs held-out test.
    train_acc = (train.get("rule") or {}).get("accuracy_argmax")
    test_acc = (test.get("rule") or {}).get("accuracy_argmax")
    gap_pp = (train_acc - test_acc) * 100.0 if (train_acc is not None and test_acc is not None) else None
    out.append({
        "id": "overfit",
        "title": "In-sample performance does not survive the holdout",
        "claim": "The same booster is scored on the rows it was fitted on and on the sealed test "
                 "split. A large gap means the fit is memorising.",
        "verdict": VERDICT_SUPPORTED if (gap_pp is not None and gap_pp >= 10.0) else VERDICT_REJECTED,
        "split": "train → test",
        "chart": "split_table",
        "evidence": [
            {"label": "train accuracy", "value": f"{_pct(train_acc)}%" if train_acc is not None else "n/a"},
            {"label": "test accuracy", "value": f"{_pct(test_acc)}%" if test_acc is not None else "n/a"},
            {"label": "gap", "value": f"{gap_pp:+.1f} pp" if gap_pp is not None else "n/a", "threshold": "≥ 10 pp"},
        ],
    })

    # 5. Drift: is the model leaning on features that moved?
    top = [r for r in importance if r.get("in_top15")]
    drifting = [r for r in top if (r.get("psi") or 0.0) >= psi_threshold]
    share = (len(drifting) / len(top)) if top else None
    alert_level = VERDICT_INCONCLUSIVE if not top else (
        VERDICT_SUPPORTED if share is not None and share >= 0.30 else VERDICT_REJECTED
    )
    out.append({
        "id": "drift",
        "title": "Top features are drifting",
        "claim": "PSI from results/monitoring.json, joined onto the top-15 SHAP features. If the "
                 "features the model leans on are drifting, its behaviour will move too.",
        "verdict": alert_level,
        "split": "test",
        "chart": "importance",
        "evidence": [
            {"label": "drifting top-15", "value": f"{len(drifting)}/{len(top)}" if top else "n/a", "threshold": f"PSI ≥ {psi_threshold}"},
            {"label": "share", "value": f"{_pct(share)}%" if share is not None else "n/a", "threshold": "≥ 30%"},
            {"label": "calibration", "value": (calibration or {}).get("method") or "n/a"},
        ],
    })

    # 6. The prior itself is a baseline the model has to beat.
    majority = (test.get("distribution") or {}).get("majority_share")
    edge = (test_acc - majority) if (test_acc is not None and majority is not None) else None
    out.append({
        "id": "label_prior",
        "title": "What a constant prediction would score",
        "claim": "The largest class share is the bar. A model that predicts below it has learnt "
                 "the output shape without learning the discrimination.",
        "verdict": VERDICT_INFO,
        "split": "test",
        "chart": "distribution",
        "evidence": [
            {"label": "prior", "value": f"{_pct(majority)}%" if majority is not None else "n/a"},
            {"label": "model accuracy", "value": f"{_pct(test_acc)}%" if test_acc is not None else "n/a"},
            {"label": "edge", "value": f"{edge * 100:+.1f} pp" if edge is not None else "n/a"},
        ],
    })
    return out


def waterfall(contributions: list[dict], base_value: float) -> dict:
    """Order contributions so a waterfall can be drawn left-to-right."""
    rows = sorted(contributions, key=lambda row: abs(row.get("contribution") or 0.0), reverse=True)
    running = float(base_value or 0.0)
    for row in rows:
        running += float(row.get("contribution") or 0.0)
        row["cumulative"] = _r(running)
    return {"base_value": _r(base_value), "final": _r(running), "steps": rows}

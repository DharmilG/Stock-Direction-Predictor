from __future__ import annotations

import json

import pandas as pd

from ..config import load_settings
from ..utils.io import read_json


def _pct(x) -> str:
    return f"{100.0 * float(x):.2f}%" if x is not None else "n/a"


def build_final_report() -> dict:
    """Assemble the §12.2 final result block from results/ artifacts.

    Never recomputes correctness from raw frames when the ledger exists —
    predictions.parquet is the source of truth for per-row outcomes.
    Missing backtest output yields economics = n/a (evaluate runs before backtest).
    """
    settings = load_settings()
    paths = settings.paths
    metrics = read_json(paths.results / "metrics.json", {}) or {}
    backtest = read_json(paths.results / "backtest_metrics.json", {}) or {}

    n_test = int(metrics.get("n_test", 0))
    acc = metrics.get("accuracy")
    n_correct = int(round(acc * n_test)) if acc is not None and n_test else 0
    n_wrong = n_test - n_correct
    majority = metrics.get("majority_baseline_accuracy")
    linear = metrics.get("baseline_accuracy")
    edge_pp = None
    if acc is not None and majority is not None:
        edge_pp = (acc - majority) * 100.0
    beats = bool(metrics.get("beats_majority_baseline", False))

    # Signal-only slice at the configured confidence threshold.
    sig_total = sig_correct = None
    sig_acc = None
    try:
        pred_path = paths.results / "predictions.parquet"
        if pred_path.exists():
            df = pd.read_parquet(pred_path)
            if "confidence" not in df.columns and {"p_down", "p_flat", "p_up"} <= set(df.columns):
                df["confidence"] = df[["p_down", "p_flat", "p_up"]].max(axis=1)
            thr = float(settings.backtest_confidence_threshold)
            sig = df[df.get("confidence", 0) >= thr] if "confidence" in df.columns else df.iloc[0:0]
            sig_total = int(len(sig))
            if "correct" in sig.columns:
                sig_correct = int(sig["correct"].sum())
                sig_acc = float(sig["correct"].mean()) if len(sig) else None
    except Exception:
        pass

    test_start = metrics.get("test_start", "n/a")
    test_end = metrics.get("test_end", "")
    period = f"{test_start} → {test_end}" if test_end else str(test_start)

    economics = {
        "sharpe": backtest.get("sharpe"),
        "total_return": backtest.get("total_return"),
        "gross": backtest.get("gross_return_sum"),
        "costs": backtest.get("cost_paid"),
    }
    return {
        "period": period,
        "n_test": n_test,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "accuracy": acc,
        "majority": majority,
        "linear": linear,
        "edge_pp": edge_pp,
        "beats_majority_baseline": beats,
        "sig_total": sig_total,
        "sig_correct": sig_correct,
        "sig_acc": sig_acc,
        "sig_coverage": (sig_total / n_test) if sig_total is not None and n_test else None,
        "threshold": float(settings.backtest_confidence_threshold),
        "roc_auc": metrics.get("roc_auc_ovr_macro"),
        "mcc": metrics.get("mcc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "ece": metrics.get("ece"),
        **economics,
    }


def format_final_report(r: dict) -> str:
    ver = "✓ ABOVE BASELINE" if r["beats_majority_baseline"] else "✗ BELOW BASELINE"
    ece = r.get("ece")
    ece_s = f"{ece:.4f}" if isinstance(ece, (int, float)) else "n/a"
    lines = [
        "══════════════════════════════════════════════════════════",
        "  FINAL TEST RESULT",
        "══════════════════════════════════════════════════════════",
        f"  Test period            : {r['period']}",
        f"  Total test samples     : {r['n_test']}",
        "",
        f"  CORRECT PREDICTIONS    : {r['n_correct']}  /  {r['n_test']}        ({_pct(r['accuracy'])})",
        f"  WRONG PREDICTIONS      : {r['n_wrong']}  /  {r['n_test']}        ({_pct(1 - r['accuracy']) if r['accuracy'] is not None else 'n/a'})",
        "",
        "  ── Reference baselines ─────────────────────────────────",
        f"  Majority-class baseline:                     ({_pct(r['majority'])})",
        f"  Linear baseline        :                     ({_pct(r['linear'])})",
        "",
        f"  EDGE OVER MAJORITY     :                     ({r['edge_pp']:+.2f} pp)   {ver}" if r["edge_pp"] is not None else "  EDGE OVER MAJORITY     :                     (n/a)",
        "",
        f"  ── Signal-only (confidence >= {r['threshold']:.2f}) ────────────────────",
        f"  Signals emitted        : {r['sig_total']}  /  {r['n_test']}        ({_pct(r['sig_coverage'])})" if r["sig_total"] is not None else "  Signals emitted        : n/a",
        f"  CORRECT                :  {r['sig_correct']}  /  {r['sig_total']}        ({_pct(r['sig_acc'])})" if r["sig_total"] else "",
        "",
        "  ── Ranking & calibration ───────────────────────────────",
        f"  ROC-AUC (OvR macro)    : {r['roc_auc']:.4f}" if isinstance(r.get("roc_auc"), (int, float)) else "  ROC-AUC (OvR macro)    : n/a",
        f"  MCC                    : {r['mcc']:.4f}" if isinstance(r.get("mcc"), (int, float)) else "  MCC                    : n/a",
        f"  Balanced accuracy      : {r['balanced_accuracy']:.4f}" if isinstance(r.get("balanced_accuracy"), (int, float)) else "  Balanced accuracy      : n/a",
        f"  ECE                    : {ece_s}",
        "",
        "  ── Economics (after Indian costs) ──────────────────────",
        f"  Net Sharpe             : {r['sharpe']:.2f}" if isinstance(r.get("sharpe"), (int, float)) else "  Net Sharpe             : n/a (run backtest)",
        f"  Net total return       : {_pct(r['total_return'])}" if r.get("total_return") is not None else "  Net total return       : n/a (run backtest)",
        f"  Gross return           : {_pct(r['gross'])}" if r.get("gross") is not None else "",
        f"  Costs paid             : {_pct(r['costs'])}" if r.get("costs") is not None else "",
        "══════════════════════════════════════════════════════════",
    ]
    return "\n".join(line for line in lines if line != "" or True).rstrip() + "\n"


def print_final_report() -> bool:
    """Print the §12.2 block. Returns True iff the model beats the majority baseline."""
    import sys
    # Windows consoles default to cp1252, which cannot encode the box-drawing
    # characters. Prefer UTF-8; fall back to replace rather than crash.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    report = build_final_report()
    print(format_final_report(report), flush=True)
    return bool(report["beats_majority_baseline"])


def run_report() -> None:
    """CLI entry: print block, exit 1 when below the majority baseline (F-22 step 9)."""
    beats = print_final_report()
    if not beats:
        raise SystemExit(1)


if __name__ == "__main__":
    run_report()

from __future__ import annotations

"""Phase 3 — Thinking-cap micro-ablation.

CODE READY — UNTESTED
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_settings
from ..qpipeline.paths import Q3_LEDGER, Q3_RESULTS, Q3_TRACES, Q3_HEARTBEAT, Q3_GPU, Q3_MAIN_LOG, p0_from
from ..qpipeline.fio import atomic_append_jsonl, read_jsonl_safe, file_fingerprint, write_text_atomic
from ..data.q0_d1 import (
    LABEL_TO_VOTE,
    ThermalMonitor,
    ThermalState,
    _fmt_temp,
    _track_temp,
    gpu_temp_c,
    handle_hard_trip,
    thermal_gate_before_case,
)
from ..data.qwen_rater import rate_case_budgeted, write_trace_budgeted
from ..qpipeline import state as q_state
from ..qpipeline.contracts import load_q_track_config, phase_artifacts
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging


def _load_pool() -> pd.DataFrame:
    # Pool C lives in the phase-0 folder (single source of truth: qpipeline.paths).
    path = p0_from(load_settings().paths.root)["data"] / "pool_c.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Pool C not found: {path}")
    return pd.read_parquet(path)


def _sample_pool(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(pool) < n:
        raise ValueError(f"Pool C has only {len(pool)} rows; need {n}")
    return pool.sample(n=n, random_state=seed).reset_index(drop=True)


def _append_ledger_row(path: Path, row: dict) -> None:
    # Durable append (flush + fsync): a crash can never lose a completed pair.
    atomic_append_jsonl(path, row)


def _load_ledger(path: Path) -> pd.DataFrame:
    # Defensive read: a truncated line from a crash is quarantined, not fatal.
    rows = read_jsonl_safe(path, quarantine_dir=path.parent / "quarantine")
    return pd.DataFrame(rows)


def _mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar on paired correct/incorrect vectors."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    total = n01 + n10
    if total == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0, "note": "no discordant pairs"}
    try:
        from scipy.stats import binomtest

        result = binomtest(min(n01, n10), total, p=0.5)
        p_value = float(result.pvalue)
    except Exception:
        # Normal approximation fallback — never used when scipy is present.
        if total == 0:
            p_value = 1.0
        else:
            z = (abs(n01 - n10) - 1) / np.sqrt(total)
            p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / np.sqrt(2)))))
    return {"n01": n01, "n10": n10, "p_value": p_value}


def _budget_ledger_path() -> Path:
    return phase_artifacts()["phase3"]["ledger"]


def _results_path() -> Path:
    return phase_artifacts()["phase3"]["results"]


def _trace_dir() -> Path:
    return phase_artifacts()["phase3"]["trace_dir"]


def run_phase3(
    limit: int = 0,
    sample_size: int = 0,
    thinking_budget: int = 0,
    **_kwargs,
) -> dict:
    """Run the paired thinking-cap ablation. No Q0/Pool D writes."""
    cfg = load_q_track_config().get("phase3", {})
    n = int(sample_size or cfg.get("n", 1000))
    seed = int(cfg.get("seed", 43))
    budgets = [int(b) for b in cfg.get("budgets", [256, 512, 1024])]
    if thinking_budget:
        budgets = [int(thinking_budget)]
    if not budgets:
        raise ValueError("No thinking budgets configured")

    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(Q3_MAIN_LOG)
    pool = _sample_pool(_load_pool(), n, seed)
    if limit:
        pool = pool.head(int(limit))
    ledger_path = Path(_budget_ledger_path())
    trace_dir = Path(_trace_dir())
    sample_path = Q3_GPU
    heartbeat_path = Q3_HEARTBEAT

    # Config fingerprint: refuse to resume with a different configuration.
    fp = file_fingerprint([Q3_LEDGER]) if Q3_LEDGER.exists() else ""
    config_fp = dict_fingerprint({"n": n, "seed": seed, "budgets": budgets})
    manifest_path = Q3_RESULTS.parent / "q3_manifest.json"
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior.get("config_fingerprint") and prior["config_fingerprint"] != config_fp:
                raise RuntimeError(
                    "Phase 3 config changed since the last run. Use --no-resume "
                    "(delete the ledger) or restore the original config."
                )
        except json.JSONDecodeError:
            pass
    write_text_atomic(manifest_path, json.dumps(
        {"config_fingerprint": config_fp, "n": n, "seed": seed, "budgets": budgets}, indent=2))

    run_started_at = datetime.now(timezone.utc).isoformat()
    attempted = 0

    def hb(**fields) -> None:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": "phase3", "n_target": n, "budgets": budgets,
            "attempted": attempted, "ledger_rows": int(len(_load_ledger(ledger_path))),
            "harness_fail_rate": 0.0,
        }
        payload.update(fields)
        try:
            write_text_atomic(heartbeat_path, json.dumps(payload, indent=2, default=str))
        except Exception:
            pass
    hb(status="starting")
    state = ThermalState()
    existing = _load_ledger(ledger_path)
    done: set[tuple[str, int]] = set()
    if len(existing):
        done = {
            (str(row.case_id), int(row.thinking_budget))
            for row in existing.itertuples(index=False)
        }

    for _, row in pool.iterrows():
        case_id = str(row["case_id"])
        ctx = (
            f"5-day trend: {row['trend_5d']:.2%}, volume shock (z): {row['volume_z']:.2f}, "
            f"FinBERT prior: {row['finbert_score']:.2f}"
            if pd.notna(row["finbert_score"])
            else "5-day trend: unavailable, NO_NEWS FinBERT prior"
        )
        for budget in budgets:
            if (case_id, budget) in done:
                continue
            start_temp = thermal_gate_before_case(logger, state, case_id)
            monitor = ThermalMonitor(logger, case_id, sample_path)
            monitor.start()
            try:
                result, prompt = rate_case_budgeted(
                    symbol=str(row["symbol"]),
                    sector=str(row["sector"] or "unknown"),
                    pre_t_context=ctx,
                    article_text=str(row["text"]),
                    thinking_budget=budget,
                    abort_event=monitor.abort_event,
                )
            finally:
                monitor.stop()
                monitor.join(timeout=40)
            attempted += 1
            if monitor.max_temp_c is not None:
                _track_temp(state, monitor.max_temp_c)
            end_temp = monitor.last_temp_c if monitor.last_temp_c is not None else gpu_temp_c()
            _track_temp(state, end_temp)
            if result.thermal_abort or monitor.abort_event.is_set():
                state.aborts += 1
                trip_temp = monitor.abort_temp_c or end_temp
                result.thermal_temp_c = trip_temp
                result.fail_reason = result.fail_reason or "thermal-abort"
                write_trace_budgeted(
                    trace_dir, case_id, prompt, result, budget,
                    phase="phase3", extra={"budget": budget, "case_row": row.to_dict()},
                )
                logger.warning("Thermal abort on %s budget %d at %s.", case_id, budget, _fmt_temp(trip_temp))
                handle_hard_trip(logger, state, trip_temp, case_id, "phase3")
                continue
            trace_path = write_trace_budgeted(
                trace_dir, case_id, prompt, result, budget,
                phase="phase3", extra={"budget": budget, "case_row": row.to_dict()},
            )
            ledger_row = {
                "case_id": case_id,
                "symbol": str(row["symbol"]),
                "decision_date": pd.Timestamp(row["decision_date"]).date().isoformat(),
                "pool": "C",
                "phase": "phase3",
                "model_stage": "zero_shot",
                "thinking_budget": int(budget),
                "qwen_vote": result.vote,
                "qwen_sent": result.sentiment,
                "qwen_conf": result.confidence,
                "qwen_trace_len": result.think_len,
                "historical_label": row["historical_label"],
                "vote_matches_label": bool(result.vote == row["historical_label"]) if result.vote else False,
                "harness_fail": bool(result.harness_fail),
                "fail_reason": result.fail_reason,
                "trace_path": str(trace_path),
                "gpu_temp_start": start_temp,
                "gpu_temp_end": end_temp,
                "seconds": round(result.seconds, 3),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _append_ledger_row(ledger_path, ledger_row)
            done.add((case_id, budget))
            ledger_df = _load_ledger(ledger_path)
            hb(status="running", current_case=case_id, current_budget=budget,
               harness_fail_rate=float(ledger_df["harness_fail"].astype(bool).mean()) if len(ledger_df) else 0.0,
               last_case_id=case_id, last_budget=budget)
            logger.info("Phase3 checkpoint: case=%s budget=%d vote=%s", case_id, budget, result.vote)

    ledger = _load_ledger(ledger_path)
    agreement: dict[int, float] = {}
    judged_counts: dict[int, int] = {}
    comparisons: dict[str, dict] = {}
    if ledger.empty:
        valid_budgets: list[int] = []
    else:
        for budget in budgets:
            sub = ledger[ledger["thinking_budget"] == budget]
            judged = sub[(~sub["harness_fail"].astype(bool)) & sub["qwen_vote"].isin(["UP", "DOWN", "FLAT"])]
            judged_counts[budget] = int(len(judged))
            agreement[budget] = float(judged["vote_matches_label"].mean()) if len(judged) else float("nan")
        valid_budgets = [b for b in budgets if judged_counts.get(b, 0) > 0]

    if len(valid_budgets) > 1:
        for i, b1 in enumerate(valid_budgets):
            for b2 in valid_budgets[i + 1 :]:
                s1 = ledger[ledger["thinking_budget"] == b1].set_index("case_id")["vote_matches_label"]
                s2 = ledger[ledger["thinking_budget"] == b2].set_index("case_id")["vote_matches_label"]
                joined = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
                if len(joined) < 10:
                    comparisons[f"{b1}_vs_{b2}"] = {"n": int(len(joined)), "note": "too few paired cases"}
                    continue
                comparisons[f"{b1}_vs_{b2}"] = {
                    "n": int(len(joined)),
                    **_mcnemar_exact(joined["a"].to_numpy(), joined["b"].to_numpy()),
                }

    usable = {b: agreement[b] for b in valid_budgets if np.isfinite(agreement[b])}
    winner = max(usable, key=lambda b: (usable[b], -b)) if usable else None
    if winner is None:
        winner = 512

    report = {
        "phase": "phase3",
        "run_started_at": run_started_at,
        "n_target": n,
        "budgets": budgets,
        "judged_counts": judged_counts,
        "agreement": agreement,
        "mcnemar": comparisons,
        "winner_budget": winner,
        "pass": bool(usable and len(usable) == len(budgets)),
        "attempts": attempted,
        "ledger_rows": int(len(ledger)),
        "harness_fail_rate": float(ledger["harness_fail"].astype(bool).mean()) if len(ledger) else 0.0,
        "thermal": {
            "rests": state.rests,
            "hard_trips": state.hard_trips,
            "cooldowns": state.cooldowns,
            "aborts": state.aborts,
            "max_temp_c": state.max_temp_c,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(Path(_results_path()), report)
    hb(status="verdict", complete=True, pass_=bool(report["pass"]),
       winner_budget=winner, harness_fail_rate=report["harness_fail_rate"])
    write_text_atomic(Q3_DONE, json.dumps(report, indent=2, default=str))
    q_state.set_phase_status("phase3", "passed" if report["pass"] else "failed", result=report)
    logger.info("Phase3 winner budget=%s report=%s", winner, _results_path())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 thinking-cap ablation.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--thinking-budget", type=int, default=0)
    args = parser.parse_args()
    print(
        run_phase3(
            limit=args.limit,
            sample_size=args.sample_size,
            thinking_budget=args.thinking_budget,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Phase 7 — Nightly rater using the Phase 6 LoRA adapter.

CODE READY — UNTESTED
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_settings
from ..data.q0_d1 import ThermalMonitor, ThermalState, _track_temp, thermal_gate_before_case
from ..data.qwen_rater import rate_case_budgeted, write_trace_budgeted
from ..features.decision_date import decision_dates
from ..qpipeline import state as q_state
from ..qpipeline.contracts import load_q_track_config, phase_artifacts
from ..utils.io import atomic_write_json, atomic_write_parquet
from ..utils.logging import setup_logging


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")


def _case_id(symbol: str, decision_date: pd.Timestamp, text: str) -> str:
    digest = hashlib.md5(text[:200].encode(errors="ignore")).hexdigest()[:8]
    return f"{symbol}|{decision_date.date().isoformat()}|{digest}"


def build_nightly_cases(settings, symbols: list[str] | None = None) -> pd.DataFrame:
    """Build nightly candidate cases from raw news + announcements."""
    symbols = symbols or settings.symbols
    calendar = pd.DatetimeIndex(
        pd.read_parquet(settings.paths.root / "data" / "reference" / "trading_calendar.parquet").iloc[:, 0]
    )
    rows: list[dict] = []
    for symbol in symbols:
        safe = _safe_symbol(symbol)
        for source_dir in ("news", "announcements"):
            path = settings.paths.data_raw / source_dir / f"{safe}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if frame.empty or "published_utc" not in frame.columns:
                continue
            try:
                sector_map = json.loads((settings.paths.root / "data" / "reference" / "sector_map.json").read_text())
            except Exception:
                sector_map = {}
            sector = ""
            for sec, members in sector_map.items():
                if symbol in (members or []):
                    sector = str(sec)
                    break
            dd = decision_dates(frame, calendar, col="published_utc")
            frame = frame.copy()
            frame["decision_date"] = pd.to_datetime(dd).dt.tz_localize(None).dt.normalize()
            frame = frame.dropna(subset=["decision_date"])
            for _, row in frame.iterrows():
                text = (str(row.get("subject") or "") + ". " + str(row.get("text") or "")).strip(". ")
                rows.append(
                    {
                        "case_id": _case_id(symbol, pd.Timestamp(row["decision_date"]), text),
                        "symbol": symbol,
                        "safe_symbol": safe,
                        "sector": sector or "unknown",
                        "decision_date": pd.Timestamp(row["decision_date"]).normalize(),
                        "published_utc": str(row.get("published_utc")),
                        "text": text,
                        "source": source_dir,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["case_id"]).reset_index(drop=True)


def run_phase7(
    limit: int = 0,
    thinking_budget: int = 0,
    model: str = "",
    **_kwargs,
) -> dict:
    """Score nightly candidates through the LoRA adapter model."""
    cfg = load_q_track_config()
    nightly_cfg = cfg.get("nightly", {})
    model = model or cfg.get("models", {}).get("lora_model", "qwen3-8b-lora-r1")
    budget = int(thinking_budget or nightly_cfg.get("default_thinking_budget", 512))
    artifacts = phase_artifacts()
    ledger_path = Path(artifacts["phase7"]["ledger"])
    trace_dir = Path(nightly_cfg.get("trace_dir", "results/traces_q7"))
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "qwen_nightly.log")
    cases = build_nightly_cases(settings)
    if cases.empty:
        raise RuntimeError("No nightly cases found.")
    if limit:
        cases = cases.head(int(limit))
    done: set[str] = set()
    if ledger_path.exists():
        existing = pd.read_parquet(ledger_path)
        if len(existing) and "case_id" in existing.columns:
            done = set(existing["case_id"].astype(str))
    todo = cases[~cases["case_id"].astype(str).isin(done)]
    if todo.empty:
        logger.info("No new nightly cases.")
        return {"phase": "phase7", "new_cases": 0, "pass": True}

    state = ThermalState()
    sample_path = paths.logs / "qwen_nightly_gpu_5s.csv"
    new_rows: list[dict] = []
    for _, row in todo.iterrows():
        case_id = str(row["case_id"])
        ctx = "Nightly news item; no price context beyond decision_date."
        start_temp = thermal_gate_before_case(logger, state, case_id)
        monitor = ThermalMonitor(logger, case_id, sample_path)
        monitor.start()
        try:
            result, prompt = rate_case_budgeted(
                symbol=str(row["symbol"]),
                sector=str(row["sector"]),
                pre_t_context=ctx,
                article_text=str(row["text"]),
                thinking_budget=budget,
                model=model,
                abort_event=monitor.abort_event,
            )
        finally:
            monitor.stop()
            monitor.join(timeout=40)
        if result.thermal_abort or monitor.abort_event.is_set():
            state.aborts += 1
            continue
        if monitor.max_temp_c is not None:
            _track_temp(state, monitor.max_temp_c)
        trace_path = write_trace_budgeted(
            trace_dir, case_id, prompt, result, budget, model=model, phase="phase7"
        )
        new_rows.append(
            {
                "case_id": case_id,
                "symbol": row["symbol"],
                "safe_symbol": row["safe_symbol"],
                "sector": row["sector"],
                "decision_date": pd.Timestamp(row["decision_date"]).date().isoformat(),
                "published_utc": row["published_utc"],
                "text": row["text"],
                "source": row["source"],
                "model": model,
                "thinking_budget": budget,
                "qwen_vote": result.vote,
                "qwen_score": result.sentiment,
                "qwen_confidence": result.confidence,
                "qwen_trace_len": result.think_len,
                "harness_fail": bool(result.harness_fail),
                "fail_reason": result.fail_reason,
                "trace_path": str(trace_path),
                "seconds": round(result.seconds, 3),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("Nightly case=%s vote=%s fail=%s", case_id, result.vote, result.harness_fail)
        if len(new_rows) % 25 == 0 and new_rows:
            ledger = pd.DataFrame(new_rows)
            if ledger_path.exists():
                prev = pd.read_parquet(ledger_path)
                ledger = pd.concat([prev, ledger], ignore_index=True).drop_duplicates(subset=["case_id"])
            atomic_write_parquet(ledger, ledger_path)

    if new_rows:
        ledger = pd.DataFrame(new_rows)
        if ledger_path.exists():
            prev = pd.read_parquet(ledger_path)
            ledger = pd.concat([prev, ledger], ignore_index=True).drop_duplicates(subset=["case_id"])
        atomic_write_parquet(ledger, ledger_path)
        # Nightly feature refresh: build the per-symbol qwen feature parquet.
        calendar = pd.DatetimeIndex(
            pd.read_parquet(settings.paths.root / "data" / "reference" / "trading_calendar.parquet").iloc[:, 0]
        )
        trading_index = pd.DatetimeIndex(pd.to_datetime(ledger["decision_date"]).dt.tz_localize(None).dt.normalize()).unique()
        from ..features.qwen_features import save_qwen_features_panel

        feature_paths = save_qwen_features_panel(trading_index, calendar)
        report_extra = {"feature_paths": [str(p) for p in feature_paths]}
    else:
        report_extra = {"feature_paths": []}
    report = {
        "phase": "phase7",
        "new_cases": len(new_rows),
        "ledger_path": str(ledger_path),
        "model": model,
        "thinking_budget": budget,
        **report_extra,
        "pass": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(Path(artifacts["phase7"]["results"]), report)
    q_state.set_phase_status("phase7", "passed" if report["pass"] else "failed", result=report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 nightly rater.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()
    print(run_phase7(limit=args.limit, thinking_budget=args.thinking_budget, model=args.model))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Phases 4–5 — STaR builder, leakage filter, audit and keep-rate report.

CODE READY — UNTESTED
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_settings
from ..data.leakage_scanner import scan_candidate, scanner_version
from ..data.q0_d1 import LABEL_TO_VOTE, ThermalMonitor, ThermalState, _track_temp, thermal_gate_before_case
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


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_pool_a(
    sample_size: int = 0,
    seed: int = 43,
    **_kwargs,
) -> pd.DataFrame:
    """Build Pool A candidates from announcements + labels, strictly pre-cutoff."""
    cfg = load_q_track_config().get("star", {})
    sample_size = int(sample_size or cfg.get("default_sample_size", 5000))
    seed = int(seed or cfg.get("default_seed", 43))
    cutoff = pd.Timestamp(cfg.get("cutoff", "2023-08-24"))
    relevance_min = float(cfg.get("relevance_min", 0.5))

    settings = load_settings()
    paths = settings.paths
    calendar = pd.DatetimeIndex(
        pd.read_parquet(paths.root / "data" / "reference" / "trading_calendar.parquet").iloc[:, 0]
    )
    try:
        sector_map = json.loads((paths.root / "data" / "reference" / "sector_map.json").read_text())
    except Exception:
        sector_map = {}

    rows: list[dict] = []
    for symbol in settings.symbols:
        safe = _safe_symbol(symbol)
        ann_path = paths.data_raw / "announcements" / f"{safe}.parquet"
        ds_path = paths.data_processed / f"{safe}_dataset.parquet"
        if not ann_path.exists() or not ds_path.exists():
            continue
        ann = pd.read_parquet(ann_path)
        ds = pd.read_parquet(ds_path)
        if ann.empty or "published_utc" not in ann.columns or "target" not in ds.columns:
            continue
        labels = ds["target"].dropna().astype(int)
        label_by_date = {pd.Timestamp(t).normalize(): int(v) for t, v in labels.items()}
        dd = decision_dates(ann, calendar, col="published_utc")
        ann = ann.copy()
        ann["decision_date"] = pd.to_datetime(dd).dt.tz_localize(None).dt.normalize()
        ann = ann.dropna(subset=["decision_date"])
        ann = ann[ann["decision_date"] < cutoff]
        if ann.empty:
            continue
        if "relevance" in ann.columns:
            ann = ann[pd.to_numeric(ann["relevance"], errors="coerce").fillna(0.0) >= relevance_min]
            if ann.empty:
                continue
        price = ds[["close", "volume"]].copy() if {"close", "volume"} <= set(ds.columns) else None
        sector = ""
        for sec, members in (sector_map.items() if isinstance(sector_map, dict) else []):
            if symbol in (members or []):
                sector = str(sec)
                break
        for _, row in ann.iterrows():
            d = pd.Timestamp(row["decision_date"]).normalize()
            label = label_by_date.get(d)
            if label is None:
                continue
            text = (str(row.get("subject") or "") + ". " + str(row.get("text") or "")).strip(". ")
            fin = pd.to_numeric(pd.Series([row.get("finbert_score")]), errors="coerce").iloc[0]
            trend = volz = float("nan")
            if price is not None and d in price.index:
                try:
                    loc = price.index.get_indexer([d])[0]
                    if loc >= 6:
                        c0 = float(price["close"].iloc[loc - 5])
                        c1 = float(price["close"].iloc[loc])
                        trend = (c1 / c0 - 1.0) if c0 else float("nan")
                    if loc >= 21:
                        v = price["volume"].iloc[loc - 20 : loc]
                        volz = float((price["volume"].iloc[loc] - v.mean()) / (v.std() or np.nan))
                except Exception:
                    pass
            rows.append(
                {
                    "case_id": _case_id(symbol, d, text),
                    "symbol": symbol,
                    "safe_symbol": safe,
                    "sector": sector or "unknown",
                    "decision_date": d,
                    "published_utc": str(row.get("published_utc")),
                    "text": text,
                    "relevance": float(row.get("relevance", 1.0)),
                    "finbert_score": float(fin) if pd.notna(fin) else float("nan"),
                    "trend_5d": trend,
                    "volume_z": volz,
                    "historical_label": LABEL_TO_VOTE[label],
                    "pool": "A",
                }
            )

    if not rows:
        raise RuntimeError("Pool A is empty — check announcements and labels.")
    pool = pd.DataFrame(rows).drop_duplicates(subset=["case_id"]).reset_index(drop=True)
    if sample_size and len(pool) > sample_size:
        pool = pool.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    assert pool["decision_date"].max() < cutoff, "Pool A leaked past STAR_CUTOFF"
    pool_path = Path(phase_artifacts()["phase4"]["pool_a"])
    atomic_write_parquet(pool, pool_path)
    return pool


def generate_star_candidates(
    sample_size: int = 0,
    seed: int = 43,
    thinking_budget: int = 0,
    limit: int = 0,
    **_kwargs,
) -> dict:
    """Generate STaR candidates from Pool A. Resumable per case."""
    cfg = load_q_track_config().get("star", {})
    sample_size = int(sample_size or cfg.get("default_sample_size", 5000))
    seed = int(seed or cfg.get("default_seed", 43))
    cutoff = pd.Timestamp(cfg.get("cutoff", "2023-08-24"))
    relevance_min = float(cfg.get("relevance_min", 0.5))
    pool_path = Path(phase_artifacts()["phase4"]["pool_a"])
    if not pool_path.exists():
        pool = build_pool_a(sample_size=sample_size, seed=seed)
    else:
        pool = pd.read_parquet(pool_path)
        if sample_size and len(pool) > sample_size:
            pool = pool.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    assert pool["decision_date"].max() < cutoff, "STAR_CUTOFF violated before generation"
    if "relevance" in pool.columns:
        pool = pool[pd.to_numeric(pool["relevance"], errors="coerce").fillna(0.0) >= relevance_min]
    if limit:
        pool = pool.head(int(limit))

    p3_results_path = Path(phase_artifacts()["phase3"]["results"])
    p3_winner = 0
    if p3_results_path.exists():
        try:
            p3_winner = int(json.loads(p3_results_path.read_text(encoding="utf-8")).get("winner_budget") or 0)
        except Exception:
            p3_winner = 0
    budget = int(thinking_budget or p3_winner or 512)
    candidates_path = Path(phase_artifacts()["phase4"]["candidates"])
    trace_dir = Path(phase_artifacts()["phase4"]["trace_dir"])
    done = {str(row.get("case_id")) for row in _read_jsonl(candidates_path)}

    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "star_builder.log")
    sample_path = paths.logs / "star_gpu_5s.csv"
    state = ThermalState()
    attempted = 0
    harness_fail_count = 0
    for _, row in pool.iterrows():
        case_id = str(row["case_id"])
        if case_id in done:
            continue
        ctx = (
            f"5-day trend: {row['trend_5d']:.2%}, volume shock (z): {row['volume_z']:.2f}, "
            f"FinBERT prior: {row['finbert_score']:.2f}"
            if pd.notna(row["finbert_score"])
            else "5-day trend: unavailable, NO_NEWS FinBERT prior"
        )
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
                abort_event=monitor.abort_event,
            )
        finally:
            monitor.stop()
            monitor.join(timeout=40)
        attempted += 1
        if monitor.max_temp_c is not None:
            _track_temp(state, monitor.max_temp_c)
        if result.thermal_abort or monitor.abort_event.is_set():
            state.aborts += 1
            continue
        if result.harness_fail:
            harness_fail_count += 1
        trace_path = write_trace_budgeted(
            trace_dir, case_id, prompt, result, budget,
            phase="phase4", extra={"case_row": row.to_dict()},
        )
        candidate = {
            "case_id": case_id,
            "symbol": str(row["symbol"]),
            "decision_date": pd.Timestamp(row["decision_date"]).date().isoformat(),
            "historical_label": row["historical_label"],
            "sector": row["sector"],
            "prompt": prompt,
            "thinking": result.thinking,
            "content": result.content,
            "vote": result.vote,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "trace_len": result.think_len,
            "harness_fail": bool(result.harness_fail),
            "fail_reason": result.fail_reason,
            "trace_path": str(trace_path),
            "thinking_budget": budget,
            "seconds": round(result.seconds, 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(candidates_path, candidate)
        done.add(case_id)
        logger.info("Phase4 candidate: case=%s vote=%s fail=%s", case_id, result.vote, result.harness_fail)

    candidates_written = len(_read_jsonl(candidates_path))
    return {
        "phase": "phase4",
        "attempted": attempted,
        "harness_fail_count": harness_fail_count,
        "harness_fail_rate": float(harness_fail_count / attempted) if attempted else 0.0,
        "candidates_path": str(candidates_path),
        "budget": budget,
        "pool_size": int(len(pool)),
        "candidates_written": candidates_written,
        "pass": bool(candidates_written > 0),
    }


def filter_star_candidates() -> dict:
    """Phase 5: outcome filter + leakage scanner + audit sample/keep-rate."""
    cfg = load_q_track_config().get("star", {})
    audit_n = int(cfg.get("audit_n", 200))
    artifacts = phase_artifacts()
    candidates = _read_jsonl(Path(artifacts["phase4"]["candidates"]))
    if not candidates:
        raise FileNotFoundError("No STaR candidates found — run Phase 4 first.")
    kept_rows: list[dict] = []
    dropped_rows: list[dict] = []
    scanner_v = scanner_version()
    for row in candidates:
        reason: list[str] = []
        if bool(row.get("harness_fail")):
            reason.append("harness_fail")
        vote = row.get("vote")
        label = row.get("historical_label")
        if not reason and vote not in {"UP", "DOWN", "FLAT"}:
            reason.append("invalid_vote")
        if not reason and vote != label:
            reason.append("outcome_mismatch")
        hits = scan_candidate(row)
        if hits:
            reason.append("leakage:" + ",".join(sorted({h["class"] for h in hits})))
        enriched = dict(row)
        enriched["scanner_version"] = scanner_v
        enriched["leak_hits"] = hits
        enriched["kept_for_training"] = False
        if reason:
            enriched["drop_reason"] = ";".join(reason)
            dropped_rows.append(enriched)
        else:
            enriched["kept_for_training"] = True
            enriched["drop_reason"] = ""
            kept_rows.append(enriched)

    kept_path = Path(artifacts["phase5"]["kept"])
    dropped_path = Path(artifacts["phase5"]["dropped"])
    _write_jsonl(kept_path, kept_rows)
    _write_jsonl(dropped_path, dropped_rows)

    audit_path = Path(artifacts["phase5"]["audit_template"])
    audit_results_path = Path(artifacts["phase5"]["audit_results"])
    audit_sample = kept_rows[: min(audit_n, len(kept_rows))]
    audit_df = pd.DataFrame(
        [
            {
                "case_id": r["case_id"],
                "symbol": r["symbol"],
                "decision_date": r["decision_date"],
                "historical_label": r["historical_label"],
                "vote": r["vote"],
                "trace_len": r["trace_len"],
                "thinking": r["thinking"],
                "audit_verdict": "",
                "quality_score": "",
                "notes": "",
            }
            for r in audit_sample
        ]
    )
    audit_df.to_csv(audit_path, index=False)

    audit_results: dict[str, dict] = {}
    if audit_results_path.exists():
        results_df = pd.read_csv(audit_results_path)
        for _, r in results_df.iterrows():
            audit_results[str(r.get("case_id"))] = {
                "audit_verdict": str(r.get("audit_verdict") or ""),
                "quality_score": r.get("quality_score"),
                "notes": r.get("notes"),
            }
    for row in kept_rows:
        info = audit_results.get(str(row["case_id"]))
        if info:
            row["audit_verdict"] = info.get("audit_verdict")
            row["quality_score"] = info.get("quality_score")
            row["audit_notes"] = info.get("notes")

    keep_rate_by_class: dict[str, dict] = {}
    label_series = pd.Series([r["historical_label"] for r in candidates])
    kept_series = pd.Series([r["historical_label"] for r in kept_rows])
    for label in ("DOWN", "FLAT", "UP"):
        total = int((label_series == label).sum())
        kept = int((kept_series == label).sum())
        keep_rate_by_class[label] = {
            "candidates": total,
            "kept": kept,
            "keep_rate": (kept / total) if total else None,
        }

    if len(keep_rate_by_class["DOWN"]["candidates"]) and len(keep_rate_by_class["UP"]["candidates"]):
        p_down = keep_rate_by_class["DOWN"]["keep_rate"] or 0.0
        p_up = keep_rate_by_class["UP"]["keep_rate"] or 0.0
        n_down = keep_rate_by_class["DOWN"]["candidates"]
        n_up = keep_rate_by_class["UP"]["candidates"]
        se = math.sqrt(p_down * (1 - p_down) / n_down + p_up * (1 - p_up) / n_up) if n_down and n_up else float("nan")
        keep_rate_by_class["down_minus_up_2se"] = 2 * se
    else:
        keep_rate_by_class["down_minus_up_2se"] = None

    audit_fail_rate = None
    if audit_results:
        audited = [row for row in kept_rows if str(row.get("case_id")) in audit_results]
        fails = [row for row in audited if str(row.get("audit_verdict")).lower() in {"fail", "failed", "reject"}]
        audit_fail_rate = float(len(fails) / len(audited)) if audited else None

    report = {
        "phase": "phase5",
        "candidates": len(candidates),
        "kept": len(kept_rows),
        "dropped": len(dropped_rows),
        "keep_rate_overall": len(kept_rows) / len(candidates) if candidates else None,
        "keep_rate_by_class": keep_rate_by_class,
        "scanner_version": scanner_v,
        "audit_sample_n": len(audit_df),
        "audit_results_present": bool(audit_results),
        "audit_fail_rate": audit_fail_rate,
        "audit_gate_pass": bool(audit_fail_rate is None or audit_fail_rate <= 0.30),
        "kept_path": str(kept_path),
        "dropped_path": str(dropped_path),
        "audit_template_path": str(audit_path),
        "audit_results_path": str(audit_results_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(Path(artifacts["phase5"]["report"]), report)
    return report


def _split_train_val(kept_path: Path, val_path: Path, frac: float = 0.05, seed: int = 43) -> dict:
    rows = _read_jsonl(kept_path)
    if not rows:
        return {"train_rows": 0, "val_rows": 0}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n_val = max(1, int(len(rows) * frac))
    val_idx = set(idx[:n_val].tolist())
    train_rows = [row for i, row in enumerate(rows) if i not in val_idx]
    val_rows = [row for i, row in enumerate(rows) if i in val_idx]
    _write_jsonl(kept_path, train_rows)
    _write_jsonl(val_path, val_rows)
    return {"train_rows": len(train_rows), "val_rows": len(val_rows)}


def run_phase4(
    limit: int = 0,
    sample_size: int = 0,
    thinking_budget: int = 0,
    **_kwargs,
) -> dict:
    result = generate_star_candidates(
        limit=limit, sample_size=sample_size, thinking_budget=thinking_budget
    )
    q_state.set_phase_status("phase4", "passed" if result.get("pass") else "failed", result=result)
    return result


def run_phase5(**_kwargs) -> dict:
    report = filter_star_candidates()
    cfg = load_q_track_config().get("phase6", {})
    val_path = Path(load_q_track_config().get("phase6", {}).get("eval_jsonl", "data/llm/star_val.jsonl"))
    split = _split_train_val(
        Path(phase_artifacts()["phase5"]["kept"]), val_path, frac=0.05, seed=int(cfg.get("seed", 43))
    )
    report["train_val_split"] = split
    atomic_write_json(Path(phase_artifacts()["phase5"]["report"]), report)
    pass_gate = bool(report.get("audit_gate_pass", True))
    report["pass"] = pass_gate
    q_state.set_phase_status("phase5", "passed" if pass_gate else "failed", result=report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4/5 STaR builder.")
    parser.add_argument("--build-pool", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--thinking-budget", type=int, default=0)
    args = parser.parse_args()
    if args.build_pool or args.all:
        print(build_pool_a(sample_size=args.sample_size))
    if args.generate or args.all:
        print(generate_star_candidates(limit=args.limit, sample_size=args.sample_size, thinking_budget=args.thinking_budget))
    if args.filter or args.all:
        print(filter_star_candidates())
    if not any([args.build_pool, args.generate, args.filter, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()

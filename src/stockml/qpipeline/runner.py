from __future__ import annotations

"""Gate-enforcing runner for the Track Q phases.

CODE READY — UNTESTED
"""

import argparse
from typing import Any

from .contracts import PHASES, q0_gate_ok, q0_summary
from . import state


def _q0_block() -> None:
    if not q0_gate_ok():
        summary = q0_summary()
        raise RuntimeError(
            "Q0-D1 gate is not PASS. Refusing to run any Track Q phase. "
            f"Q0 summary: {summary}"
        )


def _phase_gate(phase: str) -> None:
    ok, reason = state.gate_ready(phase)
    if not ok:
        raise RuntimeError(f"Phase gate failed for {phase}: {reason}")


def run_phase(phase: str, **kwargs: Any) -> dict[str, Any]:
    """Run one phase only if Q0 passed and all prior phases passed."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    _q0_block()
    _phase_gate(phase)
    if phase == "phase8" and state.phase_status(phase) != "pending":
        raise RuntimeError("Phase 8 is one-shot only and has already been attempted.")
    state.set_phase_status(phase, "running")
    try:
        if phase == "phase3":
            from ..evaluation.q3_thinking_cap import run_phase3

            result = run_phase3(**kwargs)
        elif phase == "phase4":
            from ..data.star_builder import run_phase4

            result = run_phase4(**kwargs)
        elif phase == "phase5":
            from ..data.star_builder import run_phase5

            result = run_phase5(**kwargs)
        elif phase == "phase6_probe":
            from ..training.qwen_qlora import run_phase6

            result = run_phase6(probe=True, **kwargs)
        elif phase == "phase6_train":
            from ..training.qwen_qlora import run_phase6

            result = run_phase6(**kwargs)
        elif phase == "phase7":
            from ..data.qwen_nightly import run_phase7

            result = run_phase7(**kwargs)
        elif phase == "phase8":
            from ..evaluation.q8_ablation import run_phase8

            result = run_phase8(**kwargs)
        else:  # pragma: no cover
            raise ValueError(f"No runner wired for phase: {phase}")
    except Exception as exc:
        state.set_phase_status(phase, "failed", error=str(exc))
        raise
    passed = bool(result.get("pass", True)) if isinstance(result, dict) else True
    state.set_phase_status(phase, "passed" if passed else "failed", result=result)
    if not passed:
        raise RuntimeError(f"Phase {phase} completed but did not pass: {result}")
    return result


def status() -> dict[str, Any]:
    return {
        "q0": q0_summary(),
        "phases": state.load_state()["phases"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track Q phases with gates.")
    parser.add_argument("--phase", choices=PHASES, help="Phase to run.", default=None)
    parser.add_argument("--status", action="store_true", help="Print phase status and Q0 gate.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row/case limit for the phase.")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional sample size override.")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Optional thinking budget override.")
    args = parser.parse_args()

    if args.status:
        print(status())
        return
    if not args.phase:
        parser.print_help()
        return

    if args.phase:
        kwargs: dict[str, Any] = {}
        if args.limit:
            kwargs["limit"] = args.limit
        if args.sample_size:
            kwargs["sample_size"] = args.sample_size
        if args.thinking_budget:
            kwargs["thinking_budget"] = args.thinking_budget
        print(run_phase(args.phase, **kwargs))


if __name__ == "__main__":
    main()

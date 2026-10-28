from __future__ import annotations

"""10-step dummy pre-probe before Phase 4 generation is approved.

CODE READY — UNTESTED
"""

import argparse

from stockml.training.qwen_qlora import run_phase6


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 10-step dummy pre-probe.")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    print(run_phase6(probe=True, max_steps=args.max_steps))


if __name__ == "__main__":
    main()

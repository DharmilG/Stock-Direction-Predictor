"""Command-line entry point for the CV lane.

All commands are idempotent and resumable. Run from the CV directory:
    python -m chartsense.cli --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from .config import (
    CV_ROOT,
    LABELS_CSV,
    QUARANTINE_DIR,
    REAL_LABELS_CSV,
    REAL_DIR,
    SYNTHETIC_DIR,
    ensure_dirs,
)
from .fault import GracefulStop, sweep_stale_temps
from .fault.state import read_state


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _run_with_stop(function, stop: GracefulStop, **kwargs):
    stop.install()
    try:
        return function(stop=stop, **kwargs)
    finally:
        stop.restore()


def _cmd_status(_args) -> int:
    from .adapter import get_status

    _print(get_status())
    return 0


def _cmd_doctor(args) -> int:
    ensure_dirs()
    moved = sweep_stale_temps(CV_ROOT, max_age_seconds=args.max_age, quarantine_dir=QUARANTINE_DIR)
    _print({"moved_to_quarantine": len(moved), "samples": moved[:10]})
    return 0


def _cmd_generate_synthetic(args) -> int:
    from .data import generate_dataset

    stop = GracefulStop()
    result = _run_with_stop(
        generate_dataset,
        stop,
        n_per_pattern=args.n_per_pattern,
        resume=args.resume,
        seed=args.seed,
    )
    if stop.requested:
        result["stopped"] = True
    _print(result)
    return 0 if result.get("failed", 0) == 0 else 2


def _cmd_generate_own(args) -> int:
    from .data import build_own_dataset

    stop = GracefulStop()
    result = _run_with_stop(
        build_own_dataset,
        stop,
        limit=args.limit,
        symbols=args.symbols or None,
        resume=args.resume,
        seed=args.seed,
    )
    if stop.requested:
        result["stopped"] = True
    _print(result)
    return 0 if result.get("failed", 0) == 0 else 2


def _cmd_generate_all(args) -> int:
    code = _cmd_generate_synthetic(args)
    if code not in (0,):
        return code
    return _cmd_generate_own(args)


def _cmd_validate(_args) -> int:
    ensure_dirs()
    problems: list[str] = []
    counts: dict[str, int] = {}
    for csv_path, image_dir in ((LABELS_CSV, SYNTHETIC_DIR), (REAL_LABELS_CSV, REAL_DIR)):
        if not csv_path.exists():
            problems.append(f"missing labels file: {csv_path}")
            continue
        import pandas as pd

        frame = pd.read_csv(csv_path)
        counts[csv_path.name] = len(frame)
        for filename in frame["filename"]:
            path = image_dir / filename
            if not path.exists():
                problems.append(f"missing image: {path}")
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                problems.append(f"invalid image {path}: {exc}")
    _print({"counts": counts, "problems": problems[:50], "problem_count": len(problems)})
    return 1 if problems else 0


def _cmd_audit_labels(args) -> int:
    """Verify that each image ends at T and its label uses only T+1..T+5."""
    import pandas as pd

    from stockml.config import load_settings
    from stockml.targets.triple_barrier import triple_barrier_labels

    if not REAL_LABELS_CSV.exists():
        _print({"checked": 0, "problems": ["no real labels file"]})
        return 1
    settings = load_settings()
    frame = pd.read_csv(REAL_LABELS_CSV)
    sample = frame.sample(n=min(args.sample, len(frame)), random_state=args.seed)
    problems: list[str] = []
    checked = 0
    for symbol, group in sample.groupby("symbol"):
        path = settings.paths.data_raw / f"{symbol}.parquet"
        if not path.exists():
            problems.append(f"missing parquet for {symbol}")
            continue
        try:
            price = pd.read_parquet(path).sort_index()
        except Exception as exc:
            problems.append(f"unreadable parquet for {symbol}: {exc}")
            continue
        try:
            labels = triple_barrier_labels(
                price,
                horizon=settings.triple_barrier_horizon,
                volatility_window=settings.triple_barrier_vol_window,
                upper_multiplier=settings.triple_barrier_up_mult,
                lower_multiplier=settings.triple_barrier_down_mult,
                flat_band=settings.triple_barrier_flat_band,
            )
        except Exception as exc:
            problems.append(f"label error for {symbol}: {exc}")
            continue
        for _, row in group.iterrows():
            timestamp = pd.Timestamp(row["date"])
            image = REAL_DIR / row["filename"]
            if not image.exists():
                problems.append(f"missing image {image}")
                continue
            if timestamp not in price.index or timestamp not in labels.index:
                problems.append(f"{symbol} {timestamp.date()} not in price/labels index")
                continue
            event_start = pd.Timestamp(labels.loc[timestamp, "event_start"])
            if event_start <= timestamp:
                problems.append(f"{symbol} {timestamp.date()}: label window starts at or before image end")
            checked += 1
    _print({"checked": checked, "problems": problems[:20], "problem_count": len(problems)})
    return 1 if problems else 0


def _cmd_train(args) -> int:
    from .training import train_stage1, train_stage2

    stop = GracefulStop()
    function = train_stage1 if args.stage == 1 else train_stage2
    lr = args.lr if args.lr is not None else (1e-3 if args.stage == 1 else 1e-4)
    kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "resume": args.resume,
        "seed": args.seed,
    }
    if args.stage == 2:
        kwargs.update(
            reinit_head=args.reinit_head,
            unfreeze_layer4=args.unfreeze_layer4,
            label_smoothing=args.label_smoothing,
        )
    result = _run_with_stop(function, stop, **kwargs)
    if stop.requested:
        result["stopped"] = True
    _print(result)
    return 0


def _cmd_train_all(args) -> int:
    code = _cmd_train(args)
    if code != 0:
        return code
    args.stage = 2
    return _cmd_train(args)


def _cmd_evaluate(_args) -> int:
    from .training import evaluate

    _print(evaluate())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chartsense.cli", description="CV lane (code-only integration)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show readiness and stage progress").set_defaults(func=_cmd_status)
    doctor = sub.add_parser("doctor", help="quarantine stale temp files")
    doctor.add_argument("--max-age", type=int, default=3600)
    doctor.set_defaults(func=_cmd_doctor)

    synthetic = sub.add_parser("generate-synthetic", help="generate synthetic pattern charts")
    synthetic.add_argument("--n-per-pattern", type=int, default=500)
    synthetic.add_argument("--resume", dest="resume", action="store_true", default=True, help="resume completed images (default)")
    synthetic.add_argument("--no-resume", dest="resume", action="store_false")
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.set_defaults(func=_cmd_generate_synthetic)

    own = sub.add_parser("generate-own", help="generate charts from this platform's OHLCV parquet files")
    own.add_argument("--limit", type=int, default=3000)
    own.add_argument("--symbols", nargs="*", help="optional symbol filter, e.g. ADANIPORTS.NS")
    own.add_argument("--resume", dest="resume", action="store_true", default=True, help="resume completed images (default)")
    own.add_argument("--no-resume", dest="resume", action="store_false")
    own.add_argument("--seed", type=int, default=42)
    own.set_defaults(func=_cmd_generate_own)

    all_gen = sub.add_parser("generate-all", help="generate synthetic then own-data charts")
    all_gen.add_argument("--n-per-pattern", type=int, default=500)
    all_gen.add_argument("--limit", type=int, default=3000)
    all_gen.add_argument("--symbols", nargs="*")
    all_gen.add_argument("--resume", dest="resume", action="store_true", default=True, help="resume completed images (default)")
    all_gen.add_argument("--no-resume", dest="resume", action="store_false")
    all_gen.add_argument("--seed", type=int, default=42)
    all_gen.set_defaults(func=_cmd_generate_all)

    sub.add_parser("validate", help="verify every labeled image exists and decodes").set_defaults(func=_cmd_validate)

    audit = sub.add_parser("audit-labels", help="verify image windows end before their label window")
    audit.add_argument("--sample", type=int, default=60)
    audit.add_argument("--seed", type=int, default=42)
    audit.set_defaults(func=_cmd_audit_labels)

    train = sub.add_parser("train", help="train a stage")
    train.add_argument("--stage", type=int, choices=(1, 2), required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=None, help="stage 1 defaults to 1e-3, stage 2 to 1e-4")
    train.add_argument("--resume", dest="resume", action="store_true", default=True, help="resume from last checkpoint (default)")
    train.add_argument("--no-resume", dest="resume", action="store_false")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--reinit-head", action="store_true", help="stage 2: reinitialize direction head from stage 1 prior")
    train.add_argument("--unfreeze-layer4", action="store_true", help="stage 2: also fine-tune ResNet18 layer4 at lr/10")
    train.add_argument("--label-smoothing", type=float, default=0.0, help="cross-entropy label smoothing")
    train.set_defaults(func=_cmd_train)

    train_all = sub.add_parser("train-all", help="train stage 1 then stage 2")
    train_all.add_argument("--stage", type=int, choices=(1, 2), default=1)
    train_all.add_argument("--epochs", type=int, default=5)
    train_all.add_argument("--batch-size", type=int, default=32)
    train_all.add_argument("--lr", type=float, default=None, help="stage 1 defaults to 1e-3, stage 2 to 1e-4")
    train_all.add_argument("--resume", dest="resume", action="store_true", default=True, help="resume from last checkpoint (default)")
    train_all.add_argument("--no-resume", dest="resume", action="store_false")
    train_all.add_argument("--seed", type=int, default=42)
    train_all.add_argument("--reinit-head", action="store_true")
    train_all.add_argument("--unfreeze-layer4", action="store_true")
    train_all.add_argument("--label-smoothing", type=float, default=0.0)
    train_all.set_defaults(func=_cmd_train_all)

    sub.add_parser("evaluate", help="evaluate the serving checkpoint").set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

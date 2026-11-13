from __future__ import annotations

import argparse

from .backtest.backtest import run_backtest
from .data.news_fetch import run_news_fetch
from .data.fetch_data import run_fetch
from .data.bhavcopy import run_bhavcopy_fetch
from .data.trading_calendar import run_build_calendar
from .data.announcements import run_announcement_fetch, run_ann_scoring
from .data.bulkdeals import run_bulk_fetch
from .data.gdelt_bq import run_gdelt_bq_fetch
from .data.heal import run_heal_holes
from .data.refresh import run_refresh
from .data.rss import run_rss_fetch
from .evaluation.down_flag import run_down_flag
from .evaluation.evaluate import run_evaluation
from .evaluation.ablation import run_ablation
from .evaluation.final_report import run_report
from .evaluation.model_report import run_model_report
from .evaluation.flat_band_ablation import run_flat_band_ablation
from .evaluation.flat_threshold import run_flat_threshold
from .evaluation.u2_d1 import run_u2_d1
from .evaluation.u3_d1 import run_u3_d1
from .evaluation.economics import run_economics
from .evaluation.i01_d1 import run_i01_d1
from .evaluation.label_grid import run_label_grid
from .evaluation.up_autopsy import run_up_autopsy
from .features.cross_sectional import run_cross_sectional
from .features.feature_engineering import run_feature_engineering
from .inference.predict import run_prediction
from .inference.explain import run_explain
from .models.baseline import train_baseline
from .models.lightgbm_train import run_training
from .monitoring.monitor import run_monitor
from .pipeline import run_all
from .targets.build_dataset import run_build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-oriented stock direction ML platform")
    parser.add_argument("command", nargs="?", default="all", choices=[
        "all", "fetch", "heal", "refresh", "train-fresh", "bhavcopy", "calendar", "announcements", "ann-score",
        "news", "features", "ablation", "cross-sectional", "labels", "baseline", "train", "evaluate", "backtest",
        "monitor", "predict", "report", "model-report", "explain", "flat-ablation", "up-autopsy", "flat-threshold",
        "u2-d1", "u3-d1",
        "economics", "label-grid", "i01-d1", "bulkdeals", "down-flag", "rss", "gdelt-bq",
    ])
    parser.add_argument("--force", action="store_true", help="Re-run a single-shot backfill (gdelt-bq).")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Print the plan and change nothing (refresh, heal).")
    parser.add_argument("--skip-news", action="store_true", dest="skip_news", help="Refresh without the news steps.")
    parser.add_argument("--skip-announcements", action="store_true", dest="skip_announcements", help="Refresh without NSE announcements.")
    parser.add_argument("--ann-score", action="store_true", dest="ann_score", help="Also re-score announcements with FinBERT (slow).")
    parser.add_argument("--include-gdelt", action="store_true", dest="include_gdelt", help="Refresh including the GDELT BigQuery scan.")
    parser.add_argument("--no-heal", action="store_true", dest="skip_heal", help="Refresh without repairing internal price holes.")
    parser.add_argument("--symbol", default=None, help="Single symbol for explain (default: every symbol in the universe).")
    parser.add_argument("--top-k", type=int, default=12, dest="top_k", help="Features per class in explain (default: 12).")
    args = parser.parse_args()
    mapping = {
        "all": run_all, "fetch": run_fetch, "bhavcopy": run_bhavcopy_fetch,
        "calendar": run_build_calendar, "announcements": run_announcement_fetch,
        "ann-score": run_ann_scoring, "bulkdeals": run_bulk_fetch,
        "news": run_news_fetch, "features": run_feature_engineering,
        "cross-sectional": run_cross_sectional, "labels": run_build_dataset, "baseline": train_baseline,
        "train": run_training, "evaluate": run_evaluation, "backtest": run_backtest,
        "monitor": run_monitor, "predict": run_prediction, "report": run_report,
        "model-report": run_model_report,
        "explain": lambda: run_explain(symbol=args.symbol, top_k=args.top_k),
        "ablation": run_ablation,
        "flat-ablation": run_flat_band_ablation, "up-autopsy": run_up_autopsy,
        "flat-threshold": run_flat_threshold, "u2-d1": run_u2_d1, "u3-d1": run_u3_d1,
        "economics": run_economics, "label-grid": run_label_grid, "i01-d1": run_i01_d1,
        "bulkdeals": run_bulk_fetch, "down-flag": run_down_flag, "rss": run_rss_fetch,
        "gdelt-bq": lambda: run_gdelt_bq_fetch(force=args.force),
        "heal": lambda: run_heal_holes(dry_run=args.dry_run),
        "refresh": lambda: run_refresh(
            skip_heal=args.skip_heal,
            skip_news=args.skip_news,
            skip_gdelt=not args.include_gdelt,
            skip_announcements=args.skip_announcements,
            skip_ann_score=not args.ann_score,
            dry_run=args.dry_run,
        ),
        "train-fresh": lambda: run_refresh(
            skip_heal=args.skip_heal,
            skip_news=args.skip_news,
            skip_gdelt=not args.include_gdelt,
            skip_announcements=args.skip_announcements,
            # Retraining needs the announcement scores; they are cached once scored.
            skip_ann_score=False,
            retrain=True,
        ),
    }
    mapping[args.command]()


if __name__ == "__main__":
    main()

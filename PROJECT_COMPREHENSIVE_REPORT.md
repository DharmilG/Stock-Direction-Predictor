# Stock Direction Platform - Complete Technical Report (Detailed Answers)

Generated: October 28, 2026

## Table of Contents
1. [Project Overview](#project-overview)
2. [What model are we using?](#what-model-are-we-using)
3. [What ML method did we use earlier?](#what-ml-method-did-we-use-earlier)
4. [What dataset are we using?](#what-dataset-are-we-using)
5. [Why that format to store data and not CSV?](#why-that-format-to-store-data-and-not-csv)
6. [What embeddings are we using, are we using embedding for our ML model or not, why?](#what-embeddings-are-we-using-are-we-using-embedding-for-our-ml-model-or-not-why)
7. [Entry point of our project and how to check it?](#entry-point-of-our-project-and-how-to-check-it)
8. [How to run and test the ML model of LightGBM?](#how-to-run-and-test-the-ml-model-of-lightgbm)
9. [Why are we using LightGBM?](#why-are-we-using-lightgbm)
10. [How to check the trained dataset and etc?](#how-to-check-the-trained-dataset-and-etc)
11. [Current state - Q0 training](#current-state---q0-training)
12. [Data flow and storage structure (explained)](#data-flow-and-storage-structure-explained)
13. [Important concepts explained in simple words](#important-concepts-explained-in-simple-words)
14. [Quick reference commands](#quick-reference-commands)

## Project Overview
This is a production-oriented stock direction prediction system for Indian equities (NIFTY 50). The goal is to predict whether a stock will go UP, stay FLAT, or go DOWN over the next few days. Built with leakage prevention as a core principle. Two tracks exist: Track F (LightGBM main pipeline) and Track Q (LLM/Qwen approach, Q0 currently being trained).

## What model are we using?
- **Primary model (Track F):** LightGBM multiclass classifier (Gradient Boosted Decision Trees). 
- **Classes:** 0 = DOWN, 1 = FLAT, 2 = UP
- **Model file:** `artifacts/models/lightgbm_final.txt`
- **Code:** `src/stockml/models/lightgbm_train.py`
- **Baseline:** Logistic Regression for comparison at `artifacts/models/baseline_logistic.joblib` (`src/stockml/models/baseline.py`)
- **Calibration:** Temperature scaling saved in `artifacts/calibration.json`
- **Metadata:** `artifacts/model_metadata.json` (features, rows, accuracies, splits, etc.)

## What ML method did we use earlier?
1. Started with simple baselines (Logistic Regression) for comparison.
2. Moved to Gradient Boosting (LightGBM) as the main model.
3. Adopted **Triple-Barrier Labeling** (more realistic than just "next day up/down").
4. Implemented **Purged Walk-Forward Validation** (time-based, prevents leakage between folds).
5. Added **Optuna** for hyperparameter tuning.
6. Added **Temperature Scaling** for better probability calibration.
7. Extensive evaluation/ablation scripts over time (e.g., G1 tests, U/D tests) to iterate.

## What dataset are we using?
- **Universe:** NIFTY 50 stocks (50 Indian large-cap stocks). Config shows static list as of Sept 2026 in `config/config.yaml` (documented survivorship bias noted).
- **History:** ~16 years of daily OHLCV data (`history_years: 16`).
- **Price data:** `data/raw/{SYMBOL}.NS.parquet` from yfinance (default provider; noted as runnable fallback).
- **News/announcements:** `data/raw/news/`, `data/raw/announcements/`, AlphaVantage + GDELT (last ~90 days), RSS.
- **Other sources:** Bhavcopy (`data/raw/bhavcopy/` from 2020+), macro (`data/raw/macro.parquet`), market context (NIFTY/VIX), calendar.
- **Processed/labeled:** `data/processed/{SYMBOL}_dataset.parquet` with features + `target` (0/1/2) and event metadata.
- **Labels (Triple-Barrier):** Enter at t+1 open, set upper/lower barriers based on recent volatility (20-day rolling std). Look ahead up to horizon_days (currently 5). If upper barrier hit first → UP (2). If lower hit first → DOWN (0). If neither hit within horizon, classify by final return vs flat threshold (vol*flat_band) → FLAT (1) if small move else UP/DOWN. Implemented in `src/stockml/targets/triple_barrier.py`.

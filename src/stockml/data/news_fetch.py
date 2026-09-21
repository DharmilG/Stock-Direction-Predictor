from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from ..config import load_settings
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .news import collect_news_for_symbol
from .finbert import FinBERTScorer
from ..features.news_features import add_finbert_scores


DEFAULT_NEWS_QUERIES = {
    "RELIANCE.NS": "Reliance Industries",
    "BHARTIARTL.NS": "Bharti Airtel",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "TCS.NS": "Tata Consultancy Services",
    "BAJFINANCE.NS": "Bajaj Finance",
    "LT.NS": "Larsen Toubro",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "TITAN.NS": "Titan Company",
    "SUNPHARMA.NS": "Sun Pharma",
    "INFY.NS": "Infosys",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "MARUTI.NS": "Maruti Suzuki",
    "AXISBANK.NS": "Axis Bank",
    "M&M.NS": "Mahindra Mahindra",
    "HCLTECH.NS": "HCL Technologies",
    "ITC.NS": "ITC Limited",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "NTPC.NS": "NTPC Limited",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "ETERNAL.NS": "Eternal Zomato",
    "JSWSTEEL.NS": "JSW Steel",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BEL.NS": "Bharat Electronics",
    "ONGC.NS": "ONGC Oil Natural Gas",
    "COALINDIA.NS": "Coal India",
    "POWERGRID.NS": "Power Grid Corporation India",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "ASIANPAINT.NS": "Asian Paints",
    "TATASTEEL.NS": "Tata Steel",
    "GRASIM.NS": "Grasim Industries",
    "HINDALCO.NS": "Hindalco Industries",
    "EICHERMOT.NS": "Eicher Motors",
    "INDIGO.NS": "Interglobe Aviation IndiGo",
    "SBILIFE.NS": "SBI Life Insurance",
    "WIPRO.NS": "Wipro",
    "JIOFIN.NS": "Jio Financial Services",
    "TECHM.NS": "Tech Mahindra",
    "TRENT.NS": "Trent Limited",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "TMPV.NS": "Tata Motors",
    "CIPLA.NS": "Cipla",
    "MAXHEALTH.NS": "Max Healthcare",
    "TATACONSUM.NS": "Tata Consumer Products",
    "DRREDDY.NS": "Dr Reddy Laboratories",
    "NESTLEIND.NS": "Nestle India",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "^NSEI": "Nifty 50 OR Indian stock market",
}


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")


def _query_for(symbol: str) -> str:
    return DEFAULT_NEWS_QUERIES.get(symbol, symbol.replace(".NS", "").replace(".BO", "").replace("^", ""))


def run_news_fetch() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "news.log")
    if not settings.enable_news:
        logger.info("News ingestion disabled; skipping.")
        update_state(paths.state, "news", status="skipped")
        return

    end = date.today()
    start = end - timedelta(days=round(settings.history_years * 365.25))
    out_dir = paths.data_raw / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    provider_names = {x.strip().lower() for x in settings.news_provider.split(",") if x.strip()}
    for symbol in settings.symbols:
        output = out_dir / f"{_safe_symbol(symbol)}.parquet"
        query = _query_for(symbol)
        # Alpha Vantage symbol can be overridden with NEWS_TICKER_<SAFE_SYMBOL>.
        env_key = "NEWS_TICKER_" + _safe_symbol(symbol).upper()
        import os
        default_av_symbol = None if symbol.startswith("^") else symbol.replace(".NS", ".BSE")
        av_symbol = os.getenv(env_key, default_av_symbol or "")
        frame = collect_news_for_symbol(
            symbol=symbol,
            start=start,
            end=end,
            output_path=output,
            query=query,
            alphavantage_symbol=av_symbol,
            enable_alphavantage="alphavantage" in provider_names and bool(settings.alphavantage_api_key) and bool(av_symbol),
            enable_gdelt_recent="gdelt" in provider_names,
            gdelt_days=settings.news_gdelt_days,
        )
        if settings.enable_sentiment and not frame.empty:
            if "finbert_score" not in frame.columns:
                frame["finbert_score"] = float("nan")
            missing = frame["finbert_score"].isna()
            if bool(missing.any()):
                scorer = FinBERTScorer(settings.finbert_model, settings.finbert_cache_dir, settings.finbert_device)
                subset = frame.loc[missing, ["title", "description"]].fillna("")
                texts = (subset["title"].astype(str) + ". " + subset["description"].astype(str)).tolist()
                labels, scores = scorer(texts)
                frame.loc[missing, "finbert_label"] = labels
                frame.loc[missing, "finbert_score"] = scores
                frame.loc[missing, "sentiment"] = scores
                tmp = output.with_suffix(output.suffix + ".tmp")
                frame.to_parquet(tmp, index=False)
                tmp.replace(output)
        logger.info("News corpus for %s: %d unique articles", symbol, len(frame))
        completed.append(symbol)
    update_state(paths.state, "news", status="complete", symbols=completed, start=str(start), end=str(end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_news_fetch()

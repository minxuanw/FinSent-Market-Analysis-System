"""
Main pipeline runner for FinSent-Market Analysis System.
Orchestrates data collection, sentiment analysis, and analysis.
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from src.database.db_manager import DatabaseManager
from src.data_collection.news_collector import NewsCollector
from src.data_collection.market_collector import MarketCollector
from src.preprocessing.preprocessor import TextPreprocessor
from src.sentiment.vader_analyzer import VaderAnalyzer
from src.sentiment.finbert_analyzer import FinBERTAnalyzer
from src.analysis.market_correlation import MarketSentimentAnalyzer
from comprehensive_evaluation import run_full_evaluation


def _get_col(results_df, *candidates):
    """Return the first existing column name from candidates."""
    for c in candidates:
        if c in results_df.columns:
            return c
    return candidates[-1]  # fallback


def step_1_collect_data(db: DatabaseManager, args):
    """Step 1: Collect news and market data."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)

    # News collection
    if not args.skip_news:
        news_collector = NewsCollector()
        news_df = news_collector.collect_all()

        if not news_df.empty:
            preprocessor = TextPreprocessor()
            news_df = preprocessor.preprocess_dataframe(news_df)
            db.insert_news(news_df)
            print(f"\n  News stored: {db.get_table_count('news_articles')} articles")
        else:
            print("\n  No news data collected")
    else:
        print("\n  Skipping news collection")

    # Market data collection
    if not args.skip_market:
        market_collector = MarketCollector()
        market_df = market_collector.collect_all()

        if not market_df.empty:
            db.insert_market_data(market_df)
            print(f"\n  Market data stored: {db.get_table_count('market_data')} records")
        else:
            print("\n  No market data collected")
    else:
        print("\n  Skipping market data collection")


def step_2_sentiment_analysis(db: DatabaseManager, args):
    """Step 2: Apply sentiment analysis models."""
    print("\n" + "=" * 60)
    print("STEP 2: SENTIMENT ANALYSIS")
    print("=" * 60)

    # Get all news articles
    news_df = db.get_news()

    if news_df.empty:
        print("\n  No news articles found. Run data collection first.")
        return

    print(f"\n  Analyzing {len(news_df)} articles...")

    # VADER
    if not args.skip_vader:
        vader = VaderAnalyzer()
        vader_results = vader.analyze_batch(news_df)

        # Detect column names (with or without prefix)
        label_col = _get_col(vader_results, "vader_label", "label")
        compound_col = _get_col(vader_results, "vader_compound_score", "compound_score")
        pos_col = _get_col(vader_results, "vader_score_positive", "score_positive")
        neg_col = _get_col(vader_results, "vader_score_negative", "score_negative")
        neu_col = _get_col(vader_results, "vader_score_neutral", "score_neutral")

        # Store per-article sentiment
        id_col = "id" if "id" in vader_results.columns else None
        cols_to_store = [id_col, label_col, compound_col, pos_col, neg_col, neu_col]
        cols_to_store = [c for c in cols_to_store if c is not None]

        sentiment_df = vader_results[cols_to_store].copy()
        sentiment_df.columns = ["article_id", "label", "compound_score",
                                "score_positive", "score_negative", "score_neutral"][:len(cols_to_store)]
        sentiment_df["model"] = vader.model_name
        db.insert_sentiment(sentiment_df)

        # Aggregate daily
        news_for_agg = news_df.copy()
        news_for_agg["date"] = pd.to_datetime(news_for_agg["published_at"], format="ISO8601", utc=True).dt.date.astype(str)
        merged = news_for_agg.merge(sentiment_df, left_index=True, right_index=True, how="inner")
        daily_vader = vader.aggregate_daily(merged)
        db.insert_daily_sentiment(daily_vader)

        print(f"\n  VADER sentiment stored")
    else:
        print("\n  Skipping VADER analysis")

    # FinBERT
    if not args.skip_finbert:
        finbert = FinBERTAnalyzer()
        finbert_results = finbert.analyze_batch(news_df)

        # Detect column names (with or without prefix)
        label_col = _get_col(finbert_results, "finbert_label", "label")
        compound_col = _get_col(finbert_results, "finbert_compound_score", "compound_score")
        pos_col = _get_col(finbert_results, "finbert_score_positive", "score_positive")
        neg_col = _get_col(finbert_results, "finbert_score_negative", "score_negative")
        neu_col = _get_col(finbert_results, "finbert_score_neutral", "score_neutral")

        # Store per-article sentiment
        id_col = "id" if "id" in finbert_results.columns else None
        cols_to_store = [id_col, label_col, compound_col, pos_col, neg_col, neu_col]
        cols_to_store = [c for c in cols_to_store if c is not None]

        sentiment_df = finbert_results[cols_to_store].copy()
        sentiment_df.columns = ["article_id", "label", "compound_score",
                                "score_positive", "score_negative", "score_neutral"][:len(cols_to_store)]
        sentiment_df["model"] = finbert.model_label
        db.insert_sentiment(sentiment_df)

        # Aggregate daily
        news_for_agg = news_df.copy()
        news_for_agg["date"] = pd.to_datetime(news_for_agg["published_at"], format="ISO8601", utc=True).dt.date.astype(str)
        merged = news_for_agg.merge(sentiment_df, left_index=True, right_index=True, how="inner")
        daily_finbert = finbert.aggregate_daily(merged)
        db.insert_daily_sentiment(daily_finbert)

        print(f"\n  FinBERT sentiment stored")
    else:
        print("\n  Skipping FinBERT analysis")


def step_3_analysis(db: DatabaseManager, args):
    """Step 3: Perform correlation analysis and generate reports."""
    print("\n" + "=" * 60)
    print("STEP 3: ANALYSIS & REPORTING")
    print("=" * 60)

    analyzer = MarketSentimentAnalyzer()
    tickers = db.get_stored_tickers()

    if not tickers:
        print("\n  No tickers found in database")
        return

    print(f"\n  Analyzing {len(tickers)} tickers...")

    for model in ["VADER", "FinBERT"]:
        print(f"\n{'=' * 40}")
        print(f"  Analysis: {model}")
        print(f"{'=' * 40}")

        for ticker in tqdm(tickers, desc=f"{model} tickers"):
            df = db.get_sentiment_with_market(ticker, model)

            if len(df) < 5:
                print(f"    Insufficient data for {ticker}: {len(df)} records")
                continue

            # Analyze and plot
            result = analyzer.analyze_ticker(df, ticker)
            if "error" not in result:
                print(f"    {ticker}: r={result['return_corr']['pearson_r']} vs return")

            if not args.no_plots:
                analyzer.plot_sentiment_vs_return(df, ticker, model)

        # Generate model comparison report
        print(f"\n  Generating {model} report...")
        daily_df = db.get_daily_sentiment(model=model)
        if not daily_df.empty:
            analyzer.generate_report(daily_df, model)
            if not args.no_plots:
                analyzer.plot_correlation_heatmap(daily_df)

    # Cross-model comparison
    print(f"\n{'=' * 40}")
    print("  Cross-Model Comparison")
    print(f"{'=' * 40}")

    vader_daily = db.get_daily_sentiment(model="VADER")
    finbert_daily = db.get_daily_sentiment(model="FinBERT")

    if not vader_daily.empty and not finbert_daily.empty:
        comparison = analyzer.cross_model_comparison(vader_daily, finbert_daily)
        if not comparison.empty:
            print(comparison.to_string(index=False))
            comparison.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    if not args.no_plots:
        # Get per-article sentiment for distribution plot
        vader_articles = db.get_sentiment(model="VADER")
        finbert_articles = db.get_sentiment(model="FinBERT")

        if not vader_articles.empty and not finbert_articles.empty:
            analyzer.plot_sentiment_distribution(vader_articles, finbert_articles)

    print(f"\n  Analysis complete. Results in: {PROCESSED_DIR / 'analysis'}")


def run_full_pipeline(args):
    """Execute all steps."""
    print("\n" + "=" * 60)
    print("  FinSent-Market Analysis Pipeline")
    print("=" * 60)

    # Initialize database
    db = DatabaseManager()

    # Run steps
    step_1_collect_data(db, args)
    step_2_sentiment_analysis(db, args)
    step_3_analysis(db, args)

    # Step 4: Comprehensive Evaluation
    if not args.skip_evaluation:
        run_full_evaluation()
    else:
        print("\n  Skipping comprehensive evaluation")

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Database: {DB_PATH}")
    print(f"  News articles: {db.get_table_count('news_articles')}")
    print(f"  Market records: {db.get_table_count('market_data')}")
    print(f"  Sentiment scores: {db.get_table_count('sentiment_scores')}")
    print(f"  Daily aggregates: {db.get_table_count('daily_sentiment')}")
    print(f"\n  All outputs saved to: {PROCESSED_DIR}")
    print("\n  Run 'streamlit run dashboard/app.py' to explore results!")


def main():
    parser = argparse.ArgumentParser(description="FinSent-Market Analysis Pipeline")
    parser.add_argument("--skip-news", action="store_true", help="Skip news collection")
    parser.add_argument("--skip-market", action="store_true", help="Skip market data collection")
    parser.add_argument("--skip-vader", action="store_true", help="Skip VADER analysis")
    parser.add_argument("--skip-finbert", action="store_true", help="Skip FinBERT analysis")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip comprehensive evaluation")
    parser.add_argument("--collect-only", action="store_true", help="Only run data collection")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis (skip collection)")
    parser.add_argument("--evaluate-only", action="store_true", help="Only run comprehensive evaluation")

    args = parser.parse_args()

    if args.collect_only:
        db = DatabaseManager()
        step_1_collect_data(db, args)
    elif args.analyze_only:
        db = DatabaseManager()
        step_2_sentiment_analysis(db, args)
        step_3_analysis(db, args)
    elif args.evaluate_only:
        run_full_evaluation()
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()

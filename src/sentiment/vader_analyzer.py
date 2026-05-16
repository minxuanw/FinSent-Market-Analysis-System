"""
VADER Sentiment Analyzer — Baseline model.
Lightweight lexicon-and-rule-based sentiment analysis.
"""
import pandas as pd
import nltk
from tqdm import tqdm
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
import config

# Download VADER lexicon on first use
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class VaderAnalyzer:
    """VADER-based sentiment analyzer for financial text."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.model_name = "VADER"

    def analyze_single(self, text: str) -> dict:
        """
        Analyze sentiment of a single text.

        Returns:
            dict with label, compound, pos, neg, neu scores
        """
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Map compound score to label
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "model": self.model_name,
            "label": label,
            "compound_score": compound,
            "score_positive": scores["pos"],
            "score_negative": scores["neg"],
            "score_neutral": scores["neu"],
        }

    def analyze_batch(self, df: pd.DataFrame, text_col: str = "headline_clean") -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.

        Args:
            df: DataFrame with text column
            text_col: Name of the column containing cleaned text

        Returns:
            Original DataFrame with sentiment columns appended
        """
        print(f"\n  VADER: Analyzing {len(df)} headlines...")

        results = []
        for text in tqdm(df[text_col], desc="VADER"):
            result = self.analyze_single(str(text))
            results.append(result)

        sentiment_df = pd.DataFrame(results)
        combined = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

        # Print summary
        label_counts = combined["label"].value_counts()
        print(f"    Results: ", end="")
        for label, count in label_counts.items():
            print(f"{label}={count} ({count/len(combined):.1f}%)  ", end="")
        print()

        return combined

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment to daily level by ticker.
        """
        daily = df.groupby(["ticker", "date"]).agg(
            avg_compound=("compound_score", "mean"),
            avg_positive=("score_positive", "mean"),
            avg_negative=("score_negative", "mean"),
            avg_neutral=("score_neutral", "mean"),
            article_count=("headline", "count"),
        ).reset_index()
        daily["model"] = self.model_name
        return daily


if __name__ == "__main__":
    analyzer = VaderAnalyzer()

    # Test with sample headlines
    test_headlines = [
        "Apple reports record quarterly earnings, beating analyst expectations",
        "Market crashes as inflation fears grow among investors",
        "Federal Reserve holds interest rates steady amid economic uncertainty",
        "Tesla stock surges after announcing new model lineup",
        "Bank of America warns of potential recession in 2026",
    ]

    print("\n  VADER Sentiment Analysis Test:\n")
    for headline in test_headlines:
        result = analyzer.analyze_single(headline)
        print(f"  [{result['label'].upper():>8}] ({result['compound_score']:+.4f})  {headline}")

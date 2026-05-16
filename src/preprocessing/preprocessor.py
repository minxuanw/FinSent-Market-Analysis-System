"""
Text preprocessing pipeline for financial news.
Handles cleaning, tokenization, and date alignment.
"""
import pandas as pd
import re
from datetime import datetime
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
import config


class TextPreprocessor:
    """Cleans and prepares financial news text for sentiment analysis."""

    def __init__(self):
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """Load basic English stopwords (no NLTK dependency for preprocessing)."""
        return {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
            "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
            "or", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "about", "against", "between", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then", "once",
        }

    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\- \s.,!?;:'\"$%\-]", "", text)

        return text

    def normalize_date(self, date_val) -> str:
        """Parse and normalize date to YYYY-MM-DD format."""
        if pd.isna(date_val):
            return None
        if isinstance(date_val, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]:
                try:
                    return datetime.strptime(date_val, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
        if isinstance(date_val, (pd.Timestamp, datetime)):
            return pd.Timestamp(date_val).strftime("%Y-%m-%d")
        return None

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline for news DataFrame.

        Steps:
        1. Remove duplicates
        2. Clean text fields
        3. Normalize dates
        4. Add date-only column for aggregation
        5. Filter invalid rows
        """
        print(f"\n  Preprocessing {len(df)} articles...")

        # 1. Remove duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["headline"])
        print(f"    Duplicates removed: {before - len(df)}")

        # 2. Clean headline and summary
        df["headline_clean"] = df["headline"].apply(self.clean_text)
        df["summary_clean"] = df["summary"].apply(self.clean_text)

        # 3. Normalize dates
        df["date"] = df["published_at"].apply(self.normalize_date)
        df = df.dropna(subset=["date"])

        # 4. Ensure ticker is uppercase
        df["ticker"] = df["ticker"].str.upper().str.strip()

        # 5. Filter empty headlines after cleaning
        df = df[df["headline_clean"].str.len() > 0]

        # 6. Reset index
        df = df.reset_index(drop=True)

        print(f"  ✓ Preprocessed: {len(df)} articles remaining")
        return df

    @staticmethod
    def align_with_market(news_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align news dates with trading days.
        Forward-fills market data for non-trading days.
        """
        # Ensure date columns are strings
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")
        market_df["date"] = pd.to_datetime(market_df["date"]).dt.strftime("%Y-%m-%d")

        # Merge on ticker + date
        merged = news_df.merge(
            market_df,
            on=["ticker", "date"],
            how="left"
        )

        # Forward fill missing market data (weekends/holidays)
        market_cols = ["close", "daily_return", "volatility_5d", "volume"]
        for ticker in merged["ticker"].unique():
            mask = merged["ticker"] == ticker
            merged.loc[mask, market_cols] = (
                merged.loc[mask, market_cols]
                .sort_values("date")
                .ffill()
            )

        return merged


if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    # Load raw data if available
    news_path = config.RAW_DIR / "news_raw.csv"
    if news_path.exists():
        df = pd.read_csv(news_path)
        clean_df = preprocessor.preprocess_dataframe(df)
        clean_df.to_csv(config.PROCESSED_DIR / "news_clean.csv", index=False)
        print(f"  Saved to {config.PROCESSED_DIR / 'news_clean.csv'}")
    else:
        print("No raw news data found. Run news_collector.py first.")

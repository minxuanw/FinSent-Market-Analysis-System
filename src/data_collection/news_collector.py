"""
News data collector — fetches financial news from multiple sources.
Supports Finnhub API, NewsAPI, and CSV import from Kaggle datasets.
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional
from tqdm import tqdm
import time
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
import config


class NewsCollector:
    """Collects financial news headlines and summaries."""

    def __init__(self):
        self.finnhub_key = config.FINNHUB_API_KEY
        self.newsapi_key = config.NEWSAPI_KEY

    # --- Finnhub ---
    def fetch_finnhub(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch company news from Finnhub API.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: ticker, headline, summary, source, url, published_at
        """
        if not self.finnhub_key:
            print("  FINNHUB_API_KEY not set. Skipping Finnhub collection.")
            return pd.DataFrame()

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start_date,
            "to": end_date,
            "token": self.finnhub_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Finnhub API error for {ticker}: {e}")
            return pd.DataFrame()

        articles = []
        for item in data:
            dt = datetime.fromtimestamp(item.get("datetime", 0))
            articles.append({
                "ticker": ticker,
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "published_at": dt.strftime("%Y-%m-%d %H:%M:%S"),
            })

        df = pd.DataFrame(articles)
        if not df.empty:
            print(f"  Finnhub: {len(df)} articles for {ticker}")
        return df

    # --- NewsAPI ---
    def fetch_newsapi(self, query: str, start_date: str, end_date: str,
                      page_size: int = 100) -> pd.DataFrame:
        """
        Fetch news from NewsAPI.org.
        Free tier: 100 results per request, up to 1 month range.
        """
        if not self.newsapi_key:
            print("  NEWSAPI_KEY not set. Skipping NewsAPI collection.")
            return pd.DataFrame()

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": start_date,
            "to": end_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(page_size, 100),
            "apiKey": self.newsapi_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  NewsAPI error: {e}")
            return pd.DataFrame()

        articles = []
        for item in data.get("articles", []):
            articles.append({
                "ticker": query,
                "headline": item.get("title", ""),
                "summary": item.get("description", "") or "",
                "source": item.get("source", {}).get("name", ""),
                "url": item.get("url", ""),
                "published_at": item.get("publishedAt", ""),
            })

        df = pd.DataFrame(articles)
        if not df.empty:
            print(f"  NewsAPI: {len(df)} articles for '{query}'")
        return df

    # --- CSV Import (Kaggle datasets) ---
    def load_csv(self, filepath: str,
                 headline_col: str = "headline",
                 date_col: str = "date",
                 ticker_col: str = "stock") -> pd.DataFrame:
        """
        Load news from a CSV file (e.g., Kaggle Financial News Dataset).

        Expected columns (customizable):
            - headline_col: News headline text
            - date_col: Publication date
            - ticker_col: Stock ticker symbol
        """
        df = pd.read_csv(filepath)

        # Standardize column names
        col_map = {}
        if headline_col in df.columns:
            col_map[headline_col] = "headline"
        if date_col in df.columns:
            col_map[date_col] = "published_at"
        if ticker_col in df.columns:
            col_map[ticker_col] = "ticker"

        df = df.rename(columns=col_map)

        # Ensure required columns
        for col in ["headline", "published_at", "ticker"]:
            if col not in df.columns:
                if col == "headline":
                    # Try alternative names
                    for alt in ["title", "Title", "news", "text"]:
                        if alt in df.columns:
                            df = df.rename(columns={alt: "headline"})
                            break

        # Add missing columns
        if "summary" not in df.columns:
            df["summary"] = ""
        if "source" not in df.columns:
            df["source"] = "csv_import"
        if "url" not in df.columns:
            df["url"] = ""

        # Parse dates
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        # Drop rows without headlines
        df = df.dropna(subset=["headline"])

        print(f"  CSV import: {len(df)} articles from {filepath}")
        return df[["ticker", "headline", "summary", "source", "url", "published_at"]]

    # --- Batch Collection ---
    def collect_all(self, tickers: list = None,
                    start_date: str = None,
                    end_date: str = None) -> pd.DataFrame:
        """
        Collect news for all tickers across available sources.
        """
        tickers = tickers or config.DEFAULT_TICKERS
        start_date = start_date or config.DATA_START_DATE
        end_date = end_date or config.DATA_END_DATE

        all_articles = []

        print(f"\n  Collecting news for {len(tickers)} tickers...")
        print(f"    Period: {start_date} → {end_date}\n")

        for ticker in tqdm(tickers, desc="Tickers"):
            # Finnhub
            df_fh = self.fetch_finnhub(ticker, start_date, end_date)
            if not df_fh.empty:
                all_articles.append(df_fh)

            # NewsAPI supplement
            df_na = self.fetch_newsapi(f"{ticker} stock", start_date, end_date)
            if not df_na.empty:
                df_na["ticker"] = ticker  # override query with actual ticker
                all_articles.append(df_na)

            # Rate limiting
            time.sleep(1)

        if all_articles:
            result = pd.concat(all_articles, ignore_index=True)
            result = result.drop_duplicates(subset=["headline", "published_at"])
            print(f"\n  Total unique articles collected: {len(result)}")
            return result

        print("\n  No articles collected. Check API keys or use CSV import.")
        return pd.DataFrame()


if __name__ == "__main__":
    collector = NewsCollector()
    df = collector.collect_all()
    if not df.empty:
        df.to_csv(config.RAW_DIR / "news_raw.csv", index=False)
        print(f"Saved to {config.RAW_DIR / 'news_raw.csv'}")

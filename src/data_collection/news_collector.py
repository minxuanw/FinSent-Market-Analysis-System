"""
News data collector — fetches financial news from multiple sources.
Supports Finnhub API, NewsAPI, and CSV import from Kaggle datasets.

Rate limits (Free tiers):
  - NewsAPI Developer: 100 requests/day, max 20 results/page, ≤1 month old
  - Finnhub Free: 60 calls/min, 30 calls/sec across all free accounts
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

# ── Rate-limit constants ───────────────────────────────────────────
FINNHUB_CALLS_PER_MIN = 60          # free tier
FINNHUB_MIN_INTERVAL = 1.0          # 60/min → ≥1s between calls
FINNHUB_BURST_INTERVAL = 0.035      # 30/sec → ≥35ms, but we use 1s to stay safe

NEWSAPI_MAX_PAGE_SIZE = 20          # free tier: max 20 results per page
NEWSAPI_MAX_PAGES = 5               # 5 pages × 20 = 100 results per query (1 request/page)
NEWSAPI_DAILY_LIMIT = 100           # 100 requests/day
NEWSAPI_MAX_AGE_DAYS = 29           # free tier: only up to 1 month old


def _clamp_to_newsapi_range(start_date: str, end_date: str) -> tuple:
    """Clamp the requested date range to NewsAPI's 1-month free-tier window."""
    earliest = (datetime.utcnow() - timedelta(days=NEWSAPI_MAX_AGE_DAYS)).strftime("%Y-%m-%d")
    clamped_start = max(start_date, earliest)
    return clamped_start, end_date


class NewsCollector:
    """Collects financial news headlines and summaries."""

    def __init__(self):
        self.finnhub_key = config.FINNHUB_API_KEY
        self.newsapi_key = config.NEWSAPI_KEY
        self._newsapi_requests_today = 0

    # ── Finnhub ────────────────────────────────────────────────────────
    def fetch_finnhub(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch company news from Finnhub API.
        Free tier: 60 calls/min. We throttle to ≥1s between calls.

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
            time.sleep(FINNHUB_MIN_INTERVAL)  # rate limit: 60/min
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

    # ── NewsAPI ────────────────────────────────────────────────────────
    def fetch_newsapi(self, query: str, start_date: str, end_date: str,
                      page_size: int = NEWSAPI_MAX_PAGE_SIZE) -> pd.DataFrame:
        """
        Fetch news from NewsAPI.org (Developer / Free tier).
        Limits: 100 requests/day, max 20 results/page, articles ≤1 month old.
        """
        if not self.newsapi_key:
            print("  NEWSAPI_KEY not set. Skipping NewsAPI collection.")
            return pd.DataFrame()

        # Clamp date range to NewsAPI free-tier window
        start_clamped, end_clamped = _clamp_to_newsapi_range(start_date, end_date)
        if start_clamped > end_clamped:
            print(f"  NewsAPI: requested range {start_date}~{end_date} is older than 1-month window. Skipped.")
            return pd.DataFrame()

        url = "https://newsapi.org/v2/everything"
        page_size = min(page_size, NEWSAPI_MAX_PAGE_SIZE)
        all_articles = []

        for page in range(1, NEWSAPI_MAX_PAGES + 1):
            if self._newsapi_requests_today >= NEWSAPI_DAILY_LIMIT:
                print(f"  NewsAPI: daily limit ({NEWSAPI_DAILY_LIMIT} requests) reached. Stopping.")
                break

            params = {
                "q": query,
                "from": start_clamped,
                "to": end_clamped,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "page": page,
                "apiKey": self.newsapi_key,
            }

            try:
                resp = requests.get(url, params=params, timeout=30)
                self._newsapi_requests_today += 1
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  NewsAPI error (page {page}): {e}")
                break

            if data.get("status") != "ok":
                print(f"  NewsAPI returned status: {data.get('status', 'unknown')}, code: {data.get('code', '')}")
                break

            page_articles = data.get("articles", [])
            if not page_articles:
                break  # no more results

            for item in page_articles:
                all_articles.append({
                    "ticker": query,  # caller should override with actual ticker
                    "headline": item.get("title", ""),
                    "summary": item.get("description", "") or "",
                    "source": item.get("source", {}).get("name", ""),
                    "url": item.get("url", ""),
                    "published_at": item.get("publishedAt", ""),
                })

            # If we got fewer than page_size, this is the last page
            if len(page_articles) < page_size:
                break

            # Small delay between pages to be polite
            time.sleep(0.5)

        df = pd.DataFrame(all_articles)
        if not df.empty:
            print(f"  NewsAPI: {len(df)} articles for '{query}' "
                  f"({self._newsapi_requests_today}/{NEWSAPI_DAILY_LIMIT} daily requests used)")
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

        Respects free-tier limits:
          - Finnhub: 60 calls/min (1s min interval between calls)
          - NewsAPI: 100 requests/day, 20 results/page, ≤1 month old
        """
        tickers = tickers or config.DEFAULT_TICKERS
        start_date = start_date or config.DATA_START_DATE
        end_date = end_date or config.DATA_END_DATE

        all_articles = []

        print(f"\n  Collecting news for {len(tickers)} tickers...")
        print(f"    Period: {start_date} → {end_date}")

        # Estimate if NewsAPI will be useful (only if range overlaps with last month)
        earliest_newsapi = (datetime.utcnow() - timedelta(days=NEWSAPI_MAX_AGE_DAYS)).strftime("%Y-%m-%d")
        newsapi_useful = end_date >= earliest_newsapi
        if not newsapi_useful:
            print(f"    ⚠ NewsAPI: end_date is older than 1-month window (earliest: {earliest_newsapi}). "
                  f"Only Finnhub will be used.")

        # Rough budget estimate for NewsAPI
        if newsapi_useful:
            # Each ticker may use up to NEWSAPI_MAX_PAGES (5) requests
            max_newsapi_per_ticker = NEWSAPI_MAX_PAGES
            estimated_total = len(tickers) * max_newsapi_per_ticker
            if estimated_total > NEWSAPI_DAILY_LIMIT:
                affordable = NEWSAPI_DAILY_LIMIT // max_newsapi_per_ticker
                print(f"    ⚠ NewsAPI: {len(tickers)} tickers × {max_newsapi_per_ticker} pages = "
                      f"{estimated_total} requests > {NEWSAPI_DAILY_LIMIT}/day limit.")
                print(f"    → Will use NewsAPI for first {affordable} tickers, then Finnhub only.")

        print()

        for ticker in tqdm(tickers, desc="Tickers"):
            # Finnhub
            df_fh = self.fetch_finnhub(ticker, start_date, end_date)
            if not df_fh.empty:
                all_articles.append(df_fh)

            # NewsAPI supplement (skip if daily budget exhausted)
            if newsapi_useful and self._newsapi_requests_today < NEWSAPI_DAILY_LIMIT:
                df_na = self.fetch_newsapi(f"{ticker} stock", start_date, end_date)
                if not df_na.empty:
                    df_na["ticker"] = ticker  # override query with actual ticker
                    all_articles.append(df_na)

            # Finnhub rate limit: 60/min → 1s sleep already in fetch_finnhub
            # Extra buffer between tickers
            time.sleep(0.5)

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

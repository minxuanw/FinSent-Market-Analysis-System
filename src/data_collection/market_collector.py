"""
Market data collector — fetches historical stock prices via yfinance.
Computes daily returns and rolling volatility.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
import config


class MarketCollector:
    """Collects and processes historical market data."""

    def __init__(self):
        self.tickers = config.DEFAULT_TICKERS
        self.index_symbol = config.MARKET_INDEX

    def fetch_single(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker."""
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                print(f"    No data for {ticker}")
                return pd.DataFrame()

            # Handle multi-level columns from yfinance (newer versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df = df.reset_index()
            df["ticker"] = ticker

            # Standardize column names first (yfinance uses Title case)
            col_map = {}
            for c in df.columns:
                col_map[c] = {"Date": "date", "Open": "open", "High": "high",
                              "Low": "low", "Close": "close", "Volume": "volume"}.get(c, c)
            df = df.rename(columns=col_map)

            # Compute daily return
            df["daily_return"] = df["close"].pct_change()

            # Compute rolling volatility (5-day and 20-day)
            df["volatility_5d"] = df["daily_return"].rolling(window=5).std()
            df["volatility_20d"] = df["daily_return"].rolling(window=20).std()

            # Keep relevant columns
            cols = ["ticker", "date", "open", "high", "low", "close", "volume",
                    "daily_return", "volatility_5d", "volatility_20d"]
            df = df[[c for c in cols if c in df.columns]]

            print(f"  {ticker}: {len(df)} trading days")
            return df

        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def fetch_index(self, start: str, end: str) -> pd.DataFrame:
        """Fetch market index data (S&P 500 by default)."""
        return self.fetch_single(self.index_symbol, start, end)

    def collect_all(self, tickers: list = None,
                    start_date: str = None,
                    end_date: str = None) -> pd.DataFrame:
        """
        Fetch market data for all tickers.

        Returns:
            DataFrame with OHLCV + derived features for all tickers
        """
        tickers = tickers or self.tickers
        start_date = start_date or config.DATA_START_DATE
        end_date = end_date or config.DATA_END_DATE

        all_data = []

        # Include index
        all_tickers = tickers + [self.index_symbol]

        print(f"\n  Collecting market data for {len(all_tickers)} symbols...")
        print(f"    Period: {start_date} → {end_date}\n")

        for ticker in tqdm(all_tickers, desc="Symbols"):
            df = self.fetch_single(ticker, start_date, end_date)
            if not df.empty:
                all_data.append(df)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            print(f"\n  Total records: {len(result)}")
            return result

        return pd.DataFrame()

    def get_current_price(self, ticker: str) -> float:
        """Get the latest price for a ticker."""
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                return hist["Close"].iloc[-1]
        except Exception:
            pass
        return 0.0


if __name__ == "__main__":
    collector = MarketCollector()
    df = collector.collect_all()
    if not df.empty:
        df.to_csv(config.RAW_DIR / "market_raw.csv", index=False)
        print(f"\n  Saved to {config.RAW_DIR / 'market_raw.csv'}")

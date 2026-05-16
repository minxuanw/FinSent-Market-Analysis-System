"""
Database manager for FinSent-Market system.
Handles SQLite operations for news articles, market data, and sentiment scores.
"""
import pandas as pd
from sqlalchemy import create_engine, text
from contextlib import contextmanager
import config


class DatabaseManager:
    """Manages all database operations for the FinSent pipeline."""

    def __init__(self, db_url=None):
        self.db_url = db_url or config.DB_URL
        self.engine = create_engine(self.db_url)
        self._init_tables()

    def _init_tables(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    headline TEXT NOT NULL,
                    summary TEXT,
                    source TEXT,
                    url TEXT,
                    published_at TIMESTAMP,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    daily_return REAL,
                    volatility_5d REAL,
                    volatility_20d REAL,
                    UNIQUE(ticker, date)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER,
                    model TEXT,
                    label TEXT,
                    score_positive REAL,
                    score_negative REAL,
                    score_neutral REAL,
                    compound_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(article_id, model),
                    FOREIGN KEY (article_id) REFERENCES news_articles(id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date DATE,
                    model TEXT,
                    avg_compound REAL,
                    avg_positive REAL,
                    avg_negative REAL,
                    avg_neutral REAL,
                    article_count INTEGER,
                    UNIQUE(ticker, date, model)
                )
            """))
            conn.commit()

    @contextmanager
    def _connect(self):
        with self.engine.connect() as conn:
            yield conn
            conn.commit()

    def insert_news(self, articles_df: pd.DataFrame):
        """Insert news articles, skipping duplicates by headline."""
        # Only keep columns that exist in the table schema
        table_cols = ["ticker", "headline", "summary", "source", "url", "published_at"]
        insert_df = articles_df[[c for c in table_cols if c in articles_df.columns]].copy()

        existing = pd.read_sql(text("SELECT headline FROM news_articles"), self.engine)
        existing_headlines = set(existing["headline"].tolist()) if not existing.empty else set()
        new_articles = insert_df[~insert_df["headline"].isin(existing_headlines)]
        if not new_articles.empty:
            new_articles.to_sql("news_articles", self.engine, if_exists="append", index=False)
            print(f"    Inserted {len(new_articles)} new articles (skipped {len(insert_df) - len(new_articles)} duplicates)")
        else:
            print(f"    No new articles to insert (all {len(insert_df)} already exist)")

    def insert_market_data(self, market_df: pd.DataFrame):
        """Insert market data, using UNIQUE constraint to upsert."""
        # Convert dates to string for SQLite
        if "date" in market_df.columns:
            market_df["date"] = market_df["date"].astype(str)
        # Drop existing duplicates first
        with self._connect() as conn:
            for _, row in market_df[["ticker", "date"]].drop_duplicates().iterrows():
                conn.execute(text(
                    "DELETE FROM market_data WHERE ticker = :ticker AND date = :date"
                ), {"ticker": row["ticker"], "date": row["date"]})
        market_df.to_sql("market_data", self.engine, if_exists="append", index=False)
        print(f"    Inserted/updated {len(market_df)} market data records")

    def insert_sentiment(self, sentiment_df: pd.DataFrame):
        """Insert sentiment scores, skipping duplicates."""
        sentiment_df.to_sql("sentiment_scores", self.engine, if_exists="append", index=False)

    def insert_daily_sentiment(self, daily_df: pd.DataFrame):
        """Insert daily aggregated sentiment, using UNIQUE constraint to upsert."""
        with self._connect() as conn:
            for _, row in daily_df.iterrows():
                conn.execute(text("""
                    INSERT OR REPLACE INTO daily_sentiment
                    (ticker, date, model, avg_compound, avg_positive, avg_negative, avg_neutral, article_count)
                    VALUES (:ticker, :date, :model, :avg_compound, :avg_positive, :avg_negative, :avg_neutral, :article_count)
                """), row.to_dict())

    def get_sentiment(self, model=None) -> pd.DataFrame:
        """Retrieve per-article sentiment scores."""
        query = "SELECT * FROM sentiment_scores WHERE 1=1"
        params = {}
        if model:
            query += " AND model = :model"
            params["model"] = model
        query += " ORDER BY created_at DESC"
        return pd.read_sql(text(query), self.engine, params=params)

    def get_news(self, ticker=None, start_date=None, end_date=None) -> pd.DataFrame:
        """Retrieve news articles with optional filters."""
        query = "SELECT * FROM news_articles WHERE 1=1"
        params = {}
        if ticker:
            query += " AND ticker = :ticker"
            params["ticker"] = ticker
        if start_date:
            query += " AND published_at >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND published_at <= :end_date"
            params["end_date"] = end_date
        query += " ORDER BY published_at DESC"
        return pd.read_sql(text(query), self.engine, params=params)

    def get_market_data(self, ticker=None, start_date=None, end_date=None) -> pd.DataFrame:
        """Retrieve market data."""
        query = "SELECT * FROM market_data WHERE 1=1"
        params = {}
        if ticker:
            query += " AND ticker = :ticker"
            params["ticker"] = ticker
        if start_date:
            query += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND date <= :end_date"
            params["end_date"] = end_date
        query += " ORDER BY date ASC"
        df = pd.read_sql(text(query), self.engine, params=params)
        if not df.empty and "date" in df.columns:
            df["date"] = df["date"].astype(str)
        return df

    def get_daily_sentiment(self, ticker=None, model=None) -> pd.DataFrame:
        """Retrieve daily aggregated sentiment."""
        query = "SELECT * FROM daily_sentiment WHERE 1=1"
        params = {}
        if ticker:
            query += " AND ticker = :ticker"
            params["ticker"] = ticker
        if model:
            query += " AND model = :model"
            params["model"] = model
        query += " ORDER BY date ASC"
        return pd.read_sql(text(query), self.engine, params=params)

    def get_sentiment_with_market(self, ticker: str, model: str) -> pd.DataFrame:
        """Join daily sentiment with market data for analysis."""
        query = text("""
            SELECT 
                ds.date,
                ds.ticker,
                ds.model,
                ds.avg_compound,
                ds.avg_positive,
                ds.avg_negative,
                ds.avg_neutral,
                ds.article_count,
                md.close,
                md.daily_return,
                md.volatility_5d,
                md.volume
            FROM daily_sentiment ds
            LEFT JOIN market_data md 
                ON ds.date = md.date AND ds.ticker = md.ticker
            WHERE ds.ticker = :ticker AND ds.model = :model
            ORDER BY ds.date ASC
        """)
        return pd.read_sql(query, self.engine, params={"ticker": ticker, "model": model})

    def get_table_count(self, table: str) -> int:
        """Get row count for a table."""
        with self._connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            return result.scalar()

    def get_stored_tickers(self) -> list:
        """Get list of tickers with data in the database."""
        with self._connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM news_articles"))
            return [row[0] for row in result.fetchall()]


if __name__ == "__main__":
    db = DatabaseManager()
    print(f"Database initialized at: {config.DB_PATH}")
    for table in ["news_articles", "market_data", "sentiment_scores", "daily_sentiment"]:
        print(f"  {table}: {db.get_table_count(table)} rows")

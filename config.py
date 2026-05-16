"""
# FinSent-Market Analysis System - Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent / ".env")

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "finsent.db"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- API Keys (set via .env or environment variables) ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# --- Market Data ---
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "BAC", "XOM"]
MARKET_INDEX = "^GSPC"  # S&P 500
DATA_START_DATE = "2026-01-01"
DATA_END_DATE = "2026-05-02"

# --- Sentiment Models ---
FINBERT_MODEL = "ProsusAI/finbert"
VADER_LEXICON = "vader_lexicon"

# --- Database ---
DB_URL = f"sqlite:///{DB_PATH}"

# --- Dashboard ---
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "FinSent-Market Analysis Dashboard"

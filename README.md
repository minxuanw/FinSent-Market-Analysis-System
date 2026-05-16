# FinSent-Market Analysis System

> A Financial News Sentiment Analytics Pipeline for Market Insight

**Author:** Minxuan Wang (25007472)  
**Supervisor:** Anu Mathrani

## Overview

FinSent-Market is an end-to-end system that collects financial news, applies sentiment analysis using both a lightweight baseline (VADER) and a domain-specific transformer model (FinBERT), and explores how sentiment signals relate to stock market movements through an interactive dashboard.

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌────────────┐
│  News Data   │──▶│  Preprocess  │──▶│  Sentiment   │──▶│  Analysis  │
│  Market Data │   │  & Clean     │   │  VADER/FinBERT│  │  Correlation│
└─────────────┘   └──────────────┘   └─────────────┘   └────────────┘
                                                                │
                        ┌──────────────┐                        ▼
                        │   SQLite DB  │◀──────────────┌────────────┐
                        └──────────────┘               │  Dashboard  │
                                                      │  (Streamlit)│
                                                      └────────────┘
```

## Project Structure

```
finsent-market/
├── README.md
├── requirements.txt
├── config.py                 # Configuration constants
├── data/
│   ├── raw/                  # Raw downloaded data
│   └── processed/            # Cleaned & merged data
├── src/
│   ├── data_collection/      # News & market data collectors
│   ├── preprocessing/        # Text cleaning & alignment
│   ├── sentiment/            # VADER & FinBERT analyzers
│   ├── analysis/             # Correlation & trend analysis
│   └── database/             # SQLite database manager
└── dashboard/                # Streamlit dashboard
```

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python run_pipeline.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

## Data Sources

- **Financial News:** Finnhub API (free tier) + News API (free tier)
- **Market Data:** Yahoo Finance via `yfinance`
- **Sentiment Labels:** FinBERT comes pre-trained on FinancialPhraseBank

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| News API | Finnhub, NewsAPI |
| Market Data | yfinance |
| Database | SQLite |
| NLP Baseline | NLTK VADER |
| NLP Advanced | FinBERT (HuggingFace) |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib |
| Analysis | pandas, scipy, numpy |


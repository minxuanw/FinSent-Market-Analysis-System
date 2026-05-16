"""
Market-Sentiment Correlation Analysis.
Examines relationships between sentiment signals and market indicators.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
import config


class MarketSentimentAnalyzer:
    """Analyzes correlations between sentiment and market data."""

    def __init__(self):
        self.output_dir = config.PROCESSED_DIR / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_correlation(self, df: pd.DataFrame,
                            sentiment_col: str = "avg_compound",
                            market_col: str = "daily_return") -> dict:
        """
        Compute Pearson and Spearman correlation between sentiment and market data.

        Returns:
            dict with correlation coefficients and p-values
        """
        valid = df[[sentiment_col, market_col]].dropna()

        if len(valid) < 3:
            return {"pearson_r": np.nan, "pearson_p": np.nan,
                    "spearman_r": np.nan, "spearman_p": np.nan, "n": len(valid)}

        pearson_r, pearson_p = stats.pearsonr(valid[sentiment_col], valid[market_col])
        spearman_r, spearman_p = stats.spearmanr(valid[sentiment_col], valid[market_col])

        return {
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "n": len(valid),
        }

    def analyze_ticker(self, df: pd.DataFrame, ticker: str) -> dict:
        """
        Comprehensive analysis for a single ticker.
        """
        ticker_df = df[df["ticker"] == ticker].copy()
        ticker_df = ticker_df.sort_values("date")

        if len(ticker_df) < 5:
            return {"ticker": ticker, "error": "insufficient data"}

        results = {"ticker": ticker, "n_days": len(ticker_df)}

        # 1. Correlation: sentiment vs daily return
        results["return_corr"] = self.compute_correlation(
            ticker_df, "avg_compound", "daily_return"
        )

        # 2. Correlation: sentiment vs volatility
        vol_col = "volatility_5d" if "volatility_5d" in ticker_df.columns else "volatility_d"
        if vol_col in ticker_df.columns:
            results["volatility_corr"] = self.compute_correlation(
                ticker_df, "avg_compound", vol_col
            )

        # 3. Lagged correlation (sentiment today → return tomorrow)
        ticker_df["next_return"] = ticker_df["daily_return"].shift(-1)
        results["lagged_corr"] = self.compute_correlation(
            ticker_df, "avg_compound", "next_return"
        )

        # 4. Sentiment before/after big moves
        threshold = ticker_df["daily_return"].std()
        big_up = ticker_df[ticker_df["daily_return"] > threshold]
        big_down = ticker_df[ticker_df["daily_return"] < -threshold]

        results["avg_sentiment_big_up"] = round(big_up["avg_compound"].mean(), 4) if len(big_up) > 0 else None
        results["avg_sentiment_big_down"] = round(big_down["avg_compound"].mean(), 4) if len(big_down) > 0 else None

        return results

    def cross_model_comparison(self, vader_df: pd.DataFrame,
                               finbert_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare sentiment signals from VADER and FinBERT.
        Needs daily sentiment joined with market data.
        """
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager()
        comparison = []

        for ticker in vader_df["ticker"].unique():
            v_df = db.get_sentiment_with_market(ticker, "VADER")
            f_df = db.get_sentiment_with_market(ticker, "FinBERT")

            if len(v_df) < 3 or len(f_df) < 3:
                continue

            v_corr = self.compute_correlation(v_df, "avg_compound", "daily_return")
            f_corr = self.compute_correlation(f_df, "avg_compound", "daily_return")

            comparison.append({
                "ticker": ticker,
                "vader_pearson": v_corr["pearson_r"],
                "vader_pvalue": v_corr["pearson_p"],
                "finbert_pearson": f_corr["pearson_r"],
                "finbert_pvalue": f_corr["pearson_p"],
                "vader_n": v_corr["n"],
                "finbert_n": f_corr["n"],
            })

        return pd.DataFrame(comparison)

    # --- Visualization ---

    def plot_sentiment_vs_return(self, df: pd.DataFrame, ticker: str,
                                  model: str = "FinBERT",
                                  save: bool = True) -> plt.Figure:
        """Plot dual-axis chart: sentiment and daily return over time."""
        ticker_df = df[(df["ticker"] == ticker)].sort_values("date")

        if ticker_df.empty:
            print(f"    No data for {ticker}")
            return None

        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Sentiment on primary axis
        ax1.fill_between(ticker_df["date"], ticker_df["avg_compound"], 0,
                         alpha=0.3, color="steelblue")
        ax1.plot(ticker_df["date"], ticker_df["avg_compound"],
                 color="steelblue", linewidth=1.5, label="Sentiment")
        ax1.set_ylabel("Avg Sentiment Score", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Returns on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(ticker_df["date"], ticker_df["daily_return"],
                alpha=0.4, color="orange", width=0.8, label="Daily Return")
        ax2.set_ylabel("Daily Return", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        ax1.set_title(f"{ticker} — {model} Sentiment vs Daily Return")
        fig.autofmt_xdate()
        plt.tight_layout()

        if save:
            path = self.output_dir / f"{ticker}_{model}_sentiment_return.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"   Saved: {path}")

        return fig

    def plot_sentiment_distribution(self, vader_df: pd.DataFrame,
                                     finbert_df: pd.DataFrame,
                                     save: bool = True) -> plt.Figure:
        """Compare sentiment label distributions between models."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, df, name in [(axes[0], vader_df, "VADER"), (axes[1], finbert_df, "FinBERT")]:
            label_counts = df["label"].value_counts()
            colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#aaa"}
            label_counts.plot(kind="bar", ax=ax, color=[colors.get(l, "#888") for l in label_counts.index])
            ax.set_title(f"{name} Sentiment Distribution")
            ax.set_ylabel("Count")
            ax.set_xlabel("")

        plt.tight_layout()

        if save:
            path = self.output_dir / "sentiment_distribution_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"   Saved: {path}")

        return fig

    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                  save: bool = True) -> plt.Figure:
        """Plot correlation heatmap for sentiment and market variables."""
        cols = ["avg_compound", "avg_positive", "avg_negative", "close", "daily_return", "volume"]
        vol_col = "volatility_5d" if "volatility_5d" in df.columns else "volatility_d"
        if vol_col in df.columns:
            cols.append(vol_col)
        available = [c for c in cols if c in df.columns]

        corr_matrix = df[available].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0,
                    fmt=".2f", ax=ax, square=True, linewidths=0.5)
        ax.set_title("Sentiment-Market Correlation Matrix")
        plt.tight_layout()

        if save:
            path = self.output_dir / "correlation_heatmap.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"   Saved: {path}")

        return fig

    def generate_report(self, df: pd.DataFrame, model: str = "FinBERT") -> str:
        """Generate a text summary report of analysis findings."""
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager()
        report_lines = [
            "=" * 60,
            f"  FinSent-Market Analysis Report ({model})",
            "=" * 60,
            "",
            f"Total daily records: {len(df)}",
            f"Tickers analyzed: {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}",
            f"Date range: {df['date'].min()} → {df['date'].max()}" if 'date' in df.columns else "",
            "",
            "--- Correlation Analysis ---",
        ]

        for ticker in df["ticker"].unique() if "ticker" in df.columns else []:
            # Use joined data for analysis
            joined = db.get_sentiment_with_market(ticker, model)
            if len(joined) < 5:
                continue
            result = self.analyze_ticker(joined, ticker)
            if "error" in result:
                continue

            report_lines.append(f"\n  {ticker} ({result.get('n_days', 0)} days):")
            rc = result.get("return_corr", {})
            report_lines.append(f"    Sentiment vs Return:     r={rc.get('pearson_r', 'N/A')}, p={rc.get('pearson_p', 'N/A')}")
            vc = result.get("volatility_corr", {})
            if vc:
                report_lines.append(f"    Sentiment vs Volatility: r={vc.get('pearson_r', 'N/A')}, p={vc.get('pearson_p', 'N/A')}")
            lc = result.get("lagged_corr", {})
            report_lines.append(f"    Sentiment→Next Return:   r={lc.get('pearson_r', 'N/A')}, p={lc.get('pearson_p', 'N/A')}")

            if result.get("avg_sentiment_big_up") is not None:
                report_lines.append(f"    Avg sentiment (big up days):   {result['avg_sentiment_big_up']}")
            if result.get("avg_sentiment_big_down") is not None:
                report_lines.append(f"    Avg sentiment (big down days): {result['avg_sentiment_big_down']}")

        report_lines.append(f"\n{'=' * 60}")

        report = "\n".join(report_lines)

        report_path = self.output_dir / f"analysis_report_{model}.txt"
        report_path.write_text(report)
        print(f"\n  Report saved: {report_path}")

        return report


if __name__ == "__main__":
    analyzer = MarketSentimentAnalyzer()
    print("MarketSentimentAnalyzer initialized. Use via run_pipeline.py or dashboard.")

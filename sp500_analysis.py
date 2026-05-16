"""
S&P  Comprehensive Analysis.

Combines:
  . Real S&P  market data (yfinance,  years)
  . Financial PhraseBank headlines (, labeled sentences)
  . yfinance real-time news headlines (top  S&P components)
  
Runs full VADER + FinBERT sentiment pipeline and produces:
  - Per-ticker sentiment vs market correlation
  - Sentiment distribution analysis
  - Model comparison
  - Report-ready charts

Usage:
    python sp_analysis.py
    python sp_analysis.py --period y
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import re
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(, str(Path(__file__).parent))
from config import 

  S&P  Top Components 
SP_TOP = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "META": "Meta", "GOOGL": "Alphabet", "TSLA": "Tesla",
    "JPM": "JPMorgan", "V": "Visa", "XOM": "ExxonMobil",
    "UNH": "UnitedHealth", "MA": "Mastercard", "HD": "Home Depot",
    "COST": "Costco", "ABBV": "AbbVie", "CRM": "Salesforce",
    "AVGO": "Broadcom", "LLY": "Eli Lilly", "PG": "P&G", "BRK-B": "Berkshire",
}


def fetch_market_data(tickers: list, period: str = "y") -> pd.DataFrame:
    """Fetch OHLCV data via yfinance."""
    import yfinance as yf
    
    print(f"\n Fetching market data ({period})...")
    all_data = []
    all_tickers = list(tickers) + ["SPY"]
    
    for ticker in all_tickers:
        try:
            symbol = "^GSPC" if ticker == "SPY" else ticker
            t = yf.Ticker(symbol)
            hist = t.history(period=period, auto_adjust=True)
            if hist.empty:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel()
            
            hist = hist.reset_index()
            hist["ticker"] = ticker
            hist["daily_return"] = hist["Close"].pct_change()
            hist["volatility_d"] = hist["daily_return"].rolling().std()
            hist["volatility_d"] = hist["daily_return"].rolling().std()
            hist["cum_return"] = ( + hist["daily_return"]).cumprod() - 
            
            hist = hist.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            cols = ["ticker", "date", "open", "high", "low", "close", "volume",
                    "daily_return", "volatility_d", "volatility_d", "cum_return"]
            hist = hist[[c for c in cols if c in hist.columns]]
            print(f"   {ticker}: {len(hist)} trading days")
            all_data.append(hist)
        except Exception as e:
            print(f"    {ticker}: {e}")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
        print(f"\n   Total: {len(result)} records")
        return result
    return pd.DataFrame()


def fetch_yfinance_news(tickers: list) -> pd.DataFrame:
    """Fetch recent news from yfinance (new API format)."""
    import yfinance as yf
    
    print(f"\n Fetching real-time news from yfinance...")
    all_news = []
    seen = set()
    
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            items = t.news or []
            count = 
            for item in items:
                content = item.get("content", item)
                title = content.get("title", "")
                if not title or title in seen:
                    continue
                seen.add(title)
                
                pub_date_str = content.get("pubDate", "")
                pub_date = pd.to_datetime(pub_date_str) if pub_date_str else datetime.now()
                
                provider = content.get("provider", {})
                publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
                
                url = ""
                ct = content.get("clickThroughUrl") or content.get("canonicalUrl")
                if isinstance(ct, dict):
                    url = ct.get("url", "")
                
                summary = content.get("summary", "") or ""
                summary = re.sub(r"<[^>]+>", "", summary)
                
                all_news.append({
                    "ticker": ticker, "headline": title,
                    "summary": summary, "source": publisher, "url": url,
                    "published_at": pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": pub_date.strftime("%Y-%m-%d"),
                })
                count += 
            if count:
                print(f"   {ticker}: {count} articles")
        except Exception as e:
            print(f"    {ticker}: {e}")
    
    if all_news:
        df = pd.DataFrame(all_news).drop_duplicates(subset=["headline"])
        print(f"\n   Total: {len(df)} unique articles")
        return df
    return pd.DataFrame()


def load_phrasebank_with_dates() -> pd.DataFrame:
    """Load Financial PhraseBank and assign synthetic dates for analysis."""
    from evaluate_phrasebank import load_phrasebank
    
    print(f"\n Loading Financial PhraseBank...")
    df = load_phrasebank("sentences_allagree")
    
     Assign random dates within a -year window for temporal analysis
    np.random.seed()
    start = pd.Timestamp("--")
    end = pd.Timestamp("--")
    date_range = pd.bdate_range(start, end)
    
    dates = np.random.choice(date_range, size=len(df))
    df["date"] = pd.to_datetime(dates).strftime("%Y-%m-%d")
    df["published_at"] = df["date"] + " ::"
    
     Assign tickers based on sentence content
    ticker_keywords = {
        "AAPL": ["apple", "iphone", "ipad", "mac"],
        "MSFT": ["microsoft", "windows", "azure", "surface"],
        "GOOGL": ["google", "alphabet", "android", "youtube"],
        "AMZN": ["amazon", "aws", "prime"],
        "TSLA": ["tesla", "model s", "model ", "elon musk"],
        "NVDA": ["nvidia", "gpu", "geforce"],
        "META": ["meta", "facebook", "instagram", "whatsapp"],
        "JPM": ["jpmorgan", "jpm", "chase"],
        "XOM": ["exxon", "xom"],
    }
    
    df["ticker"] = "SPY"   Default: general market
    for ticker, keywords in ticker_keywords.items():
        mask = df["sentence"].str.lower().apply(
            lambda x: any(kw in str(x) for kw in keywords)
        )
        df.loc[mask, "ticker"] = ticker
    
    df = df.rename(columns={"sentence": "headline"})
    df["source"] = "FinancialPhraseBank"
    df["summary"] = ""
    df["url"] = ""
    
    print(f"   Ticker distribution: {df['ticker'].value_counts().head().to_dict()}")
    return df


def run_sentiment(news_df: pd.DataFrame) -> tuple:
    """Run VADER + FinBERT."""
    from src.sentiment.vader_analyzer import VaderAnalyzer
    from src.sentiment.finbert_analyzer import FinBERTAnalyzer
    from src.preprocessing.preprocessor import TextPreprocessor
    
    pp = TextPreprocessor()
    news_df["headline_clean"] = news_df["headline"].apply(pp.clean_text)
    
    vader = VaderAnalyzer()
    vader_results = vader.analyze_batch(news_df, text_col="headline_clean")
    
    finbert = FinBERTAnalyzer()
    finbert_results = finbert.analyze_batch(news_df, text_col="headline_clean")
    
    return vader_results, finbert_results


def analyze(vader_results, finbert_results, market_df, output_dir):
    """Full correlation analysis."""
    print(f"\n{'='}")
    print(f"  Sentiment-Market Correlation Analysis")
    print(f"{'='}")
    
    analysis = {}
    
    for model_name, results in [("VADER", vader_results), ("FinBERT", finbert_results)]:
         Aggregate daily sentiment
        daily = results.groupby(["ticker", "date"]).agg(
            avg_compound=("compound_score", "mean"),
            std_compound=("compound_score", "std"),
            article_count=("headline", "count"),
            pct_positive=("label", lambda x: (x == "positive").mean()),
            pct_negative=("label", lambda x: (x == "negative").mean()),
        ).reset_index()
        
         Merge with market
        merged = daily.merge(
            market_df[["ticker", "date", "close", "daily_return",
                        "volatility_d", "volume"]],
            on=["ticker", "date"], how="inner"
        ).sort_values(["ticker", "date"])
        
        if merged.empty:
             Try assigning market data by date (for SPY-level analysis)
            spy = market_df[market_df["ticker"] == "SPY"][["date", "close", "daily_return", "volatility_d"]]
            daily_no_ticker = results.groupby("date").agg(
                avg_compound=("compound_score", "mean"),
                article_count=("headline", "count"),
                pct_positive=("label", lambda x: (x == "positive").mean()),
                pct_negative=("label", lambda x: (x == "negative").mean()),
            ).reset_index()
            merged = daily_no_ticker.merge(spy, on="date", how="inner")
            if not merged.empty:
                merged["ticker"] = "SPY"
        
        if merged.empty:
            print(f"    {model_name}: No overlapping data")
            continue
        
        print(f"\n  {model_name}: {len(merged)} overlapping observations")
        
         Correlations
        findings = {}
        valid = merged.dropna(subset=["avg_compound", "daily_return"])
        
        if len(valid) >= :
            r_ret, p_ret = stats.pearsonr(valid["avg_compound"], valid["daily_return"])
            sr_ret, sp_ret = stats.spearmanr(valid["avg_compound"], valid["daily_return"])
            findings["sentiment_vs_return"] = {
                "pearson_r": round(r_ret, ), "pearson_p": round(p_ret, ),
                "spearman_r": round(sr_ret, ), "spearman_p": round(sp_ret, ),
                "n": len(valid),
                "significant": p_ret < .,
            }
            
             Volatility
            vol_valid = merged.dropna(subset=["avg_compound", "volatility_d"])
            if len(vol_valid) >= :
                r_vol, p_vol = stats.pearsonr(vol_valid["avg_compound"], vol_valid["volatility_d"])
                findings["sentiment_vs_volatility"] = {
                    "pearson_r": round(r_vol, ), "pearson_p": round(p_vol, ), "n": len(vol_valid),
                }
            
             Directional accuracy
            same_sign = (np.sign(valid["avg_compound"]) == np.sign(valid["daily_return"])).mean()
            findings["directional_accuracy"] = round(same_sign, )
            
             Sentiment on big move days
            if len(valid) >= :
                std_ret = valid["daily_return"].std()
                big_up = valid[valid["daily_return"] > std_ret]
                big_down = valid[valid["daily_return"] < -std_ret]
                findings["sentiment_big_up_days"] = round(big_up["avg_compound"].mean(), ) if len(big_up) >  else None
                findings["sentiment_big_down_days"] = round(big_down["avg_compound"].mean(), ) if len(big_down) >  else None
                findings["sentiment_normal_days"] = round(valid[abs(valid["daily_return"]) <= std_ret]["avg_compound"].mean(), )
                findings["n_big_up"] = len(big_up)
                findings["n_big_down"] = len(big_down)
            
             Lagged: today's sentiment -> tomorrow's return
            merged_sorted = merged.sort_values("date")
            merged_sorted["next_return"] = merged_sorted["daily_return"].shift(-)
            lag_valid = merged_sorted.dropna(subset=["avg_compound", "next_return"])
            if len(lag_valid) >= :
                r_lag, p_lag = stats.pearsonr(lag_valid["avg_compound"], lag_valid["next_return"])
                findings["sentiment_vs_next_return"] = {
                    "pearson_r": round(r_lag, ), "pearson_p": round(p_lag, ), "n": len(lag_valid),
                }
        
        analysis[model_name] = {"merged": merged, "findings": findings}
    
    return analysis


def generate_charts(analysis, market_df, vader_results, finbert_results, output_dir):
    """Generate all charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
     . S&P  price chart
    spy = market_df[market_df["ticker"] == "SPY"].copy()
    if not spy.empty:
        fig, ax = plt.subplots(figsize=(, ))
        ax.plot(spy["date"], spy["close"], color="ce", linewidth=.)
        ax.fill_between(spy["date"], spy["low"], spy["high"], alpha=., color="db")
        ax.set_title("S&P  (^GSPC)  Price Range", fontsize=)
        ax.set_ylabel("Price")
        plt.xticks(rotation=, ha='right')
        ax.xaxis.set_major_locator(plt.MaxNLocator())
        plt.tight_layout()
        fig.savefig(output_dir / "sp_price.png", dpi=, bbox_inches="tight")
        plt.close()
        print(f"   sp_price.png")
    
     . Per-model charts
    for model_name, data in analysis.items():
        merged = data["merged"]
        if len(merged) < :
            continue
        
         Sentiment time series
        fig, ax = plt.subplots(figsize=(, ))
        ax.fill_between(merged["date"], merged["avg_compound"], , alpha=., color="steelblue")
        ax.plot(merged["date"], merged["avg_compound"], color="steelblue", linewidth=., label="Sentiment")
        ax.set_ylabel("Avg Sentiment", color="steelblue")
        ax.axhline(y=, color="gray", linestyle="--", alpha=.)
        
        if "daily_return" in merged.columns:
            ax = ax.twinx()
            ax.bar(merged["date"], merged["daily_return"], alpha=., color="orange", label="Return")
            ax.set_ylabel("Daily Return", color="orange")
        
        ax.set_title(f"S&P   {model_name} Sentiment vs Market", fontsize=)
        fig.autofmt_xdate()
        plt.tight_layout()
        fig.savefig(output_dir / f"sp_{model_name.lower()}_timeseries.png", dpi=, bbox_inches="tight")
        plt.close()
        print(f"   sp_{model_name.lower()}_timeseries.png")
        
         Scatter: sentiment vs return
        valid = merged.dropna(subset=["avg_compound", "daily_return"])
        if len(valid) >= :
            fig, ax = plt.subplots(figsize=(, ))
            ax.scatter(valid["avg_compound"], valid["daily_return"], alpha=., s=,
                      edgecolors="white", linewidths=., c="steelblue")
            
            z = np.polyfit(valid["avg_compound"], valid["daily_return"], )
            p_line = np.polyd(z)
            x_line = np.linspace(valid["avg_compound"].min(), valid["avg_compound"].max(), )
            ax.plot(x_line, p_line(x_line), "r--", alpha=., linewidth=)
            
            r, pval = stats.pearsonr(valid["avg_compound"], valid["daily_return"])
            sig = " " if pval < . else ""
            ax.text(., ., f"r = {r:.f}{sig}\np = {pval:.f}\nn = {len(valid)}",
                    transform=ax.transAxes, fontsize=, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=.))
            
            ax.set_xlabel("Avg Sentiment Score")
            ax.set_ylabel("Daily Return")
            ax.set_title(f"{model_name}  Sentiment vs Return")
            ax.axhline(y=, color="gray", linestyle="--", alpha=.)
            ax.axvline(x=, color="gray", linestyle="--", alpha=.)
            plt.tight_layout()
            fig.savefig(output_dir / f"sp_{model_name.lower()}_scatter.png", dpi=, bbox_inches="tight")
            plt.close()
            print(f"   sp_{model_name.lower()}_scatter.png")
        
         Heatmap
        hm_cols = ["avg_compound", "pct_positive", "pct_negative", "daily_return", "volatility_d"]
        available = [c for c in hm_cols if c in merged.columns]
        if len(available) >= :
            corr = merged[available].corr()
            fig, ax = plt.subplots(figsize=(, ))
            sns.heatmap(corr, annot=True, cmap="RdBu_r", center=, fmt=".f",
                       ax=ax, square=True, linewidths=.)
            ax.set_title(f"{model_name}  Correlation Matrix")
            plt.tight_layout()
            fig.savefig(output_dir / f"sp_{model_name.lower()}_heatmap.png", dpi=, bbox_inches="tight")
            plt.close()
            print(f"   sp_{model_name.lower()}_heatmap.png")
    
     . Sentiment distribution
    fig, axes = plt.subplots(, , figsize=(, ))
    colors = {"positive": "ecc", "negative": "ecc", "neutral": "aa"}
    for ax, df, name in [(axes[], vader_results, "VADER"), (axes[], finbert_results, "FinBERT")]:
        counts = df["label"].value_counts()
        counts.plot(kind="bar", ax=ax, color=[colors.get(l, "") for l in counts.index])
        ax.set_title(f"{name} ({len(df)} headlines)")
        ax.set_ylabel("Count")
    plt.suptitle("Sentiment Distribution  Real Financial Headlines", fontsize=)
    plt.tight_layout()
    fig.savefig(output_dir / "sp_sentiment_distribution.png", dpi=, bbox_inches="tight")
    plt.close()
    print(f"   sp_sentiment_distribution.png")
    
     . VADER vs FinBERT comparison on same headlines
    if len(vader_results) == len(finbert_results):
        comparison = pd.DataFrame({
            "headline": vader_results["headline"],
            "vader_label": vader_results["label"],
            "finbert_label": finbert_results["label"],
            "vader_compound": vader_results["compound_score"],
            "finbert_compound": finbert_results["compound_score"],
        })
        agreement = (comparison["vader_label"] == comparison["finbert_label"]).mean()
        
        fig, ax = plt.subplots(figsize=(, ))
        ax.scatter(comparison["vader_compound"], comparison["finbert_compound"],
                  alpha=., s=, c="steelblue")
        ax.axhline(y=, color="gray", linestyle="--", alpha=.)
        ax.axvline(x=, color="gray", linestyle="--", alpha=.)
        ax.set_xlabel("VADER Compound Score")
        ax.set_ylabel("FinBERT Compound Score")
        ax.set_title(f"VADER vs FinBERT Scores (agreement: {agreement:.%})")
        plt.tight_layout()
        fig.savefig(output_dir / "sp_model_agreement.png", dpi=, bbox_inches="tight")
        plt.close()
        print(f"   sp_model_agreement.png")
        
         Agreement heatmap
        from sklearn.metrics import confusion_matrix
        labels = ["negative", "neutral", "positive"]
        cm = confusion_matrix(comparison["vader_label"], comparison["finbert_label"], labels=labels)
        fig, ax = plt.subplots(figsize=(, ))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                   xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("FinBERT")
        ax.set_ylabel("VADER")
        ax.set_title(f"Model Agreement Matrix (Overall: {agreement:.%})")
        plt.tight_layout()
        fig.savefig(output_dir / "sp_agreement_matrix.png", dpi=, bbox_inches="tight")
        plt.close()
        print(f"   sp_agreement_matrix.png")


def print_report(analysis):
    """Print text report."""
    print(f"\n{'='}")
    print(f"  S&P  SENTIMENT-MARKET ANALYSIS REPORT")
    print(f"{'='}")
    
    for model_name, data in analysis.items():
        f = data["findings"]
        merged = data["merged"]
        
        print(f"\n{''}")
        print(f"  {model_name}")
        print(f"{''}")
        print(f"  Overlapping observations: {len(merged)}")
        
        if "sentiment_vs_return" in f:
            sr = f["sentiment_vs_return"]
            sig = "  SIGNIFICANT" if sr["significant"] else ""
            print(f"\n   Sentiment vs Daily Return:")
            print(f"     Pearson r  = {sr['pearson_r']:+.f}  (p = {sr['pearson_p']:.f}){sig}")
            print(f"     Spearman r = {sr['spearman_r']:+.f}  (p = {sr['spearman_p']:.f})")
            print(f"     n = {sr['n']}")
        
        if "sentiment_vs_next_return" in f:
            snr = f["sentiment_vs_next_return"]
            sig = "  SIGNIFICANT" if snr["pearson_p"] < . else ""
            print(f"\n   Sentiment  Next-Day Return (Predictive):")
            print(f"     Pearson r = {snr['pearson_r']:+.f}  (p = {snr['pearson_p']:.f}){sig}")
            print(f"     n = {snr['n']}")
        
        if "sentiment_vs_volatility" in f:
            sv = f["sentiment_vs_volatility"]
            print(f"\n   Sentiment vs -Day Volatility:")
            print(f"     Pearson r = {sv['pearson_r']:+.f}  (p = {sv['pearson_p']:.f})")
        
        if "directional_accuracy" in f:
            print(f"\n   Directional Accuracy: {f['directional_accuracy']:.%}")
            print(f"     (Does sentiment sign match return sign?)")
        
        if "sentiment_big_up_days" in f and f["sentiment_big_up_days"] is not None:
            print(f"\n   Sentiment on Extreme Days:")
            print(f"     Big up days   (>+Ï): avg sentiment = {f['sentiment_big_up_days']:+.f}  (n={f.get('n_big_up', '?')})")
            print(f"     Big down days (<-Ï): avg sentiment = {f['sentiment_big_down_days']:+.f}  (n={f.get('n_big_down', '?')})")
            print(f"     Normal days:          avg sentiment = {f.get('sentiment_normal_days', 'N/A')}")
    
     Comparison
    if "VADER" in analysis and "FinBERT" in analysis:
        v_f = analysis["VADER"]["findings"]
        f_f = analysis["FinBERT"]["findings"]
        
        print(f"\n{''}")
        print(f"  VADER vs FinBERT Comparison")
        print(f"{''}")
        
        if "sentiment_vs_return" in v_f and "sentiment_vs_return" in f_f:
            print(f"  {'Metric':<} {'VADER':>} {'FinBERT':>}")
            print(f"  {''}")
            print(f"  {'Pearson r (vs return)':<} {v_f['sentiment_vs_return']['pearson_r']:>+.f} {f_f['sentiment_vs_return']['pearson_r']:>+.f}")
            print(f"  {'p-value':<} {v_f['sentiment_vs_return']['pearson_p']:>.f} {f_f['sentiment_vs_return']['pearson_p']:>.f}")
        
        if "directional_accuracy" in v_f and "directional_accuracy" in f_f:
            print(f"  {'Directional accuracy':<} {v_f['directional_accuracy']:>.%} {f_f['directional_accuracy']:>.%}")
        
        if "sentiment_vs_next_return" in v_f and "sentiment_vs_next_return" in f_f:
            print(f"  {'Pearson r (vs next-day return)':<} {v_f['sentiment_vs_next_return']['pearson_r']:>+.f} {f_f['sentiment_vs_next_return']['pearson_r']:>+.f}")
    
    print(f"\n{'='}\n")


def main():
    parser = argparse.ArgumentParser(description="S&P  Sentiment-Market Analysis")
    parser.add_argument("--period", default="y", help="Market data period (mo, mo, mo, y, y)")
    args = parser.parse_args()
    
    output_dir = PROCESSED_DIR / "sp_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = list(SP_TOP.keys())
    
    print("=")
    print("  S&P  Comprehensive Sentiment Analysis")
    print("  Real Market Data + Financial News")
    print("=")
    
     Step : Market data
    market_df = fetch_market_data(tickers, args.period)
    if market_df.empty:
        print(" No market data.")
        return
    market_df.to_csv(output_dir / "market_data.csv", index=False)
    
     Step : News data  combine yfinance + PhraseBank
    yf_news = fetch_yfinance_news(tickers)
    pb_news = load_phrasebank_with_dates()
    
    news_dfs = []
    if not yf_news.empty:
        news_dfs.append(yf_news[["ticker", "headline", "summary", "source", "published_at", "date"]])
    if not pb_news.empty:
        news_dfs.append(pb_news[["ticker", "headline", "summary", "source", "published_at", "date"]])
    
    if news_dfs:
        news_df = pd.concat(news_dfs, ignore_index=True).drop_duplicates(subset=["headline"])
    else:
        print(" No news data.")
        return
    
    print(f"\n Combined news dataset: {len(news_df)} headlines")
    news_df.to_csv(output_dir / "news_data.csv", index=False)
    
     Step : Sentiment analysis
    print(f"\n{'='}")
    print(f"  SENTIMENT ANALYSIS")
    print(f"{'='}")
    
    vader_results, finbert_results = run_sentiment(news_df)
    vader_results.to_csv(output_dir / "vader_sentiment.csv", index=False)
    finbert_results.to_csv(output_dir / "finbert_sentiment.csv", index=False)
    
     Step : Correlation analysis
    analysis = analyze(vader_results, finbert_results, market_df, output_dir)
    
     Step : Charts
    print(f"\n{'='}")
    print(f"  GENERATING CHARTS")
    print(f"{'='}")
    
    generate_charts(analysis, market_df, vader_results, finbert_results, output_dir)
    
     Step : Report
    print_report(analysis)
    
     Save JSON
    saveable = {}
    for model, data in analysis.items():
        saveable[model] = {
            "n_observations": len(data["merged"]),
            "findings": data["findings"],
        }
    with open(output_dir / "findings.json", "w") as f:
        json.dump(saveable, f, indent=, default=str)
    
    print(f" All outputs: {output_dir}")
    print(f" Analysis complete!")


if __name__ == "__main__":
    main()

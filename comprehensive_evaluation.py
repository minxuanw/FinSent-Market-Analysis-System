"""
Comprehensive Model Evaluation and Pipeline Reliability Analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy import text
import sys
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR
from src.database.db_manager import DatabaseManager


def compute_lagged_correlation(db, tickers, models=None):
    models = models or ["VADER", "FinBERT"]
    results = []
    for model in models:
        for ticker in tickers:
            df = db.get_sentiment_with_market(ticker, model)
            if df.empty or len(df) < 10:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            for lag in range(4):
                if lag == 0:
                    sent = df["avg_compound"]
                    ret = df["daily_return"]
                else:
                    sent = df["avg_compound"].iloc[:-lag].reset_index(drop=True)
                    ret = df["daily_return"].iloc[lag:].reset_index(drop=True)
                min_len = min(len(sent), len(ret))
                sent = sent.iloc[:min_len].reset_index(drop=True)
                ret = ret.iloc[:min_len].reset_index(drop=True)
                mask = ~(sent.isna() | ret.isna())
                sent, ret = sent[mask], ret[mask]
                if len(sent) < 5:
                    continue
                pr, pp = stats.pearsonr(sent, ret)
                sr, sp = stats.spearmanr(sent, ret)
                results.append({
                    "model": model, "ticker": ticker, "lag": lag,
                    "pearson_r": round(pr, 4), "pearson_p": round(pp, 4),
                    "spearman_r": round(sr, 4), "spearman_p": round(sp, 4),
                    "n_observations": len(sent), "significant_005": pp < 0.05,
                })
    return pd.DataFrame(results)


def compute_car(db, tickers, models=None, threshold=0.3, window=3):
    models = models or ["VADER", "FinBERT"]
    results = []
    market_df = db.get_market_data("^GSPC")
    market_bench = market_df[["date", "daily_return"]].rename(
        columns={"daily_return": "market_return"}) if not market_df.empty else pd.DataFrame()
    for model in models:
        for ticker in tickers:
            df = db.get_sentiment_with_market(ticker, model)
            if df.empty or len(df) < 20:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            if not market_bench.empty:
                df = df.merge(market_bench, on="date", how="left")
                df["market_return"] = df["market_return"].fillna(0)
            else:
                df["market_return"] = 0
            df["abnormal_return"] = df["daily_return"] - df["market_return"]
            for event_type, mask_val in [("positive", df["avg_compound"] > threshold),
                                          ("negative", df["avg_compound"] < -threshold)]:
                events = df[mask_val].index
                car_values = []
                for idx in events:
                    end_idx = min(idx + window + 1, len(df))
                    if idx >= len(df) - 1:
                        continue
                    car_values.append(df.loc[idx:end_idx - 1, "abnormal_return"].sum())
                if len(car_values) < 3:
                    continue
                car_mean = np.mean(car_values)
                car_std = np.std(car_values, ddof=1)
                t_stat = car_mean / (car_std / np.sqrt(len(car_values)))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(car_values) - 1))
                results.append({
                    "model": model, "ticker": ticker, "event_type": event_type,
                    "n_events": len(car_values), "threshold": threshold,
                    "window_days": window, "car_mean": round(car_mean, 4),
                    "car_std": round(car_std, 4), "t_stat": round(t_stat, 4),
                    "p_value": round(p_value, 4), "significant_005": p_value < 0.05,
                })
    return pd.DataFrame(results)


def compute_information_coefficient(db, tickers, models=None):
    models = models or ["VADER", "FinBERT"]
    results = []
    for model in models:
        for ticker in tickers:
            df = db.get_sentiment_with_market(ticker, model)
            if df.empty or len(df) < 10:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            for lag in range(4):
                if lag == 0:
                    valid = df[["avg_compound", "daily_return"]].dropna()
                    if len(valid) < 5:
                        continue
                    ic, ic_p = stats.spearmanr(valid["avg_compound"], valid["daily_return"])
                else:
                    sent = df["avg_compound"].iloc[:-lag].reset_index(drop=True)
                    fwd = df["daily_return"].iloc[lag:].reset_index(drop=True)
                    mask = ~(sent.isna() | fwd.isna())
                    if mask.sum() < 5:
                        continue
                    ic, ic_p = stats.spearmanr(sent[mask], fwd[mask])
                results.append({
                    "model": model, "ticker": ticker, "lag": lag,
                    "IC": round(ic, 4), "p_value": round(ic_p, 4),
                    "significant_005": ic_p < 0.05, "n": len(valid) if lag == 0 else int(mask.sum()),
                })
    return pd.DataFrame(results)


def walk_forward_validation(db, tickers, models=None, train_ratio=0.7, n_splits=3):
    models = models or ["VADER", "FinBERT"]
    results = []
    for model in models:
        for ticker in tickers:
            df = db.get_sentiment_with_market(ticker, model)
            if df.empty or len(df) < 15:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            n = len(df)
            for split in range(n_splits):
                train_end = int(n * (train_ratio + (1 - train_ratio) * split / n_splits))
                train_end = min(train_end, n - 5)
                train, test = df.iloc[:train_end], df.iloc[train_end:]
                if len(train) < 5 or len(test) < 3:
                    continue
                for label, subset in [("train", train), ("test", test)]:
                    valid = subset[["avg_compound", "daily_return"]].dropna()
                    if len(valid) < 3:
                        corr_val, p_val = np.nan, np.nan
                    else:
                        corr_val, p_val = stats.pearsonr(valid["avg_compound"], valid["daily_return"])
                    results.append({
                        "model": model, "ticker": ticker, "split": split, "set": label,
                        "train_start": str(train["date"].iloc[0]),
                        "train_end": str(train["date"].iloc[-1]),
                        "test_start": str(test["date"].iloc[0]),
                        "test_end": str(test["date"].iloc[-1]),
                        "n_samples": len(valid),
                        "correlation": round(corr_val, 4) if not np.isnan(corr_val) else None,
                        "p_value": round(p_val, 4) if not np.isnan(p_val) else None,
                    })
    return pd.DataFrame(results)


def confidence_score_analysis(db, tickers, models=None, threshold=0.6):
    models = models or ["VADER", "FinBERT"]
    results = []
    for model in models:
        for ticker in tickers:
            query = text("""
                SELECT ss.label, ss.compound_score, ss.score_positive,
                       ss.score_negative, ss.score_neutral,
                       na.published_at, na.ticker
                FROM sentiment_scores ss
                JOIN news_articles na ON ss.article_id = na.id
                WHERE na.ticker = :ticker AND ss.model = :model
                ORDER BY na.published_at
            """)
            df = pd.read_sql(query, db.engine, params={"ticker": ticker, "model": model})
            if df.empty:
                continue
            df["confidence"] = df[["score_positive", "score_negative", "score_neutral"]].max(axis=1)
            market = db.get_market_data(ticker)
            if market.empty:
                continue
            df["date"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.date.astype(str)
            daily_all = df.groupby("date")["compound_score"].mean().reset_index()
            daily_all.columns = ["date", "sentiment_all"]
            high_conf = df[df["confidence"] >= threshold]
            if len(high_conf) < 3:
                daily_hc = daily_all.copy()
                daily_hc.columns = ["date", "sentiment_hc"]
            else:
                daily_hc = high_conf.groupby("date")["compound_score"].mean().reset_index()
                daily_hc.columns = ["date", "sentiment_hc"]
            merged = market[["date", "daily_return"]].merge(daily_all, on="date", how="inner")
            merged = merged.merge(daily_hc, on="date", how="inner").dropna()
            if len(merged) < 5:
                continue
            corr_all, p_all = stats.pearsonr(merged["sentiment_all"], merged["daily_return"])
            corr_hc, p_hc = stats.pearsonr(merged["sentiment_hc"], merged["daily_return"])
            results.append({
                "model": model, "ticker": ticker, "threshold": threshold,
                "n_total": len(df), "n_high_confidence": len(high_conf),
                "pct_high_confidence": round(len(high_conf) / len(df) * 100, 1),
                "corr_all": round(corr_all, 4), "p_all": round(p_all, 4),
                "corr_high_conf": round(corr_hc, 4), "p_high_conf": round(p_hc, 4),
                "improvement": round(corr_hc - corr_all, 4),
            })
    return pd.DataFrame(results)


def data_quality_report(db, tickers):
    from datetime import datetime
    with db._connect() as conn:
        news_count = conn.execute(text("SELECT COUNT(*) FROM news_articles")).scalar()
        news_tickers = conn.execute(text(
            "SELECT ticker, COUNT(*) as cnt FROM news_articles GROUP BY ticker"
        )).fetchall()
        news_range = conn.execute(text(
            "SELECT MIN(published_at), MAX(published_at) FROM news_articles"
        )).fetchone()
        market_count = conn.execute(text("SELECT COUNT(*) FROM market_data")).scalar()
        market_range = conn.execute(text(
            "SELECT MIN(date), MAX(date) FROM market_data"
        )).fetchone()
        sent_count = conn.execute(text("SELECT COUNT(*) FROM sentiment_scores")).scalar()
        join_count = conn.execute(text("""
            SELECT COUNT(DISTINCT ds.date) FROM daily_sentiment ds
            JOIN market_data md ON ds.date = md.date AND ds.ticker = md.ticker
        """)).scalar()
        total_sent = conn.execute(text(
            "SELECT COUNT(DISTINCT date) FROM daily_sentiment"
        )).scalar()
    ticker_coverage = {t: c for t, c in news_tickers}
    missing = [t for t in tickers if t not in ticker_coverage]
    try:
        last = pd.to_datetime(str(news_range[1]))
        days_since = (datetime.now() - last.replace(tzinfo=None)).days
    except Exception:
        days_since = None
    return {
        "news_articles": {"total": news_count, "date_range": [str(news_range[0]), str(news_range[1])],
                          "ticker_coverage": ticker_coverage, "missing_tickers": missing,
                          "days_since_last": days_since},
        "market_data": {"total": market_count, "date_range": [str(market_range[0]), str(market_range[1])]},
        "sentiment_scores": {"total": sent_count},
        "alignment": {"aligned_days": join_count, "total_sentiment_days": total_sent,
                      "rate_pct": round(join_count / total_sent * 100, 1) if total_sent > 0 else 0},
    }


def run_full_evaluation():
    print("=" * 60)
    print("  FinSent-Market Comprehensive Evaluation")
    print("=" * 60)
    db = DatabaseManager()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "BAC", "XOM"]
    out = PROCESSED_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Lagged Correlation
    print("\n[1/6] Lagged correlation (lag 0-3)...")
    lag_df = compute_lagged_correlation(db, tickers)
    lag_df.to_csv(out / "lagged_correlation.csv", index=False)
    print(f"  {len(lag_df)} rows saved")
    for model in ["VADER", "FinBERT"]:
        mdf = lag_df[lag_df["model"] == model]
        for lag in range(4):
            ldf = mdf[mdf["lag"] == lag]
            if not ldf.empty:
                print(f"    {model} lag {lag}: avg r={ldf['pearson_r'].mean():.4f}, sig={ldf['significant_005'].sum()}/{len(ldf)}")

    # 2. CAR Event Study
    print("\n[2/6] CAR event study...")
    car_df = compute_car(db, tickers)
    car_df.to_csv(out / "car_event_study.csv", index=False)
    print(f"  {len(car_df)} rows saved")
    for _, r in car_df.iterrows():
        s = "***" if r.get("significant_005") else ""
        print(f"    {r['model']} {r['ticker']} {r['event_type']}: CAR={r['car_mean']:.4f} t={r['t_stat']:.2f} p={r['p_value']:.4f} {s}")

    # 3. IC
    print("\n[3/6] Information Coefficient...")
    ic_df = compute_information_coefficient(db, tickers)
    ic_df.to_csv(out / "information_coefficient.csv", index=False)
    print(f"  {len(ic_df)} rows saved")
    for model in ["VADER", "FinBERT"]:
        mdf = ic_df[ic_df["model"] == model]
        for lag in range(4):
            ldf = mdf[mdf["lag"] == lag]
            if not ldf.empty:
                print(f"    {model} lag {lag}: avg IC={ldf['IC'].mean():.4f}, sig={ldf['significant_005'].sum()}/{len(ldf)}")

    # 4. Walk-Forward
    print("\n[4/6] Walk-forward validation...")
    wf_df = walk_forward_validation(db, tickers)
    wf_df.to_csv(out / "walk_forward_validation.csv", index=False)
    print(f"  {len(wf_df)} rows saved")
    for model in ["VADER", "FinBERT"]:
        mdf = wf_df[wf_df["model"] == model]
        tr = mdf[mdf["set"] == "train"]["correlation"].dropna()
        te = mdf[mdf["set"] == "test"]["correlation"].dropna()
        if len(tr) > 0 and len(te) > 0:
            print(f"    {model}: train r={tr.mean():.4f}, test r={te.mean():.4f}, gap={tr.mean()-te.mean():.4f}")

    # 5. Confidence
    print("\n[5/6] Confidence score analysis...")
    conf_df = confidence_score_analysis(db, tickers)
    conf_df.to_csv(out / "confidence_analysis.csv", index=False)
    print(f"  {len(conf_df)} rows saved")
    for _, r in conf_df.iterrows():
        d = "up" if r["improvement"] > 0 else "down"
        print(f"    {r['model']} {r['ticker']}: all={r['corr_all']:.4f} hc={r['corr_high_conf']:.4f} ({d} {abs(r['improvement']):.4f})")

    # 6. Data Quality
    print("\n[6/6] Data quality report...")
    quality = data_quality_report(db, tickers)
    with open(out / "data_quality_report.json", "w") as f:
        json.dump(quality, f, indent=2)
    print(f"  News: {quality['news_articles']['total']}, Market: {quality['market_data']['total']}")
    print(f"  Alignment: {quality['alignment']['rate_pct']}%")

    print("\n" + "=" * 60)
    print("  DONE - All outputs in:", out)
    print("=" * 60)


if __name__ == "__main__":
    run_full_evaluation()

"""
Streamlit Dashboard for FinSent-Market Analysis System.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, PROCESSED_DIR
from src.database.db_manager import DatabaseManager
from sqlalchemy import text

st.set_page_config(
    page_title="FinSent-Market Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(to right, #1f77b4, #9b59b6, #e74c3c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 1rem;
    }
    .section-divider {
        border: none !important; height: 3px !important;
        background: linear-gradient(to right, #1f77b4, #9b59b6, #e74c3c) !important;
        border-radius: 2px; margin: 1.5rem 0;
    }
    hr.section-divider {
        border: none !important;
        background: linear-gradient(to right, #1f77b4, #9b59b6, #e74c3c) !important;
        height: 3px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()


def gradient_heading(text):
    return f'<h2 style="background: linear-gradient(to right, #1f77b4, #9b59b6, #e74c3c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">{text}</h2>'


def plot_sentiment_time_series(df, ticker, model):
    df = df.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_compound"], name="Sentiment",
        fill="tozeroy", fillcolor="rgba(31,119,180,0.2)",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df["daily_return"], name="Daily Return",
        marker_color="#f1c40f", yaxis="y2", opacity=0.6,
    ))
    fig.update_layout(
        title=f"{ticker} - {model} Sentiment vs Daily Return",
        xaxis_title="Date", yaxis_title="Sentiment Score",
        yaxis2=dict(title="Daily Return (%)", overlaying="y", side="right"),
        hovermode="x unified", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_sentiment_distribution(ticker, model):
    query = """
        SELECT ss.label, COUNT(*) as count
        FROM sentiment_scores ss
        JOIN news_articles na ON ss.article_id = na.id
        WHERE ss.model = :model
    """
    params = {"model": model}
    if ticker and ticker != "MARKET":
        query += " AND na.ticker = :ticker"
        params["ticker"] = ticker
    query += " GROUP BY ss.label"
    df = pd.read_sql(text(query), db.engine, params=params)
    if df.empty:
        return go.Figure()
    color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
    fig = px.bar(df, x="label", y="count", color="label",
                 color_discrete_map=color_map,
                 title=f"{model} Sentiment Distribution")
    fig.update_layout(showlegend=False, height=300)
    return fig


def plot_correlation_scatter(df, ticker, model):
    valid = df[["avg_compound", "daily_return"]].dropna()
    if len(valid) < 3:
        return go.Figure()
    fig = px.scatter(
        valid, x="avg_compound", y="daily_return",
        title=f"{ticker} - {model}: Sentiment vs Daily Return",
        labels={"avg_compound": "Sentiment", "daily_return": "Return (%)"},
        trendline="ols", color_discrete_sequence=["#1f77b4"],
    )
    corr = valid.corr().iloc[0, 1]
    fig.add_annotation(text=f"r = {corr:.3f}", xref="paper", yref="paper",
                       x=0.05, y=0.95, showarrow=False,
                       bgcolor="white", bordercolor="gray", borderwidth=1)
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig


def load_evaluation_csv(filename):
    path = PROCESSED_DIR / "evaluation" / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def plot_lagged_correlation():
    df = load_evaluation_csv("lagged_correlation.csv")
    if df.empty:
        st.info("Run comprehensive_evaluation.py first to generate lagged correlation data.")
        return
    avg = df.groupby(["model", "lag"])["pearson_r"].mean().reset_index()
    fig = go.Figure()
    for model in ["VADER", "FinBERT"]:
        mdf = avg[avg["model"] == model]
        color = "#3498db" if model == "VADER" else "#9b59b6"
        fig.add_trace(go.Bar(x=mdf["lag"], y=mdf["pearson_r"], name=model,
                             marker_color=color, opacity=0.85))
    fig.update_layout(
        title="Lagged Correlation: Sentiment vs Returns (Avg Across Tickers)",
        xaxis_title="Lag (Days)", yaxis_title="Pearson r",
        barmode="group", height=400,
    )
    st.plotly_chart(fig, key="chart_1"  , use_container_width=True)

    # Detail table
    st.dataframe(df.groupby(["model", "lag"]).agg(
        avg_r=("pearson_r", "mean"),
        significant=("significant_005", "sum"),
        total=("pearson_r", "count"),
    ).reset_index(), use_container_width=True)


def plot_car_event_study():
    df = load_evaluation_csv("car_event_study.csv")
    if df.empty:
        st.info("Run comprehensive_evaluation.py first to generate CAR data.")
        return
    fig = go.Figure()
    for _, row in df.iterrows():
        color = "#2ecc71" if row["event_type"] == "positive" else "#e74c3c"
        label = f"{row['model']} {row['ticker']} ({row['event_type']})"
        fig.add_trace(go.Bar(
            x=[label], y=[row["car_mean"]],
            error_y=dict(type="data", array=[row["car_std"]]),
            marker_color=color, name=label, showlegend=False,
        ))
    fig.update_layout(
        title="Cumulative Abnormal Returns After Extreme Sentiment (3-Day Window)",
        xaxis_title="Model / Ticker / Event", yaxis_title="CAR (%)",
        height=400,
    )
    st.plotly_chart(fig, key="chart_2"  , use_container_width=True)
    st.dataframe(df[["model", "ticker", "event_type", "n_events",
                      "car_mean", "car_std", "t_stat", "p_value", "significant_005"]],
                 use_container_width=True)


def plot_information_coefficient():
    df = load_evaluation_csv("information_coefficient.csv")
    if df.empty:
        st.info("Run comprehensive_evaluation.py first to generate IC data.")
        return
    # Exclude AGGREGATE rows for chart
    chart_df = df[df["ticker"] != "AGGREGATE"]
    avg = chart_df.groupby(["model", "lag"])["IC"].mean().reset_index()
    fig = go.Figure()
    for model in ["VADER", "FinBERT"]:
        mdf = avg[avg["model"] == model]
        color = "#3498db" if model == "VADER" else "#9b59b6"
        fig.add_trace(go.Scatter(
            x=mdf["lag"], y=mdf["IC"], name=model,
            mode="lines+markers", line=dict(color=color, width=3),
            marker=dict(size=10),
        ))
    fig.add_hline(y=0.05, line_dash="dash", line_color="gray",
                  annotation_text="IC = 0.05 (meaningful threshold)")
    fig.update_layout(
        title="Information Coefficient by Lag (Avg Across Tickers)",
        xaxis_title="Lag (Days)", yaxis_title="IC (Spearman)",
        height=400,
    )
    st.plotly_chart(fig, key="chart_3"  , use_container_width=True)


def plot_walk_forward():
    df = load_evaluation_csv("walk_forward_validation.csv")
    if df.empty:
        st.info("Run comprehensive_evaluation.py first to generate walk-forward data.")
        return
    # Summary table
    summary = []
    for model in ["VADER", "FinBERT"]:
        mdf = df[df["model"] == model]
        train = mdf[mdf["set"] == "train"]["correlation"].dropna()
        test = mdf[mdf["set"] == "test"]["correlation"].dropna()
        if len(train) > 0 and len(test) > 0:
            summary.append({
                "Model": model,
                "Avg Train r": round(train.mean(), 4),
                "Avg Test r": round(test.mean(), 4),
                "Overfitting Gap": round(train.mean() - test.mean(), 4),
                "Interpretation": "Minimal overfitting" if abs(train.mean() - test.mean()) < 0.1 else "Potential overfitting",
            })
    if summary:
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

    # Train vs Test comparison chart
    fig = go.Figure()
    for model in ["VADER", "FinBERT"]:
        for s, color in [("train", "#3498db"), ("test", "#e74c3c")]:
            sdf = df[(df["model"] == model) & (df["set"] == s)]
            if not sdf.empty:
                fig.add_trace(go.Bar(
                    x=[f"{model} ({s})"], y=[sdf["correlation"].dropna().mean()],
                    marker_color=color, name=f"{model} - {s}",
                ))
    fig.update_layout(
        title="Walk-Forward Validation: Train vs Test Correlation",
        yaxis_title="Average Pearson r", height=350, barmode="group",
    )
    st.plotly_chart(fig, key="chart_4"  , use_container_width=True)


def plot_confidence_analysis():
    df = load_evaluation_csv("confidence_analysis.csv")
    if df.empty:
        st.info("Run comprehensive_evaluation.py first to generate confidence data.")
        return
    fig = go.Figure()
    for _, row in df.iterrows():
        label = f"{row['model']} {row['ticker']}"
        fig.add_trace(go.Bar(
            x=[label], y=[row["corr_all"]], marker_color="#3498db",
            name=f"{label} (all)", showlegend=False,
        ))
        fig.add_trace(go.Bar(
            x=[label], y=[row["corr_high_conf"]], marker_color="#9b59b6",
            name=f"{label} (high conf)", showlegend=False,
        ))
    fig.update_layout(
        title="Correlation: All Predictions vs High-Confidence Only (p >= 0.6)",
        yaxis_title="Pearson r", height=400,
    )
    st.plotly_chart(fig, key="chart_5"  , use_container_width=True)
    st.dataframe(df[["model", "ticker", "corr_all", "corr_high_conf",
                      "improvement", "pct_high_confidence"]],
                 use_container_width=True)


def show_data_quality():
    path = PROCESSED_DIR / "evaluation" / "data_quality_report.json"
    if not path.exists():
        st.info("Run comprehensive_evaluation.py first to generate quality report.")
        return
    with open(path) as f:
        q = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("News Articles", q["news_articles"]["total"])
    col2.metric("Market Records", q["market_data"]["total"])
    col3.metric("Sentiment Scores", q["sentiment_scores"]["total"])
    col4.metric("Alignment Rate", f"{q['alignment']['rate_pct']}%")

    with st.expander("Detailed Coverage"):
        st.json(q)


# ─── Main ─────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">FinSent-Market Analysis</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Controls")
    tickers = db.get_stored_tickers()
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers, index=0 if tickers else 0)
    model_options = ["VADER", "FinBERT", "Both"]
    selected_model = st.sidebar.radio("Sentiment Model", model_options, index=2)

    all_dates = pd.date_range(start="2026-01-01", end=datetime.now()).strftime("%Y-%m-%d")
    min_date, max_date = st.sidebar.select_slider(
        "Date Range", options=list(all_dates), value=(all_dates[0], all_dates[-1]))

    # Load data
    with st.spinner("Loading data..."):
        market_df = db.get_market_data(selected_ticker)

        def load_sent(ticker, model):
            df = db.get_sentiment_with_market(ticker, model)
            if df.empty:
                df = db.get_daily_sentiment(ticker, model)
                df["daily_return"] = 0.0
                df["close"] = 0.0
            return df

        if selected_model == "VADER":
            vader_df = load_sent(selected_ticker, "VADER")
            finbert_df = pd.DataFrame()
        elif selected_model == "FinBERT":
            finbert_df = load_sent(selected_ticker, "FinBERT")
            vader_df = pd.DataFrame()
        else:
            vader_df = load_sent(selected_ticker, "VADER")
            finbert_df = load_sent(selected_ticker, "FinBERT")

    # ─── Overview ─────────────────────────────────────────────
    st.markdown(gradient_heading("Overview"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Articles", f"{db.get_table_count('news_articles'):,}")
    c2.metric("Sentiment Analyses", f"{db.get_table_count('sentiment_scores'):,}")
    active_df = vader_df if not vader_df.empty else finbert_df
    if not active_df.empty:
        avg = round(active_df["avg_compound"].mean(), 3)
        c3.metric("Avg Sentiment", f"{avg:+.3f}")
    else:
        c3.metric("Avg Sentiment", "N/A")
    if not market_df.empty:
        valid = market_df.dropna(subset=["close"])
        if not valid.empty and valid["close"].iloc[0] > 0:
            ret = round((valid["close"].iloc[-1] / valid["close"].iloc[0] - 1) * 100, 2)
            c4.metric(f"{selected_ticker} Total Return", f"{ret:+.2f}%")
        else:
            c4.metric("Total Return", "N/A")
    else:
        c4.metric("Total Return", "N/A")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ─── Model Comparison ─────────────────────────────────────
    if selected_model == "Both" and not vader_df.empty and not finbert_df.empty:
        st.markdown(gradient_heading("Model Comparison"), unsafe_allow_html=True)
        v = vader_df.sort_values("date")
        f = finbert_df.sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v["date"], y=v["avg_compound"], name="VADER",
                                 line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(x=f["date"], y=f["avg_compound"], name="FinBERT",
                                 line=dict(color="#9b59b6", width=2)))
        fig.update_layout(title=f"{selected_ticker} - VADER vs FinBERT Sentiment",
                          xaxis_title="Date", yaxis_title="Score", hovermode="x unified", height=400)
        st.plotly_chart(fig, key="chart_6"  , use_container_width=True)

        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_sentiment_distribution(selected_ticker, "VADER"), key="plot_1", use_container_width=True)
        c2.plotly_chart(plot_sentiment_distribution(selected_ticker, "FinBERT"), key="plot_2", use_container_width=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ─── Sentiment Analysis ───────────────────────────────────
    st.markdown(gradient_heading("Sentiment Analysis"), unsafe_allow_html=True)

    for model, df_sent in [("VADER", vader_df), ("FinBERT", finbert_df)]:
        if df_sent.empty:
            continue
        filt = df_sent[(df_sent["date"] >= min_date) & (df_sent["date"] <= max_date)]
        if filt.empty:
            continue
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(plot_sentiment_time_series(filt, selected_ticker, model), key=f"plot_ts_{model}", use_container_width=True)
        c2.plotly_chart(plot_sentiment_distribution(selected_ticker, model), key=f"plot_dist_{model}", use_container_width=True)
        st.plotly_chart(plot_correlation_scatter(filt, selected_ticker, model), key=f"plot_corr_{model}", use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ─── Advanced Evaluation ──────────────────────────────────
    st.markdown(gradient_heading("Advanced Evaluation"), unsafe_allow_html=True)

    eval_tabs = st.tabs(["Lagged Correlation", "CAR Event Study", "Information Coefficient",
                         "Walk-Forward", "Confidence Filtering", "Data Quality"])

    with eval_tabs[0]:
        st.caption("Pearson correlation between sentiment and returns at lags 0-3 days.")
        plot_lagged_correlation()

    with eval_tabs[1]:
        st.caption("Cumulative Abnormal Returns following extreme sentiment signals.")
        plot_car_event_study()

    with eval_tabs[2]:
        st.caption("Information Coefficient (rank correlation) - IC > 0.05 is meaningful.")
        plot_information_coefficient()

    with eval_tabs[3]:
        st.caption("Walk-forward validation: train on earlier data, test on later data.")
        plot_walk_forward()

    with eval_tabs[4]:
        st.caption("Effect of filtering low-confidence predictions (max prob >= 0.6).")
        plot_confidence_analysis()

    with eval_tabs[5]:
        st.caption("Automated data quality assessment.")
        show_data_quality()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ─── Market Data ──────────────────────────────────────────
    if not market_df.empty:
        st.markdown(gradient_heading("Market Data"), unsafe_allow_html=True)
        mf = market_df[(market_df["date"] >= min_date) & (market_df["date"] <= max_date)]

        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=mf["date"], open=mf["open"], high=mf["high"],
            low=mf["low"], close=mf["close"], name="Price"))
        fig_price.update_layout(title=f"{selected_ticker} Price Chart",
                                xaxis_title="Date", yaxis_title="USD", height=450)
        st.plotly_chart(fig_price, key="chart_8"  , use_container_width=True)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=mf["date"], y=mf["volume"],
                                 name="Volume", marker_color="#2ecc71"))
        fig_vol.update_layout(title=f"{selected_ticker} Trading Volume",
                              xaxis_title="Date", yaxis_title="Volume", height=350)
        st.plotly_chart(fig_vol, key="chart_9"  , use_container_width=True)

    # ─── Recent News ──────────────────────────────────────────
    st.markdown(gradient_heading("Recent News"), unsafe_allow_html=True)
    news_df = db.get_news(selected_ticker)
    if not news_df.empty:
        recent = news_df.sort_values("published_at", ascending=False).head(10)
        for _, a in recent.iterrows():
            with st.expander(f"{a['published_at']} - {a['headline'][:80]}..."):
                st.markdown(f"""
                **Source:** {a['source']}  
                **Date:** {a['published_at']}  

                {a['summary'] or a['headline']}

                [Link]({a.get('url', '')})
                """)
    else:
        st.info("No news articles found.")

    # Footer
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; font-size: 0.9rem;">FinSent-Market Analysis System | Minxuan Wang</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure the database exists by running 'python run_pipeline.py' first.")

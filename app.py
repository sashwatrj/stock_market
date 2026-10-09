import streamlit as st
import pandas as pd


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Multi-Agent AI Research Desk",
    page_icon="📈",
    layout="wide"
)


# ==============================
# TITLE
# ==============================

st.title("📈 Multi-Agent AI Research Desk")

st.write(
    "Explainable stock market direction prediction "
    "using multiple specialized agents."
)


# ==============================
# LOAD RESEARCH REPORT
# ==============================

try:

    report = pd.read_csv(
        "data/processed/research_lead_report.csv"
    )

except FileNotFoundError:

    st.error(
        "Research report not found. "
        "Please run the Research Lead Agent first."
    )

    st.stop()


# ==============================
# MARKET REGIME
# ==============================

st.subheader("🌍 Market Regime")

if "Market_Regime" in report.columns:

    regime = report["Market_Regime"].iloc[0]

    st.metric(
        "Current Market Regime",
        regime
    )


# ==============================
# STOCK SELECTION
# ==============================

st.subheader("🔎 Stock Research")

ticker = st.selectbox(
    "Select a stock",
    sorted(report["Ticker"].unique())
)

stock = report[
    report["Ticker"] == ticker
].iloc[0]


# ==============================
# MAIN INFORMATION
# ==============================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Final Direction",
        stock["Final_Direction"]
    )


with col2:

    st.metric(
        "UP Probability",
        f"{stock['UP_Probability']:.2%}"
    )


with col3:

    st.metric(
        "Technical Signal",
        stock["Technical_Direction"]
    )


with col4:

    st.metric(
        "Risk Level",
        stock["Risk_Level"]
    )


# ==============================
# RESEARCH SCORE
# ==============================

st.subheader("🧠 Research Score")

score = float(
    stock["Research_Score"]
)

st.progress(
    min(
        max(
            (score + 1) / 2,
            0
        ),
        1
    )
)

st.write(
    f"Research Score: **{score:.4f}**"
)


# ==============================
# AGENT EVIDENCE
# ==============================

st.subheader("🤖 Agent Evidence")

col1, col2 = st.columns(2)


with col1:

    st.write("### Technical Agent")

    st.write(
        "Direction:",
        stock["Technical_Direction"]
    )

    st.write(
        "Technical Score:",
        stock["Technical_Score"]
    )


with col2:

    st.write("### News Agent")

    st.write(
        "Direction:",
        stock["News_Direction"]
    )

    st.write(
        "News Sentiment:",
        f"{stock['News_Sentiment']:.4f}"
    )


# ==============================
# RISK
# ==============================

st.subheader("⚠️ Risk Analysis")

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Risk Level",
        stock["Risk_Level"]
    )


with col2:

    st.metric(
        "Maximum Drawdown",
        f"{stock['Max_Drawdown']:.2%}"
    )


# ==============================
# ALL STOCKS
# ==============================

st.subheader("📊 All Stocks")

display_columns = [
    "Ticker",
    "UP_Probability",
    "Technical_Direction",
    "News_Direction",
    "Market_Regime",
    "Risk_Level",
    "Research_Score",
    "Final_Direction"
]

available_columns = [
    column
    for column in display_columns
    if column in report.columns
]

st.dataframe(
    report[available_columns],
    use_container_width=True
)
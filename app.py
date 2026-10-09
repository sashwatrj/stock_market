import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="MarketLens AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_FILE = DATA_DIR / "research_lead_report.csv"

# ---------------------- Visual design ----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #eef2ff; }
.block-container { max-width: 1450px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
[data-testid="stSidebar"] { background: #10182b; border-right: 1px solid #26334c; }
[data-testid="stSidebar"] * { color: #e7edff; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f4f7ff; }
p, label, li { color: #c1cbe0; }
div[data-testid="stMetric"] {
  background: #141e33; border: 1px solid #293754; border-radius: 16px;
  padding: 1rem 1.1rem;
}
div[data-testid="stMetricLabel"] { color: #9eacc8; }
div[data-testid="stMetricValue"] { color: #f4f7ff; }
div[data-testid="stMetricDelta"] { color: #8be0bc; }
.stSelectbox div[data-baseweb="select"] > div {
  background: #141e33; border-color: #354462; color: #f4f7ff;
}
div[data-testid="stDataFrame"] { border: 1px solid #293754; border-radius: 12px; overflow: hidden; }
hr { border-color: #27344e; }
.eyebrow { color: #7dd3fc; font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.hero {
  background: linear-gradient(120deg, #172844 0%, #18203b 60%, #252047 100%);
  border: 1px solid #354568; border-radius: 22px; padding: 1.5rem 1.7rem; margin-bottom: 1.2rem;
}
.hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; line-height: 1.12; color: #f7f9ff; }
.hero-sub { color: #bac8e4; margin-top: .65rem; max-width: 800px; }
.panel {
  background: #121b2e; border: 1px solid #293754; border-radius: 17px;
  padding: 1.1rem 1.2rem; height: 100%;
}
.panel-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #f3f6ff; font-size: 1.05rem; margin-bottom: .45rem; }
.panel-copy { color: #aab8d2; font-size: .9rem; line-height: 1.5; }
.pill {
  display: inline-block; padding: .28rem .65rem; border-radius: 999px; font-size: .76rem;
  font-weight: 700; background: #1e304b; color: #a5e6ff; border: 1px solid #34516d;
}
.signal-up { color: #79e0b5; }
.signal-down { color: #ff9aab; }
.signal-neutral { color: #f6cf7a; }
.big-signal { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; }
.small-note { color: #8e9db9; font-size: .8rem; }
.stButton button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Helpers ----------------------
def safe_value(row, col, fallback="Not available"):
    if col not in row.index or pd.isna(row[col]):
        return fallback
    return row[col]

def probability_text(value):
    try:
        n = float(value)
        return f"{n:.1%}" if 0 <= n <= 1 else f"{n:.1f}%"
    except (TypeError, ValueError):
        return "—"

def number_text(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"

def direction_class(value):
    v = str(value).upper()
    if v in ("UP", "BULLISH", "BUY"):
        return "signal-up"
    if v in ("DOWN", "BEARISH", "SELL"):
        return "signal-down"
    return "signal-neutral"

def plain_direction(value):
    v = str(value).upper()
    if v in ("UP", "BULLISH", "BUY"):
        return "The model leans upward"
    if v in ("DOWN", "BEARISH", "SELL"):
        return "The model leans downward"
    if v in ("NEUTRAL", "SIDEWAYS", "HOLD"):
        return "No strong direction"
    return "Direction not available"

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">MARKET INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown("## ◈ MarketLens AI")
    st.caption("Explainable stock research")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Explore a Stock", "How the AI Works", "Model Insights"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Prediction horizon**")
    st.markdown('<span class="pill">Next 5 trading days</span>', unsafe_allow_html=True)
    st.write("")
    st.caption("Research prototype · Not financial advice")

# ---------------------- Header ----------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-POWERED MARKET RESEARCH</div>
  <div class="hero-title">See the signal.<br>Understand the story.</div>
  <div class="hero-sub">MarketLens brings together price patterns, financial headlines, market conditions and risk into one clear research view.</div>
</div>
""", unsafe_allow_html=True)

if not REPORT_FILE.exists():
    st.error("Research report not found.")
    st.code("data/processed/research_lead_report.csv")
    st.info("Run the agent pipeline first, then refresh the page.")
    st.stop()

try:
    report = pd.read_csv(REPORT_FILE)
except Exception as e:
    st.error(f"Could not read the research report: {e}")
    st.stop()

if report.empty or "Ticker" not in report.columns:
    st.error("The report is empty or does not contain a Ticker column.")
    st.stop()

report["Ticker"] = report["Ticker"].astype(str)
report = report.dropna(subset=["Ticker"]).copy()
if report.empty:
    st.error("No usable stock tickers were found.")
    st.stop()

# ---------------------- Overview ----------------------
if page == "Overview":
    tickers = sorted(report["Ticker"].unique())
    total = len(tickers)
    up_count = report.get("Final_Direction", pd.Series(dtype=str)).astype(str).str.upper().isin(["UP", "BULLISH"]).sum()
    down_count = report.get("Final_Direction", pd.Series(dtype=str)).astype(str).str.upper().isin(["DOWN", "BEARISH"]).sum()
    risk_high = report.get("Risk_Level", pd.Series(dtype=str)).astype(str).str.upper().eq("HIGH").sum()

    st.markdown("### Market snapshot")
    st.caption("A quick overview of the stocks included in the latest research report.")
    a, b, c, d = st.columns(4)
    a.metric("Stocks analysed", total)
    b.metric("Upward signals", int(up_count))
    c.metric("Downward signals", int(down_count))
    d.metric("High-risk labels", int(risk_high))

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Stock signal board</div><div class="panel-copy">Compare the latest output from the research pipeline. These are model signals, not guarantees.</div></div>', unsafe_allow_html=True)
        cols = [c for c in ["Ticker", "Final_Direction", "UP_Probability", "Risk_Level", "Market_Regime"] if c in report.columns]
        st.dataframe(report[cols], width="stretch", hide_index=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">How to read this dashboard</div><div class="panel-copy">Each stock is reviewed from several angles. The model estimates direction, while other components add context about price patterns, news, the broader market and historical risk.</div><br><div class="pill">Direction</div><p class="panel-copy">What the current system leans toward.</p><div class="pill">Probability</div><p class="panel-copy">The model\'s estimated chance of an upward move.</p><div class="pill">Risk</div><p class="panel-copy">A simplified label based on historical behaviour.</p><div class="pill">Evidence</div><p class="panel-copy">Signals that help explain the research report.</p></div>', unsafe_allow_html=True)

    st.markdown("### Start exploring")
    x, y, z = st.columns(3)
    with x:
        st.markdown('<div class="panel"><div class="panel-title">01 · Explore a stock</div><div class="panel-copy">Choose a ticker and see its direction, probability, risk and agent findings.</div></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="panel"><div class="panel-title">02 · Understand the AI</div><div class="panel-copy">Learn what each analytical component does, without needing a technical background.</div></div>', unsafe_allow_html=True)
    with z:
        st.markdown('<div class="panel"><div class="panel-title">03 · Inspect model insights</div><div class="panel-copy">Review available evaluation metrics and feature-importance results.</div></div>', unsafe_allow_html=True)

# ---------------------- Stock detail ----------------------
elif page == "Explore a Stock":
    st.markdown("### Explore a stock")
    st.caption("Choose a stock to inspect its current report.")
    ticker = st.selectbox("Stock", sorted(report["Ticker"].unique()))
    row = report.loc[report["Ticker"] == ticker].iloc[0]

    direction = safe_value(row, "Final_Direction")
    prob = safe_value(row, "UP_Probability", None)
    risk = safe_value(row, "Risk_Level")
    regime = safe_value(row, "Market_Regime")
    css_class = direction_class(direction)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"""
        <div class="panel">
          <div class="eyebrow">CURRENT MODEL SIGNAL</div>
          <div class="big-signal {css_class}">{direction}</div>
          <div class="panel-copy">{plain_direction(direction)} for the next five trading days, based on the available model output.</div>
          <br><span class="pill">Estimated UP probability: {probability_text(prob)}</span>
          <p class="small-note">Probability is an estimate, not a promise or a measure of guaranteed profit.</p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">At a glance</div>', unsafe_allow_html=True)
        st.metric("Risk level", str(risk).title())
        st.metric("Broader market condition", str(regime).title())
        score = safe_value(row, "Research_Score", None)
        st.metric("Research score", number_text(score))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### What the research components found")
    st.caption("Each card explains one part of the analysis.")
    cards = [
        ("📈", "Price trend", "Technical_Direction", "Technical_Score", "Uses indicators such as RSI, MACD and moving averages to summarize recent price behaviour."),
        ("📰", "News sentiment", "News_Direction", "News_Sentiment", "Classifies available financial headlines as positive, negative or neutral. Data quality and timing matter."),
        ("🛡️", "Risk check", "Risk_Level", "Max_Drawdown", "Summarizes historical volatility and drawdown; it cannot predict all future losses."),
        ("🌐", "Market context", "Market_Regime", None, "Adds a broad-market label such as bullish, bearish or sideways."),
    ]
    cols = st.columns(2)
    for i, (icon, title, main_col, extra_col, desc) in enumerate(cards):
        with cols[i % 2]:
            main = safe_value(row, main_col)
            extra = safe_value(row, extra_col) if extra_col else None
            if extra_col == "News_Sentiment":
                extra = number_text(extra)
                extra_label = "Sentiment score"
            elif extra_col == "Max_Drawdown":
                try:
                    extra = f"{float(extra):.1%}"
                except (TypeError, ValueError):
                    extra = "—"
                extra_label = "Historical drawdown"
            else:
                extra_label = "Supporting score"
            st.markdown(f'<div class="panel"><div class="panel-title">{icon} {title}</div><div class="big-signal" style="font-size:1.3rem">{main}</div><div class="panel-copy">{desc}</div>', unsafe_allow_html=True)
            if extra_col:
                st.caption(f"{extra_label}: {extra}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### All available stocks")
    st.dataframe(report, width="stretch", hide_index=True)

# ---------------------- How it works ----------------------
elif page == "How the AI Works":
    st.markdown("### A research team, not a magic box")
    st.caption("The system splits the work into specialist components and combines their outputs.")
    stages = [
        ("01", "Prepare the data", "The Data Agent checks and organizes stock information."),
        ("02", "Read price behaviour", "The Technical Agent calculates indicators such as RSI, MACD and momentum."),
        ("03", "Read financial headlines", "The News Agent uses FinBERT to classify supplied headlines."),
        ("04", "Check market conditions", "The Market Regime Agent labels the broader market trend."),
        ("05", "Estimate direction", "The XGBoost Prediction Agent estimates the probability of an upward five-day return."),
        ("06", "Review risk", "The Risk Agent summarizes recent volatility and historical drawdown."),
        ("07", "Combine the evidence", "The Research Lead applies predefined rules to create the final report."),
        ("08", "Explain the model", "SHAP helps inspect which features influenced model output."),
    ]
    for num, title, desc in stages:
        l, r = st.columns([0.16, 1])
        with l:
            st.markdown(f'<div class="panel"><div class="big-signal" style="font-size:1.4rem;color:#7dd3fc">{num}</div></div>', unsafe_allow_html=True)
        with r:
            st.markdown(f'<div class="panel"><div class="panel-title">{title}</div><div class="panel-copy">{desc}</div></div>', unsafe_allow_html=True)
        st.write("")
    st.warning("Current limitation: agents perform specialized tasks in a structured workflow. They do not all communicate autonomously or negotiate with one another.")

    st.markdown("### Plain-English glossary")
    terms = {
        "UP probability": "The model's estimated probability that the five-day return is positive.",
        "RSI": "A measure that summarizes the strength of recent price moves.",
        "MACD": "A tool used to study trend and momentum changes.",
        "Volatility": "How much a price has tended to fluctuate.",
        "Drawdown": "A fall from a previous peak to a later low.",
        "SHAP": "A method for explaining which input features influenced a model's output.",
        "Walk-forward testing": "Training on earlier data and evaluating on later data.",
    }
    for term, desc in terms.items():
        with st.expander(term):
            st.write(desc)

# ---------------------- Model insights ----------------------
else:
    st.markdown("### Model insights")
    st.caption("Only available output files are shown. Missing files mean the relevant analysis has not been generated yet.")
    model_file = DATA_DIR / "model_comparison.csv"
    shap_file = DATA_DIR / "shap_importance.csv"
    backtest_file = DATA_DIR / "walk_forward_backtest.csv"

    if model_file.exists():
        st.markdown('<div class="panel"><div class="panel-title">Model comparison</div><div class="panel-copy">Compare the measured performance of the trained models.</div></div>', unsafe_allow_html=True)
        model_data = pd.read_csv(model_file)
        st.dataframe(model_data, width="stretch", hide_index=True)
    else:
        st.info("Model comparison results are not available yet. Run `python train_models.py`.")

    if shap_file.exists():
        st.markdown("### Which features mattered most to the model?")
        shap_data = pd.read_csv(shap_file)
        st.dataframe(shap_data, width="stretch", hide_index=True)
        name_col = next((c for c in shap_data.columns if c.lower() in {"feature", "features", "feature_name"}), None)
        value_col = next((c for c in shap_data.columns if "importance" in c.lower() or "shap" in c.lower()), None)
        if name_col and value_col:
            plot = shap_data[[name_col, value_col]].dropna().set_index(name_col)
            if not plot.empty:
                st.bar_chart(plot)
    else:
        st.info("SHAP results are not available yet. Run `python explain_model.py`.")

    if backtest_file.exists():
        st.markdown("### Walk-forward evaluation")
        backtest = pd.read_csv(backtest_file)
        st.dataframe(backtest.head(100), width="stretch", hide_index=True)
        st.caption("Showing up to 100 rows.")
    else:
        st.info("Walk-forward evaluation output is not available yet.")

    st.warning("Historical test results do not guarantee future performance. Evaluate the complete multi-agent workflow separately from the standalone model.")

st.divider()
st.markdown('<div class="small-note">MarketLens AI · Academic research prototype · Not financial advice. Market predictions are uncertain.</div>', unsafe_allow_html=True)

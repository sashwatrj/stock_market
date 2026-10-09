# stock_market

# 📈 Multi-Agent AI Research Desk for Explainable Stock Market Direction Prediction

An AI-powered stock market research system that combines Machine Learning, financial sentiment analysis, technical indicators, market regime detection, and risk assessment to predict stock market direction over the next five trading days.

The project uses specialized analytical agents to generate a consolidated research report with prediction probabilities, supporting evidence, risk information, and model explanations.

---

## 🎯 Project Overview

Stock market movements are influenced by multiple factors, including historical price movements, technical indicators, financial news, market conditions, and risk.

Traditional prediction approaches may focus on a single model or a limited set of indicators. This project explores a modular, multi-agent approach in which different components analyze different aspects of the market.

The system combines their outputs to produce an explainable stock research report.

### Key Objectives

- Predict stock price direction over a five-trading-day horizon.
- Analyze historical stock market data using technical indicators.
- Classify financial news sentiment using FinBERT.
- Estimate directional probabilities using XGBoost.
- Assess risk using volatility and historical drawdown.
- Identify broad market regimes.
- Explain model behavior using SHAP.
- Evaluate predictive performance using chronological testing and walk-forward validation.
- Present research outputs through an interactive Streamlit dashboard.

---

## ✨ Key Features

- **Multi-Agent Architecture:** Separates market analysis into specialized components.
- **Machine Learning Prediction:** Uses XGBoost to estimate the probability of upward price movement.
- **Technical Analysis:** Computes RSI, MACD, moving averages, momentum, volatility, and Bollinger Band features.
- **Financial Sentiment Analysis:** Uses FinBERT to classify financial news headlines.
- **Risk Assessment:** Evaluates recent volatility and historical maximum drawdown.
- **Market Regime Detection:** Classifies broad market conditions as bullish, bearish, or sideways.
- **Explainable AI:** Uses SHAP to investigate feature contributions to model predictions.
- **Historical Evaluation:** Includes chronological train/test splitting and a walk-forward backtesting prototype.
- **Interactive Dashboard:** Uses Streamlit to present stock research results.

---

## 🏗️ System Architecture

```text
                 Historical Stock Market Data
                            |
                            v
                      Data Agent
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
      Technical Agent   News Agent   Market Regime Agent
             |              |              |
             v              v              |
      Technical Signals  Sentiment           |
             |              |              |
             +--------------+--------------+
                            |
                            v
                    Prediction Agent
                         XGBoost
                            |
                            v
                       Risk Agent
                            |
                            v
                    Research Lead Agent
                            |
                            v
                  Consolidated Research Report
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
        Direction       Probability       Risk
             |
             v
      Explainable Research Dashboard

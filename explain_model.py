import pandas as pd
import shap

from xgboost import XGBClassifier


# ==============================
# LOAD DATA
# ==============================

data = pd.read_parquet(
    "data/processed/features_technical.parquet"
)

features = [
    "return_1d",
    "return_5d",
    "RSI",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "BB_width",
    "volatility_20d",
    "volume_change",
    "momentum_10d"
]

data = data.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

data = data.dropna(
    subset=features + ["target"]
)

data = data.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)


# ==============================
# TRAIN / TEST SPLIT
# ==============================

unique_dates = sorted(data["Date"].unique())

split_index = int(len(unique_dates) * 0.8)

train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]

train = data[data["Date"].isin(train_dates)]
test = data[data["Date"].isin(test_dates)]

X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]


# ==============================
# TRAIN XGBOOST
# ==============================

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("XGBoost model trained.")


# ==============================
# SHAP
# ==============================

explainer = shap.TreeExplainer(model)

# Use a sample so SHAP runs faster
X_sample = X_test.sample(
    min(500, len(X_test)),
    random_state=42
)

shap_values = explainer.shap_values(X_sample)


# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\nSHAP FEATURE IMPORTANCE")
print("=" * 50)
print(importance.to_string(index=False))

importance.to_csv(
    "data/processed/shap_importance.csv",
    index=False
)


# ==============================
# SHAP SUMMARY PLOT
# ==============================

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

import matplotlib.pyplot as plt

plt.tight_layout()

plt.savefig(
    "data/processed/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP results saved.")
print("data/processed/shap_importance.csv")
print("data/processed/shap_summary.png")

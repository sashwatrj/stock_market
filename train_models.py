import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==============================
# LOAD DATA
# ==============================

data = pd.read_parquet("data/processed/features_technical.parquet")

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

# Remove infinity values
data = data.replace([float("inf"), float("-inf")], float("nan"))

# Remove rows with missing values
data = data.dropna(subset=features + ["target"])

# Sort chronologically
data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

print("Total rows:", len(data))
print("Stocks:", data["Ticker"].nunique())

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

print("\nTRAINING ROWS:", len(train))
print("TEST ROWS:", len(test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True))


# ==============================
# MODELS
# ==============================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
}


# ==============================
# TRAIN + EVALUATE
# ==============================

results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })


# ==============================
# FINAL COMPARISON
# ==============================

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

results_df.to_csv(
    "data/processed/model_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/processed/model_results.csv")
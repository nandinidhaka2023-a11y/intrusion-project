"""
baseline_model.py

WHAT THIS FILE DOES:
Trains two simple, fast models (Random Forest and XGBoost) on the cleaned
flow data to predict is_attack (1/0). This gives us a benchmark score
BEFORE we build the more complex LSTM model.

WHY A BASELINE MATTERS (explain this in your report/demo):
If our "fancy" deep learning model doesn't beat this simple baseline, that's
a signal something is wrong with the DL approach — not that DL is bad. Every
serious ML project needs this comparison, and it also gives you an easy,
understandable result to show your teacher even before the LSTM is done.

WHY RANDOM FOREST / XGBOOST SPECIFICALLY:
Both are "tree-based" models that work well on tabular data like ours
(rows of numbers, not images/text) and train in seconds to minutes, not
hours. They also tell us which features matter most (feature importance),
which is useful for understanding the data before building the LSTM.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,  # this computes PR-AUC
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# These columns are NOT features — they're labels or IDs we simulated.
# We must drop them before training, or the model would "cheat" by reading
# the IP address instead of learning real traffic patterns.
NON_FEATURE_COLS = ["Label", "attack_type", "is_attack", "src_ip", "dst_ip", "source_file"]


def load_features_and_labels(path: Path = None):
    """
    Loads the cleaned data and splits it into X (features) and y (label).

    WHY WE DROP src_ip/dst_ip: even though we added them ourselves, if we
    left them in as features, the model could learn "src_ip == this exact
    attacker IP -> attack" — which is not actually learning to detect
    attacks, just memorizing addresses. A real intrusion detector must work
    on IPs it has never seen before, so IP is metadata for the dashboard,
    NOT a model input.
    """
    if path is None:
        path = PROCESSED_DATA_DIR / "cleaned_flows.parquet"
    df = pd.read_parquet(path)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["is_attack"]

    print(f"Loaded {len(df):,} rows, {len(feature_cols)} features")
    print(f"Class balance: {y.value_counts(normalize=True).to_dict()}")

    return X, y, feature_cols


def train_and_evaluate(X, y, feature_cols):
    """
    Splits data, trains Random Forest, evaluates with PR-AUC (not plain
    accuracy — see WHY below), and prints a full report.

    WHY PR-AUC INSTEAD OF ACCURACY:
    Our data is ~98% benign / ~2% attack (from Step 2's output). A model
    that just predicts "benign" for everything would already be 98%
    "accurate" while being completely useless. PR-AUC (Precision-Recall
    Area Under Curve) specifically measures how well the model finds the
    rare attack class, which is what we actually care about.

    NOTE ON THE SPLIT: normally we'd split by timestamp to avoid leakage
    (train on earlier flows, test on later ones). This dataset release has
    no timestamp column (it was part of the stripped "metadata"), so we use
    a stratified random split instead and note this as a limitation in the
    report. dhoogla's release is also documented as pre-cleaned/deduplicated,
    which reduces (but doesn't eliminate) the near-duplicate-leakage risk
    CICIDS2017 is known for.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # class_weight='balanced' tells the model to pay more attention to the
    # rare attack class instead of ignoring it — our first defense against
    # the class imbalance problem
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of "attack"

    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== RESULTS ===")
    print(f"PR-AUC: {pr_auc:.4f}  (closer to 1.0 is better; this is our main metric)")
    print(f"F1 score: {f1:.4f}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance: which columns the model relied on most
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 10 most important features:")
    print(importances.sort_values(ascending=False).head(10))

    return model, {"pr_auc": pr_auc, "f1": f1}


def save_model(model, name="baseline_random_forest.joblib"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / name
    joblib.dump(model, out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    X, y, feature_cols = load_features_and_labels()
    model, metrics = train_and_evaluate(X, y, feature_cols)
    save_model(model)

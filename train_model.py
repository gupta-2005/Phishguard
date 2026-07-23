"""
train_model.py
---------------
Trains a phishing-URL classifier on data/urls_dataset.csv and saves the
fitted model + a metrics report to the model/ directory.

Usage:
    python3 train_model.py
"""

import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)

from feature_extraction import FEATURE_COLUMNS

DATA_PATH = "data/urls_dataset.csv"
MODEL_PATH = "model/phishing_model.joblib"
METRICS_PATH = "model/metrics.json"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": dict(
            sorted(
                zip(FEATURE_COLUMNS, clf.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"] = cv_scores.std()

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))
    print("=== Metrics ===")
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "feature_importances"):
            print(f"{k}: {v:.4f}")
    print("Confusion matrix (rows=true, cols=pred) [Safe, Phishing]:")
    print(metrics["confusion_matrix"])
    print("\nTop 5 most important features:")
    for feat, imp in list(metrics["feature_importances"].items())[:5]:
        print(f"  {feat}: {imp:.4f}")

    joblib.dump(clf, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
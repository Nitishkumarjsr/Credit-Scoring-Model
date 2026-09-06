"""
CodeAlpha Task 1: Credit Scoring Model
Predict creditworthiness and loan default risk from financial & bureau features.
Standalone Model Training, Benchmarking, and Evaluation Script.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, accuracy_score, f1_score
import joblib


def load_data(path):
    return pd.read_csv(path)


def generate_default_data():
    """Generates financial credit dataset if no external CSV is provided."""
    from ml_engine import ml_engine
    return ml_engine.df.copy()


def train_and_evaluate(df, target="loan_status"):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset. Columns: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=["number", "float64", "int64"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "float64", "int64"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42)
    }

    results = []
    best_name, best_pipe, best_auc = None, None, -1.0

    print("======================================================================")
    print("CodeAlpha Task 1: Credit Scoring Model Training & Evaluation")
    print("======================================================================")

    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else 0.0
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "ROC-AUC": round(auc, 4),
            "F1-Score": round(f1, 4)
        })

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"ROC-AUC Score: {round(auc, 4)} | Accuracy: {round(acc * 100, 2)}%")

        if auc > best_auc:
            best_name, best_pipe, best_auc = name, pipe, auc

    # Save model and metrics
    joblib.dump(best_pipe, "credit_scoring_model.joblib")
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)

    # Plot ROC Curve
    if len(np.unique(y_test)) == 2 and best_pipe is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(best_pipe, X_test, y_test, ax=ax, name=best_name)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.50)")
        plt.title(f"ROC Curve &mdash; Top Model: {best_name} (AUC = {round(best_auc, 4)})", fontsize=12, fontweight="bold")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=160)
        plt.close()

    print("\n======================================================================")
    print(f"Model Training Complete! Best Performing Classifier: {best_name} (ROC-AUC: {round(best_auc, 4)})")
    print("Saved Artifacts:")
    print("  -> credit_scoring_model.joblib")
    print("  -> model_results.csv")
    print("  -> roc_curve.png")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeAlpha Credit Scoring Model Trainer")
    parser.add_argument("--data", default=None, help="Path to CSV dataset (optional)")
    parser.add_argument("--target", default="loan_status", help="Target column name")
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        df = load_data(args.data)
    else:
        df = generate_default_data()

    train_and_evaluate(df, target=args.target)

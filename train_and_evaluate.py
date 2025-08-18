"""Orchestrate training and evaluation using the modular pipeline."""

from __future__ import annotations

import argparse
import pandas as pd
from sklearn.datasets import make_classification

from data_processing import load_data, preprocess
from models import (
    evaluate,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)


def main(train_path: str | None = None, test_path: str | None = None) -> None:
    if train_path:
        train_df, _ = load_data(train_path, test_path)
        X_train, X_val, y_train, y_val = preprocess(train_df)
    else:
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10, random_state=42
        )
        df = pd.DataFrame(X)
        df["TARGET"] = y
        X_train, X_val, y_train, y_val = preprocess(df)

    models = {
        "logistic_regression": train_logistic_regression(
            X_train, y_train, class_weight="balanced"
        ),
        "random_forest": train_random_forest(
            X_train, y_train, n_estimators=200, max_depth=10, class_weight="balanced"
        ),
        "xgboost": train_xgboost(X_train, y_train, n_estimators=200, learning_rate=0.05),
    }

    for name, model in models.items():
        auc = evaluate(model, X_val, y_val)
        print(f"{name}: AUC={auc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on Home Credit data")
    parser.add_argument("--train", help="Path to training CSV", default=None)
    parser.add_argument("--test", help="Path to test CSV", default=None)
    args = parser.parse_args()
    main(args.train, args.test)

"""Model training utilities for the Home Credit Default Risk project."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def train_logistic_regression(X, y, **kwargs) -> LogisticRegression:
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000, **kwargs)
    model.fit(X, y)
    return model


def train_random_forest(X, y, **kwargs) -> RandomForestClassifier:
    """Train a random forest classifier."""
    model = RandomForestClassifier(**kwargs)
    model.fit(X, y)
    return model


def train_xgboost(X, y, **kwargs) -> XGBClassifier:
    """Train an XGBoost classifier."""
    model = XGBClassifier(eval_metric="auc", use_label_encoder=False, **kwargs)
    model.fit(X, y)
    return model


def evaluate(model: Any, X, y) -> float:
    """Compute the ROC AUC score for ``model``."""
    proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)

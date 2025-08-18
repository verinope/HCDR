"""Data loading and feature engineering utilities for the Home Credit Default Risk project."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(train_path: str, test_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load training and optional test datasets from CSV files."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None
    return train, test


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None):
    """Perform basic feature engineering.

    This function applies the key steps from the original notebook:
    * Replace the anomalous value 365243 in ``DAYS_EMPLOYED`` with the median
      employment length.
    * One-hot encode categorical variables using :func:`pandas.get_dummies`.
    * Scale features to ``[0, 1]`` with :class:`~sklearn.preprocessing.MinMaxScaler`.

    Parameters
    ----------
    train_df: pandas.DataFrame
        Training dataframe containing a ``TARGET`` column.
    test_df: pandas.DataFrame, optional
        Optional test dataframe with the same schema as ``train_df`` minus the
        ``TARGET`` column. When provided, the function returns processed train and
        test matrices along with the training labels. Otherwise it returns a
        train/validation split for model development.
    """
    for df in (train_df, test_df):
        if df is not None and 'DAYS_EMPLOYED' in df.columns:
            df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
            df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median(), inplace=True)

    y = train_df['TARGET']
    train_features = train_df.drop(columns=['TARGET'])

    if test_df is not None:
        combined = pd.concat([train_features, test_df], axis=0)
    else:
        combined = train_features
    combined = pd.get_dummies(combined)

    if test_df is not None:
        train_processed = combined.iloc[: len(train_features)]
        test_processed = combined.iloc[len(train_features) :]
    else:
        train_processed = combined
        test_processed = None

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_processed)
    test_scaled = scaler.transform(test_processed) if test_processed is not None else None

    if test_df is None:
        X_train, X_val, y_train, y_val = train_test_split(
            train_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_val, y_train, y_val
    return train_scaled, test_scaled, y

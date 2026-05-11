"""
Deployment helpers for the final saved model.

Provides a small sklearn-compatible transformer that selects a fixed subset
of columns from a pandas DataFrame. This lets the final pipeline accept
raw, full-column input (all 13 features) and internally restrict to the
stable feature subset chosen in Task 4.
"""

from __future__ import annotations
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSubsetSelector(BaseEstimator, TransformerMixin):
    """Select a fixed subset of columns from an input DataFrame.

    Designed as the first step of a deployment pipeline so that a saved
    model can accept raw 13-column input and internally restrict to the
    stable feature subset identified during Task 4.

    Parameters
    ----------
    features : List[str]
        The list of column names to retain.

    Notes
    -----
    This transformer is stateless — it does not learn anything during `fit`.
    The feature list is fixed at construction time and applied to every
    input via column-label indexing.
    """

    def __init__(self, features: List[str]):
        self.features = list(features)

    def fit(self, X, y=None):                            
        # Validate that the requested features exist
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.features if f not in X.columns]
            if missing:
                raise ValueError(
                    f'FeatureSubsetSelector: missing required columns {missing}. '
                    f'Got: {list(X.columns)}'
                )
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                'FeatureSubsetSelector expects a pandas DataFrame as input '
                '(so columns can be selected by name).'
            )
        return X[self.features].copy()

    def get_feature_names_out(self, input_features=None):
        return list(self.features)

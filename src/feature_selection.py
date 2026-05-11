"""
Feature selection utilities for the rnCV pipeline.

Implements model-agnostic permutation-importance-based feature selection,
computed strictly on training data inside each outer fold. Permutation
importance is well-suited to LDA (which lacks built-in feature importances)
and is recommended by the sklearn documentation as the default model-agnostic
method for any fitted estimator.

Method
------
For each candidate feature, the values are randomly shuffled and the drop in
a chosen metric (here: MCC, matching the inner-loop optimisation criterion)
is computed. Features causing larger drops are deemed more important.

Implementation notes
--------------------
- The estimator used to compute importances is a fresh LDA fitted on the
  outer training fold (with default HPs and the same preprocessing pipeline).
  We deliberately do not use the tuned LDA — feature selection happens BEFORE
  the inner loop, so tuned HPs are not yet known.
- `n_repeats=10` gives stable importance estimates without excessive runtime.
- Importances are computed on the outer train fold itself (no separate
  hold-out), which is acceptable here because the outer test fold is fully
  locked away.
"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection             import permutation_importance
from sklearn.metrics                import make_scorer, matthews_corrcoef
from sklearn.pipeline               import Pipeline

from .preprocessing import build_preprocessor


# Standard MCC scorer used throughout
MCC_SCORER = make_scorer(matthews_corrcoef)


def select_top_k_features(
    X_train:   pd.DataFrame,
    y_train:   np.ndarray,
    k:         int,
    seed:      int = 42,
    n_repeats: int = 10,
) -> List[str]:
    """Select the top-k features by permutation importance on the training fold.

    Parameters
    ----------
    X_train : pd.DataFrame
        Outer training fold features.
    y_train : np.ndarray
        Outer training fold labels.
    k : int
        Number of features to retain.
    seed : int
        Random state for permutation shuffling (reproducibility).
    n_repeats : int
        Number of times each feature is shuffled. 10 gives stable estimates;
        higher is more accurate but slower.

    Returns
    -------
    List[str]
        Names of the top-k features, ranked by mean importance (descending).

    Notes
    -----
    A fresh LDA with default hyperparameters is used as the importance
    estimator. The same ColumnTransformer preprocessing as the main pipeline
    is applied so importances reflect performance on the standardised
    feature space.
    """
    pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model',        LinearDiscriminantAnalysis()),
    ])
    pipe.fit(X_train, y_train)

    result = permutation_importance(
        pipe, X_train, y_train,
        scoring=MCC_SCORER, n_repeats=n_repeats,
        random_state=seed, n_jobs=1,
    )
    importances  = pd.Series(result.importances_mean, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(k).index.tolist()
    return top_features


def compute_feature_importances(
    X_train:   pd.DataFrame,
    y_train:   np.ndarray,
    seed:      int = 42,
    n_repeats: int = 10,
) -> pd.Series:
    """Return permutation importances for ALL features (no selection).

    Useful for the motivating figure that ranks all features by importance
    on a single split.
    """
    pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model',        LinearDiscriminantAnalysis()),
    ])
    pipe.fit(X_train, y_train)

    result = permutation_importance(
        pipe, X_train, y_train,
        scoring=MCC_SCORER, n_repeats=n_repeats,
        random_state=seed, n_jobs=1,
    )
    return pd.Series(result.importances_mean,
                     index=X_train.columns).sort_values(ascending=False)

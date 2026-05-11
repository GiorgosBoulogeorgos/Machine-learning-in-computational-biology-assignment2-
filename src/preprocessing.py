"""
Preprocessing module for the Heart Disease classification pipeline.

This module constructs the ColumnTransformer used inside every cross-validation
fold. Two transformations are applied:

    1. Mode imputation for the two missing values (in `ca` and `thal`).
    2. StandardScaler for the five continuous features.

Binary and categorical features are passed through unchanged. The transformer
is built fresh for each fold and fitted strictly on training data, so that
no information from validation/test folds leaks into the preprocessing step.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Feature groups (defined once; reused across folds) ────────────────────────
CONTINUOUS_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Binary + categorical features that need imputation but no scaling.
# `ca` and `thal` contain the dataset's two missing values.
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


def build_preprocessor(feature_subset: list[str] | None = None) -> ColumnTransformer:
    """Construct a fresh ColumnTransformer for one fold.

    Parameters
    ----------
    feature_subset : list of str, optional
        If provided, only these features will be processed. Continuous and
        categorical feature lists are intersected with this subset, so the
        transformer will only try to access columns that actually exist.
        This is needed when feature selection has reduced the feature set.

    Returns
    -------
    ColumnTransformer
        A transformer that applies:
          - mean imputation + StandardScaler to continuous features,
          - mode (most-frequent) imputation to categorical/binary features.

    Notes
    -----
    The transformer is unfitted. It must be fitted on the training fold only
    via `pipeline.fit(X_train, y_train)`, then applied to validation/test
    folds via `pipeline.transform(X_test)` to prevent data leakage.
    """
    # Restrict feature lists to the provided subset (if any)
    if feature_subset is not None:
        cont_feats = [f for f in CONTINUOUS_FEATURES  if f in feature_subset]
        cat_feats  = [f for f in CATEGORICAL_FEATURES if f in feature_subset]
    else:
        cont_feats = CONTINUOUS_FEATURES
        cat_feats  = CATEGORICAL_FEATURES

    # Continuous pipeline: impute then scale.
    # Mean imputation is used as a defensive default — none of the continuous
    # features have missing values in this dataset, but the pipeline must
    # remain robust if applied to new data with missing continuous entries.
    continuous_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
    ])

    # Categorical pipeline: mode imputation only (no scaling).
    # The two real missing values in the dataset live here (`ca`, `thal`).
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous',  continuous_pipeline,  cont_feats),
            ('categorical', categorical_pipeline, cat_feats),
        ],
        remainder='drop',          # any column not listed is dropped (defensive)
        verbose_feature_names_out=False,
    )
    # Preserve column names for downstream feature-importance work.
    preprocessor.set_output(transform='pandas')
    return preprocessor

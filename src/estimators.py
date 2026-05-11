"""
Estimator configurations for the rnCV pipeline.

Each estimator is paired with:
  - a factory function returning an unfitted sklearn-compatible estimator,
  - a callable that defines its hyperparameter search space for Optuna.

The hp spaces are intentionally generous (wide ranges, log-scales for
regularisation parameters) to give Optuna's TPE sampler enough room to
explore. With 50 trials per inner fold this is well within budget.

Justification of hp ranges (per algorithm) is provided in the docstrings
and in the report's Methods section.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List

from sklearn.linear_model     import LogisticRegression
from sklearn.naive_bayes      import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble         import RandomForestClassifier

# Optional gradient-boosting libraries
try:
    from lightgbm import LGBMClassifier
except ImportError:                                  
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:                                  
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:                                  
    CatBoostClassifier = None


# ── Container ─────────────────────────────────────────────────────────────────
@dataclass
class EstimatorConfig:
    """Pairs an estimator factory with its Optuna search-space callable.

    Parameters
    ----------
    name : str
        Human-readable identifier (used in result tables/plots).
    estimator_factory : Callable[..., estimator]
        Returns an unfitted sklearn-compatible estimator. Receives any
        hyperparameters as keyword arguments.
    hp_space : Callable[[optuna.Trial], Dict[str, Any]] | None
        Maps an Optuna trial to a hyperparameter dict. None disables tuning
        (useful for the default-hyperparameter baseline in Task 3.2).
    default_kwargs : Dict[str, Any]
        Fixed kwargs always passed to the factory (e.g. random_state).
    """
    name: str
    estimator_factory: Callable[..., Any]
    hp_space: Callable | None = None
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self, hp: Dict[str, Any] | None = None) -> Any:
        """Instantiate the estimator with the given hyperparameters."""
        kwargs = {**self.default_kwargs, **(hp or {})}
        return self.estimator_factory(**kwargs)


# ── Hyperparameter spaces ────────────────────────────────────────────────────
# Each callable takes an Optuna trial and returns a dict of suggested HPs.
# Ranges are justified with brief comments — these will be expanded in the
# report's Methods section.

def hp_logistic_regression(trial) -> Dict[str, Any]:
    """Elastic-Net Logistic Regression.

    `C` (inverse regularisation strength) spans 4 orders of magnitude on a
    log scale. `l1_ratio` covers the
    full L1↔L2 mixing range. `saga` is the only solver that supports
    elastic-net penalty.
    """
    return {
        'C':        trial.suggest_float('C', 1e-3, 1e1, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
    }


def hp_gaussian_nb(trial) -> Dict[str, Any]:
    """Gaussian NB. Only `var_smoothing` is meaningfully tunable."""
    return {
        'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True),
    }


def hp_lda(trial) -> Dict[str, Any]:
    """Linear Discriminant Analysis with optional shrinkage.

    Shrinkage is a regularised covariance estimate (Ledoit-Wolf style) — useful
    when n_features is large relative to n_samples per class, as is the case
    in this small dataset.
    """
    solver = trial.suggest_categorical('solver', ['lsqr', 'eigen'])
    shrinkage = trial.suggest_float('shrinkage', 0.0, 1.0)
    return {'solver': solver, 'shrinkage': shrinkage}


def hp_random_forest(trial) -> Dict[str, Any]:
    """Random Forest.

    Ranges chosen to balance variance reduction (more trees) with overfit
    risk (depth, min samples). 200–500 trees is a reasonable default for
    a dataset this small.
    """
    return {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth':         trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }


def hp_lightgbm(trial) -> Dict[str, Any]:
    """LightGBM. `num_leaves` and `min_child_samples` control tree complexity;
    `learning_rate` log-scaled is standard for boosters."""
    return {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500, step=50),
        'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 3e-1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 8, 64),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha':         trial.suggest_float('reg_alpha',  1e-3, 1.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
    }


def hp_xgboost(trial) -> Dict[str, Any]:
    """XGBoost — analogous space to LightGBM."""
    return {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500, step=50),
        'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 3e-1, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha',  1e-3, 1.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
    }


def hp_catboost(trial) -> Dict[str, Any]:
    """CatBoost — fewer parameters needed."""
    return {
        'iterations':    trial.suggest_int('iterations', 100, 500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-1, log=True),
        'depth':         trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    }


# ── Factory: build the full list of estimator configs ────────────────────────
def get_estimator_configs(seed: int = 42) -> List[EstimatorConfig]:
    """Return the standard list of seven estimator configs for the assignment.

    Parameters
    ----------
    seed : int
        Random state to pass to all stochastic estimators for reproducibility.
    """
    configs: List[EstimatorConfig] = [
        EstimatorConfig(
            name='LogisticRegression',
            estimator_factory=LogisticRegression,
            hp_space=hp_logistic_regression,
            default_kwargs={'penalty': 'elasticnet', 'solver': 'saga',
                            'l1_ratio': 0.5, 'max_iter': 5000, 'random_state': seed},
        ),
        EstimatorConfig(
            name='GaussianNB',
            estimator_factory=GaussianNB,
            hp_space=hp_gaussian_nb,
            default_kwargs={},
        ),
        EstimatorConfig(
            name='LDA',
            estimator_factory=LinearDiscriminantAnalysis,
            hp_space=hp_lda,
            default_kwargs={},
        ),
        EstimatorConfig(
            name='RandomForest',
            estimator_factory=RandomForestClassifier,
            hp_space=hp_random_forest,
            # n_jobs=-1 is safe here: each tree gets its own seed via
            # random_state, so parallelism doesn't introduce non-determinism.
            default_kwargs={'random_state': seed, 'n_jobs': -1},
        ),
    ]

    if LGBMClassifier is not None:
        configs.append(EstimatorConfig(
            name='LightGBM',
            estimator_factory=LGBMClassifier,
            hp_space=hp_lightgbm,
            default_kwargs={'random_state': seed, 'n_jobs': 1, 'verbose': -1},
        ))
    if XGBClassifier is not None:
        configs.append(EstimatorConfig(
            name='XGBoost',
            estimator_factory=XGBClassifier,
            hp_space=hp_xgboost,
            default_kwargs={'random_state': seed, 'n_jobs': 1,
                            'use_label_encoder': False, 'eval_metric': 'logloss',
                            'verbosity': 0},
        ))
    if CatBoostClassifier is not None:
        configs.append(EstimatorConfig(
            name='CatBoost',
            estimator_factory=CatBoostClassifier,
            hp_space=hp_catboost,
            default_kwargs={'random_state': seed, 'thread_count': 1,
                            'verbose': 0, 'allow_writing_files': False},
        ))
    return configs

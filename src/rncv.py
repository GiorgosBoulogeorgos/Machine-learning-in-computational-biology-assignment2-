"""
RepeatedNestedCV — the core OOP class for Assignment 2.

Implements R repetitions of N-fold outer × K-fold inner stratified cross-
validation, with hyperparameter tuning via Optuna in the inner loop.

Key design decisions (justified in the report):
    - Per-repetition seeding (base_seed + r) for reproducibility.
    - Stratified folds throughout (preserves the 54/46 class ratio).
    - Preprocessing fitted strictly on training folds (no leakage).
    - Inner loop optimises MCC (single number, threshold-aware,
      handles class imbalance well).
    - Best hyperparameters across the K inner folds are selected by
      mean validation MCC (more robust than per-fold best).
    - Inner loop can be disabled (inner_cv=None) for the default-HP
      baseline comparison required in Task 3.2.
    - Optional feature selection on the outer loop (Task 4):
      `feature_selection_k` retains the top-k features by permutation
      importance, fitted on the outer training fold only.

Example
-------
>>> rncv = RepeatedNestedCV(
...     estimator_configs=get_estimator_configs(seed=42),
...     n_outer=5, n_inner=3, n_repetitions=10,
...     n_optuna_trials=50, base_seed=42,
... )
>>> results_df = rncv.run(X, y)
>>> summary = rncv.summary()                # median + 95% CI per algorithm
"""

from __future__ import annotations

import warnings
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline        import Pipeline

from .preprocessing      import build_preprocessor
from .estimators         import EstimatorConfig
from .metrics            import compute_metrics, bootstrap_median_ci, METRIC_NAMES
from .feature_selection  import select_top_k_features


# ── Suppress noisy library output (assignment hint) ──────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


class RepeatedNestedCV:
    """Repeated Nested Cross-Validation pipeline (OOP implementation).

    Parameters
    ----------
    estimator_configs : List[EstimatorConfig]
        Algorithms to evaluate. Each config bundles an estimator factory
        with its Optuna hyperparameter-space callable.
    n_outer : int, default 5
        Outer-loop folds (used for generalisation-performance estimation).
    n_inner : int or None, default 3
        Inner-loop folds (used for hyperparameter tuning). Set to None to
        disable tuning entirely — the algorithms are then evaluated with
        their default hyperparameters (Task 3.2 baseline).
    n_repetitions : int, default 10
        Number of independent repetitions of the full nCV procedure.
    n_optuna_trials : int, default 50
        Trials per inner Optuna study.
    base_seed : int, default 42
        Repetition r uses seed = base_seed + r.
    inner_metric : str, default 'mcc'
        Metric optimised by Optuna in the inner loop.
    feature_selection_k : int or None, default None
        If set, retain the top-k features (by permutation importance) on
        the outer training fold of each iteration before passing to the
        inner loop. If None, all features are used (Task 3 behaviour).
    """

    def __init__(
        self,
        estimator_configs:    List[EstimatorConfig],
        n_outer:              int = 5,
        n_inner:              Optional[int] = 3,
        n_repetitions:        int = 10,
        n_optuna_trials:      int = 50,
        base_seed:            int = 42,
        inner_metric:         str = 'mcc',
        feature_selection_k:  Optional[int] = None,
    ):
        self.estimator_configs   = estimator_configs
        self.n_outer             = n_outer
        self.n_inner             = n_inner
        self.n_repetitions       = n_repetitions
        self.n_optuna_trials     = n_optuna_trials
        self.base_seed           = base_seed
        self.inner_metric        = inner_metric
        self.feature_selection_k = feature_selection_k

        # Filled in by .run()
        self.results_:           Optional[pd.DataFrame] = None
        self.best_hps_:          Optional[List[Dict]]   = None
        self.selected_features_: Optional[List[Dict]]   = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def run(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """Execute the full rnCV procedure.

        Returns
        -------
        results_ : pd.DataFrame
            One row per (algorithm × repetition × outer-fold) with columns
            for each metric in METRIC_NAMES, plus identifying columns
            ['algorithm', 'repetition', 'outer_fold'].
        """
        y = np.asarray(y)
        all_rows:     List[Dict] = []
        all_hps:      List[Dict] = []
        all_selected: List[Dict] = []

        for r in range(self.n_repetitions):
            seed_r = self.base_seed + r
            outer_cv = StratifiedKFold(n_splits=self.n_outer,
                                       shuffle=True, random_state=seed_r)

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y[train_idx],     y[test_idx]

                # ── Optional feature selection (outer-train only) ─────────
                # Selection runs once per outer fold and the same feature
                # subset is used for both the inner HP search and the final
                # outer-test evaluation. The locked outer test fold is never
                # seen by the selector — preventing data leakage.
                if self.feature_selection_k is not None:
                    selected = select_top_k_features(
                        X_tr, y_tr,
                        k=self.feature_selection_k,
                        seed=seed_r,
                    )
                    X_tr_sel = X_tr[selected]
                    X_te_sel = X_te[selected]
                else:
                    selected = X_tr.columns.tolist()
                    X_tr_sel, X_te_sel = X_tr, X_te

                for cfg in self.estimator_configs:
                    # Inner loop (HP tuning) — uses the already-selected features
                    if self.n_inner is not None and cfg.hp_space is not None:
                        best_hp = self._inner_loop(cfg, X_tr_sel, y_tr, seed_r)
                    else:
                        best_hp = {}

                    # Refit best HPs on the full outer train fold (selected feats)
                    pipeline = self._make_pipeline(cfg, best_hp,
                                                   feature_subset=selected)
                    pipeline.fit(X_tr_sel, y_tr)

                    # Evaluate on the locked outer test fold (same features)
                    y_pred  = pipeline.predict(X_te_sel)
                    y_proba = (pipeline.predict_proba(X_te_sel)[:, 1]
                               if hasattr(pipeline.named_steps['model'],
                                          'predict_proba') else None)
                    metrics = compute_metrics(y_te, y_pred, y_proba)

                    all_rows.append({
                        'algorithm':  cfg.name,
                        'repetition': r,
                        'outer_fold': fold_idx,
                        **metrics,
                    })
                    all_hps.append({'algorithm': cfg.name, 'repetition': r,
                                    'outer_fold': fold_idx, 'hp': best_hp})
                    all_selected.append({
                        'algorithm':  cfg.name,
                        'repetition': r,
                        'outer_fold': fold_idx,
                        'features':   selected,
                    })

            print(f'  Repetition {r + 1}/{self.n_repetitions} complete.')

        cols = ['algorithm', 'repetition', 'outer_fold'] + list(METRIC_NAMES)
        self.results_           = pd.DataFrame(all_rows, columns=cols)
        self.best_hps_          = all_hps
        self.selected_features_ = all_selected
        return self.results_

    def summary(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Per-algorithm median + 95% bootstrap CI for each metric.

        Parameters
        ----------
        metrics : list of str, optional
            Subset of metrics to summarise. Defaults to all.

        Returns
        -------
        pd.DataFrame  (algorithms × metrics)
            Each cell formatted as 'median [ci_lo, ci_hi]'.
        """
        if self.results_ is None:
            raise RuntimeError('Call .run() before .summary().')
        metrics = metrics or list(METRIC_NAMES)

        rows = []
        for algo, group in self.results_.groupby('algorithm', sort=False):
            row = {'algorithm': algo}
            for m in metrics:
                med, lo, hi = bootstrap_median_ci(group[m].to_numpy(),
                                                  seed=self.base_seed)
                row[m] = f'{med:.3f} [{lo:.3f}, {hi:.3f}]'
            rows.append(row)
        return pd.DataFrame(rows).set_index('algorithm')

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _make_pipeline(self, cfg: EstimatorConfig,
                       hp: Dict | None,
                       feature_subset: Optional[List[str]] = None) -> Pipeline:
        """Build a fresh preprocessing + estimator pipeline for one fold.

        If `feature_subset` is provided, the preprocessor will be configured
        to operate only on those columns — necessary when feature selection
        has reduced the dataframe to a subset of the original 13 features.
        """
        return Pipeline([
            ('preprocessor', build_preprocessor(feature_subset=feature_subset)),
            ('model',        cfg.build(hp)),
        ])

    def _inner_loop(
        self, cfg: EstimatorConfig, X_outer_tr: pd.DataFrame,
        y_outer_tr: np.ndarray, seed_r: int,
    ) -> Dict:
        """Run the K-fold inner loop with Optuna; return best hyperparameters.

        For each Optuna trial we evaluate the candidate hyperparameters
        across all K inner folds and report the *mean* inner-val MCC.
        Aggregating across folds before returning to Optuna is more robust
        than per-fold optimisation: TPE selects HPs that generalise across
        the inner-train splits rather than ones that exploit a single
        favourable split.
        """
        inner_cv = StratifiedKFold(n_splits=self.n_inner,
                                   shuffle=True, random_state=seed_r)
        splits = list(inner_cv.split(X_outer_tr, y_outer_tr))

        def objective(trial) -> float:
            hp = cfg.hp_space(trial)
            fold_scores = []
            # The features we're working with in the inner loop are exactly
            # the columns of X_outer_tr (already feature-selected if applicable).
            inner_features = X_outer_tr.columns.tolist()
            for tr_idx, val_idx in splits:
                X_in_tr, X_in_val = X_outer_tr.iloc[tr_idx], X_outer_tr.iloc[val_idx]
                y_in_tr, y_in_val = y_outer_tr[tr_idx],      y_outer_tr[val_idx]

                pipe = self._make_pipeline(cfg, hp, feature_subset=inner_features)
                pipe.fit(X_in_tr, y_in_tr)
                y_pred  = pipe.predict(X_in_val)
                y_proba = (pipe.predict_proba(X_in_val)[:, 1]
                           if hasattr(pipe.named_steps['model'],
                                      'predict_proba') else None)
                fold_scores.append(
                    compute_metrics(y_in_val, y_pred, y_proba)[self.inner_metric]
                )
            return float(np.mean(fold_scores))

        sampler = optuna.samplers.TPESampler(seed=seed_r)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_optuna_trials,
                       show_progress_bar=False)
        return study.best_params

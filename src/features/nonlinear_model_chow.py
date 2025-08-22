# src/features/extract_statistical_ml_features.py
from typing import Dict, Union, List, Tuple, Optional
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import pickle
from pathlib import Path
import joblib

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class StatisticalMLTestFeatureExtractor(BaseFeatureExtractor):
    """
    Create custom statistical tests using ML models as test statistics.

    Core idea:
    1. Use training data to estimate null distributions of model performance differences
    2. For new data, compute performance differences and calculate p-values
    3. These p-values become interpretable features

    This creates proper statistical tests where:
    - Test statistic = how much better separate models fit vs single model
    - Null hypothesis = no structural break (estimated from training data)
    - P-value = probability of seeing this improvement under null hypothesis
    """

    def __init__(self,
                 models_config: Dict[str, Dict] = None,
                 min_samples_per_period: int = 30,
                 lags: int = 5,
                 null_distribution_cache: str = None,
                 bootstrap_samples: int = 1000,
                 cache_name: str = 'statistical_ml_features',
                 force_recompute: bool = False,
                 no_cache: bool = False):
        """
        Initialize the statistical ML test feature extractor.

        Args:
            models_config: Dictionary of model configurations
            min_samples_per_period: Minimum samples needed in each period
            lags: Number of autoregressive lags
            null_distribution_cache: Path to cache null distributions
            bootstrap_samples: Number of bootstrap samples for distribution estimation
            cache_name: Name for the cache file
            force_recompute: If True, ignore cache and recompute all features
            no_cache: If True, skip all cache operations
        """

        if models_config is None:
            models_config = {
                'random_forest': {
                    'model': RandomForestRegressor,
                    'params': {
                        'n_estimators': 30,
                        'max_depth': 6,
                        'min_samples_split': 5,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor,
                    'params': {
                        'n_estimators': 30,
                        'max_depth': 4,
                        'learning_rate': 0.1,
                        'random_state': 42
                    }
                }
            }

        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            no_cache=no_cache,
            models_config=models_config,
            min_samples_per_period=min_samples_per_period,
            lags=lags,
            null_distribution_cache=null_distribution_cache,
            bootstrap_samples=bootstrap_samples
        )

        self.models_config = models_config
        self.min_samples_per_period = min_samples_per_period
        self.lags = lags
        self.bootstrap_samples = bootstrap_samples

        # Null distributions (learned from training data)
        self.null_distributions = {}
        self.null_distributions_fitted = False

        # Cache for null distributions
        if null_distribution_cache:
            self.null_cache_path = Path(null_distribution_cache)
        else:
            self.null_cache_path = Path("cache/ml_null_distributions.pkl")

        self._load_null_distributions()

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names produced by this extractor."""
        base_features = []

        for model_name in self.models_config.keys():
            model_features = [
                # P-values (main statistical tests)
                f'{model_name}_loss_pvalue',
                f'{model_name}_loss_log_pvalue',
                f'{model_name}_loss_significant',

                # Effect sizes (standardized improvements)
                f'{model_name}_effect_size',
                f'{model_name}_effect_size_abs',

                # Percentile ranks (where does this fall in null distribution?)
                f'{model_name}_percentile_rank',
                f'{model_name}_extreme_percentile',  # >95th or <5th percentile

                # Raw statistics (for debugging)
                f'{model_name}_loss_improvement',
                f'{model_name}_null_mean',
                f'{model_name}_null_std'
            ]
            base_features.extend(model_features)

        # Ensemble statistical features
        ensemble_features = [
            'ensemble_fisher_combined_pvalue',  # Fisher's method to combine p-values
            'ensemble_min_pvalue',  # Most significant test
            'ensemble_bonferroni_significant',  # Bonferroni-corrected significance
            'ensemble_consensus_significant',  # Fraction of tests that are significant
            'ensemble_effect_size_max',  # Largest effect size
            'ensemble_effect_size_mean'  # Average effect size
        ]

        return base_features + ensemble_features

    def _load_null_distributions(self):
        """Load previously computed null distributions."""
        try:
            if self.null_cache_path.exists():
                with open(self.null_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.null_distributions = cached_data.get('null_distributions', {})
                    self.null_distributions_fitted = len(self.null_distributions) > 0
                    print(f"Loaded null distributions for {len(self.null_distributions)} models")
        except Exception as e:
            warnings.warn(f"Failed to load null distributions: {e}")
            self.null_distributions = {}
            self.null_distributions_fitted = False

    def _save_null_distributions(self):
        """Save computed null distributions."""
        try:
            self.null_cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'null_distributions': self.null_distributions,
                'models_config': self.models_config,
                'parameters': {
                    'lags': self.lags,
                    'min_samples_per_period': self.min_samples_per_period
                }
            }
            with open(self.null_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved null distributions to {self.null_cache_path}")
        except Exception as e:
            warnings.warn(f"Failed to save null distributions: {e}")

    def fit_null_distributions(self, training_data_dict: Dict[int, TimeSeriesData]):
        """
        Fit null distributions using no-break examples from training data.

        Args:
            training_data_dict: Dictionary of training time series data
        """
        print("Fitting null distributions from training data...")

        # Get no-break examples
        no_break_examples = [ts for ts in training_data_dict.values() if not ts.has_break]
        print(f"Found {len(no_break_examples)} no-break examples for null distribution")

        if len(no_break_examples) < 50:
            warnings.warn(f"Only {len(no_break_examples)} no-break examples. Null distributions may be unreliable.")

        # Collect loss improvements for each model
        model_improvements = {model_name: [] for model_name in self.models_config.keys()}

        for i, ts_data in enumerate(no_break_examples):
            if i % 100 == 0:
                print(f"Processing no-break example {i + 1}/{len(no_break_examples)}")

            try:
                # Skip if insufficient data
                if (len(ts_data.period_0_values) < self.min_samples_per_period or
                        len(ts_data.period_1_values) < self.min_samples_per_period):
                    continue

                # Compute loss improvements for each model
                improvements = self._compute_model_improvements(ts_data)

                for model_name, improvement in improvements.items():
                    if not np.isnan(improvement):
                        model_improvements[model_name].append(improvement)

            except Exception as e:
                warnings.warn(f"Failed to process no-break example {i}: {e}")
                continue

        # Fit distributions to the collected improvements
        for model_name, improvements in model_improvements.items():
            if len(improvements) >= 20:  # Need minimum samples
                improvements_array = np.array(improvements)

                # Store empirical distribution
                self.null_distributions[model_name] = {
                    'data': improvements_array,
                    'mean': float(np.mean(improvements_array)),
                    'std': float(np.std(improvements_array)),
                    'median': float(np.median(improvements_array)),
                    'q25': float(np.percentile(improvements_array, 25)),
                    'q75': float(np.percentile(improvements_array, 75)),
                    'n_samples': len(improvements_array)
                }

                print(f"{model_name}: {len(improvements)} samples, "
                      f"mean={np.mean(improvements_array):.4f}, "
                      f"std={np.std(improvements_array):.4f}")
            else:
                warnings.warn(f"Only {len(improvements)} samples for {model_name}. Skipping.")

        self.null_distributions_fitted = True
        self._save_null_distributions()
        print("Null distributions fitted and saved!")

    def _create_lagged_features(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create simple lagged features for modeling."""
        n = len(series)

        if n <= self.lags:
            return None, None

        # Create target and features
        y = series[self.lags:]
        X = np.zeros((len(y), self.lags))

        for i in range(self.lags):
            X[:, i] = series[self.lags - 1 - i:-1 - i]

        return X, y

    def _fit_model_and_get_loss(self, model_class, model_params: Dict, X: np.ndarray, y: np.ndarray) -> float:
        """Fit model and return MSE loss."""
        try:
            if len(X) < 10:
                return np.nan

            # Standardize for neural networks
            if model_class == MLPRegressor:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            # Fit model
            model = model_class(**model_params)
            model.fit(X_scaled, y)

            # Get predictions and compute loss
            y_pred = model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)

            return float(mse)

        except Exception:
            return np.nan

    def _compute_model_improvements(self, ts_data: TimeSeriesData) -> Dict[str, float]:
        """Compute loss improvements for all models on this time series."""
        improvements = {}

        try:
            # Extract data
            pre_series = ts_data.period_0_values
            post_series = ts_data.period_1_values
            full_series = np.concatenate([pre_series, post_series])

            # Create features
            X_pre, y_pre = self._create_lagged_features(pre_series)
            X_post, y_post = self._create_lagged_features(post_series)
            X_full, y_full = self._create_lagged_features(full_series)

            if X_pre is None or X_post is None or X_full is None:
                return {model_name: np.nan for model_name in self.models_config.keys()}

            # Test each model
            for model_name, model_config in self.models_config.items():
                model_class = model_config['model']
                model_params = model_config['params']

                # Fit separate models
                loss_pre = self._fit_model_and_get_loss(model_class, model_params, X_pre, y_pre)
                loss_post = self._fit_model_and_get_loss(model_class, model_params, X_post, y_post)

                # Fit single model
                loss_full = self._fit_model_and_get_loss(model_class, model_params, X_full, y_full)

                # Compute improvement (positive = separate models better)
                if not (np.isnan(loss_pre) or np.isnan(loss_post) or np.isnan(loss_full)):
                    loss_separate = loss_pre + loss_post
                    improvement = loss_full - loss_separate
                    improvements[model_name] = improvement
                else:
                    improvements[model_name] = np.nan

        except Exception:
            improvements = {model_name: np.nan for model_name in self.models_config.keys()}

        return improvements

    def _compute_statistical_features(self, observed_improvement: float, model_name: str) -> Dict[str, float]:
        """Compute statistical features given observed improvement and null distribution."""
        features = {
            f'{model_name}_loss_pvalue': np.nan,
            f'{model_name}_loss_log_pvalue': np.nan,
            f'{model_name}_loss_significant': np.nan,
            f'{model_name}_effect_size': np.nan,
            f'{model_name}_effect_size_abs': np.nan,
            f'{model_name}_percentile_rank': np.nan,
            f'{model_name}_extreme_percentile': np.nan,
            f'{model_name}_loss_improvement': np.nan,
            f'{model_name}_null_mean': np.nan,
            f'{model_name}_null_std': np.nan
        }

        if np.isnan(observed_improvement) or model_name not in self.null_distributions:
            return features

    def _compute_raw_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute raw loss improvements when null distributions aren't available."""
        features = {}

        try:
            # Compute model improvements
            improvements = self._compute_model_improvements(data)

            # Return raw improvements instead of p-values
            for model_name, improvement in improvements.items():
                features.update({
                    f'{model_name}_loss_improvement': improvement,
                    f'{model_name}_loss_pvalue': np.nan,
                    f'{model_name}_loss_log_pvalue': np.nan,
                    f'{model_name}_loss_significant': np.nan,
                    f'{model_name}_effect_size': np.nan,
                    f'{model_name}_effect_size_abs': np.nan,
                    f'{model_name}_percentile_rank': np.nan,
                    f'{model_name}_extreme_percentile': np.nan,
                    f'{model_name}_null_mean': np.nan,
                    f'{model_name}_null_std': np.nan
                })

            # Ensemble features (basic)
            valid_improvements = [imp for imp in improvements.values() if not np.isnan(imp)]
            if valid_improvements:
                features.update({
                    'ensemble_fisher_combined_pvalue': np.nan,
                    'ensemble_min_pvalue': np.nan,
                    'ensemble_bonferroni_significant': np.nan,
                    'ensemble_consensus_significant': np.nan,
                    'ensemble_effect_size_max': float(np.max(np.abs(valid_improvements))),
                    'ensemble_effect_size_mean': float(np.mean(np.abs(valid_improvements)))
                })
            else:
                for feature_name in ['ensemble_fisher_combined_pvalue', 'ensemble_min_pvalue',
                                     'ensemble_bonferroni_significant', 'ensemble_consensus_significant',
                                     'ensemble_effect_size_max', 'ensemble_effect_size_mean']:
                    features[feature_name] = np.nan

        except Exception as e:
            warnings.warn(f"Error in raw feature extraction: {e}")
            for feature_name in self.get_feature_names():
                features[feature_name] = np.nan

        return features

        try:
            null_dist = self.null_distributions[model_name]
            null_data = null_dist['data']
            null_mean = null_dist['mean']
            null_std = null_dist['std']

            # Compute p-value (one-tailed: P(X >= observed))
            p_value = np.mean(null_data >= observed_improvement)

            # Handle edge cases
            if p_value == 0:
                p_value = 1.0 / (len(null_data) + 1)  # Conservative estimate
            elif p_value == 1:
                p_value = 1.0 - 1.0 / (len(null_data) + 1)

            # Effect size (Cohen's d equivalent)
            effect_size = (observed_improvement - null_mean) / null_std if null_std > 0 else 0

            # Percentile rank
            percentile_rank = np.mean(null_data <= observed_improvement)

            # Extreme percentile indicator
            extreme_percentile = float(percentile_rank > 0.95 or percentile_rank < 0.05)

            features.update({
                f'{model_name}_loss_pvalue': float(p_value),
                f'{model_name}_loss_log_pvalue': float(-np.log10(p_value)) if p_value > 0 else 10.0,
                f'{model_name}_loss_significant': float(p_value < 0.05),
                f'{model_name}_effect_size': float(effect_size),
                f'{model_name}_effect_size_abs': float(abs(effect_size)),
                f'{model_name}_percentile_rank': float(percentile_rank),
                f'{model_name}_extreme_percentile': extreme_percentile,
                f'{model_name}_loss_improvement': float(observed_improvement),
                f'{model_name}_null_mean': float(null_mean),
                f'{model_name}_null_std': float(null_std)
            })

        except Exception as e:
            warnings.warn(f"Failed to compute statistical features for {model_name}: {e}")

        return features

    def _compute_ensemble_features(self, all_pvalues: List[float], all_effect_sizes: List[float]) -> Dict[str, float]:
        """Compute ensemble statistical features."""
        features = {
            'ensemble_fisher_combined_pvalue': np.nan,
            'ensemble_min_pvalue': np.nan,
            'ensemble_bonferroni_significant': np.nan,
            'ensemble_consensus_significant': np.nan,
            'ensemble_effect_size_max': np.nan,
            'ensemble_effect_size_mean': np.nan
        }

        try:
            valid_pvalues = [p for p in all_pvalues if not np.isnan(p) and p > 0]
            valid_effect_sizes = [e for e in all_effect_sizes if not np.isnan(e)]

            if valid_pvalues:
                # Fisher's method to combine p-values
                fisher_stat = -2 * np.sum(np.log(valid_pvalues))
                fisher_pvalue = 1 - stats.chi2.cdf(fisher_stat, 2 * len(valid_pvalues))

                # Minimum p-value
                min_pvalue = np.min(valid_pvalues)

                # Bonferroni correction
                bonferroni_threshold = 0.05 / len(valid_pvalues)
                bonferroni_significant = float(min_pvalue < bonferroni_threshold)

                # Consensus (fraction significant)
                consensus_significant = np.mean([p < 0.05 for p in valid_pvalues])

                features.update({
                    'ensemble_fisher_combined_pvalue': float(fisher_pvalue),
                    'ensemble_min_pvalue': float(min_pvalue),
                    'ensemble_bonferroni_significant': bonferroni_significant,
                    'ensemble_consensus_significant': float(consensus_significant)
                })

            if valid_effect_sizes:
                features.update({
                    'ensemble_effect_size_max': float(np.max(np.abs(valid_effect_sizes))),
                    'ensemble_effect_size_mean': float(np.mean(np.abs(valid_effect_sizes)))
                })

        except Exception as e:
            warnings.warn(f"Failed to compute ensemble features: {e}")

        return features

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute all statistical ML test features."""
        features = {}

        if not self.null_distributions_fitted:
            # Try to load from cache first
            self._load_null_distributions()

            if not self.null_distributions_fitted:
                # If still not fitted, return raw improvements instead of p-values
                warnings.warn("Null distributions not fitted! Returning raw loss improvements instead of p-values. "
                              "Call fit_null_distributions() for proper statistical tests.")
                return self._compute_raw_features(data)

        # Continue with normal statistical feature computation...

        try:
            # Compute model improvements
            improvements = self._compute_model_improvements(data)

            all_pvalues = []
            all_effect_sizes = []

            # Compute statistical features for each model
            for model_name, improvement in improvements.items():
                model_features = self._compute_statistical_features(improvement, model_name)
                features.update(model_features)

                # Collect for ensemble features
                if not np.isnan(improvement) and model_name in self.null_distributions:
                    p_val = model_features.get(f'{model_name}_loss_pvalue', np.nan)
                    effect_size = model_features.get(f'{model_name}_effect_size', np.nan)

                    if not np.isnan(p_val):
                        all_pvalues.append(p_val)
                    if not np.isnan(effect_size):
                        all_effect_sizes.append(effect_size)

            # Compute ensemble features
            ensemble_features = self._compute_ensemble_features(all_pvalues, all_effect_sizes)
            features.update(ensemble_features)

        except Exception as e:
            warnings.warn(f"Error in statistical ML feature extraction: {e}")
            # Fill all features with NaN on error
            for feature_name in self.get_feature_names():
                features[feature_name] = np.nan

        return features
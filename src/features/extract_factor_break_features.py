# src/features/extract_factor_break_features.py
from typing import Dict, Union, Optional, List
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS
from scipy import stats

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class FactorBreakFeatureExtractor(BaseFeatureExtractor):
    """
    Extract factor-based structural break features with caching support.
    """

    def __init__(self,
                 n_factors: int = 3,
                 embed_dim: int = 20,
                 min_samples: int = 20,
                 cache_name: str = 'factor_break_features',
                 force_recompute: bool = False):
        """
        Initialize the feature extractor.

        Args:
            n_factors: Number of factors to extract
            embed_dim: Embedding dimension for creating multivariate representation
            min_samples: Minimum samples needed for factor analysis
            cache_name: Name for the cache file
            force_recompute: If True, ignore cache and recompute all features
        """
        # Store parameters that affect feature computation
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            n_factors=n_factors,
            embed_dim=embed_dim,
            min_samples=min_samples
        )

        self.n_factors = n_factors
        self.embed_dim = embed_dim
        self.min_samples = min_samples

        # These will be set during computation
        self.pca0 = None
        self.pca1 = None
        self.factors0 = None
        self.factors1 = None

    @staticmethod
    def create_embedded_matrix(series: np.ndarray, embed_dim: int = 10, stride: int = 1) -> Optional[np.ndarray]:
        """Create an embedded matrix from univariate series for factor analysis."""
        n = len(series)
        if n < embed_dim:
            return None

        n_windows = (n - embed_dim) // stride + 1
        embedded = np.zeros((n_windows, embed_dim))

        for i in range(n_windows):
            start_idx = i * stride
            embedded[i, :] = series[start_idx:start_idx + embed_dim]

        return embedded

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names produced by this extractor."""
        base_features = [
            'factor_break_wald_stat', 'factor_break_wald_pvalue',
            'factor_regression_r2_period0', 'factor_regression_r2_period1',
            'factor_regression_r2_diff', 'factor_loadings_distance',
            'factor_variance_ratio_change', 'factor_correlation_change',
            'factor_correlation_change_max', 'factor_space_angle',
            'factor_space_angle_mean', 'factor_break_chow_stat',
            'factor_stability_score', 'factor_evolution_rate',
            'factor_break_location_score'
        ]

        # Add variance ratio features
        for i in range(min(3, self.n_factors)):
            base_features.extend([
                f'factor_var_ratio_p0_comp{i}',
                f'factor_var_ratio_p1_comp{i}'
            ])

        # Add factor count features
        for embed_dim in [10, 15, 20]:
            base_features.extend([
                f'n_factors_95var_period0_embed{embed_dim}',
                f'n_factors_95var_period1_embed{embed_dim}',
                f'n_factors_difference_embed{embed_dim}',
                f'n_factors_90var_period0_embed{embed_dim}',
                f'n_factors_90var_period1_embed{embed_dim}'
            ])

        return base_features

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute all factor-based features for a single time series."""
        # Extract period values
        vals0, vals1 = self._extract_period_values(data)

        # Create embedded matrices
        embed0 = self.create_embedded_matrix(vals0, self.embed_dim)
        embed1 = self.create_embedded_matrix(vals1, self.embed_dim)

        # Check if we have enough data
        if (embed0 is None or embed1 is None or
                len(embed0) < self.min_samples or len(embed1) < self.min_samples):
            # Return NaN features
            return {name: np.nan for name in self.get_feature_names()}

        # Extract all feature groups
        features = {}

        # Basic variance features (this also sets self.pca0, self.pca1, etc.)
        features.update(self._extract_variance_features(vals0, vals1, embed0, embed1))

        # Regression features
        features.update(self._extract_regression_features())

        # Loading features
        features.update(self._extract_loading_features())

        # Correlation features
        features.update(self._extract_correlation_features())

        # Factor count features
        features.update(self._extract_factor_count_features(vals0, vals1))

        # Dynamic features
        features.update(self._extract_dynamic_features(data))

        return features

    def _extract_variance_features(self, vals0: np.ndarray, vals1: np.ndarray,
                                   embed0: np.ndarray, embed1: np.ndarray) -> Dict[str, float]:
        """Extract features based on explained variance ratios."""
        features = {}

        # Standardize embedded matrices
        embed0_std = (embed0 - np.mean(embed0, axis=0)) / (np.std(embed0, axis=0) + 1e-8)
        embed1_std = (embed1 - np.mean(embed1, axis=0)) / (np.std(embed1, axis=0) + 1e-8)

        # Extract factors using PCA
        self.pca0 = PCA(n_components=self.n_factors)
        self.pca1 = PCA(n_components=self.n_factors)

        self.factors0 = self.pca0.fit_transform(embed0_std)
        self.factors1 = self.pca1.fit_transform(embed1_std)

        # Change in explained variance ratios
        var_ratio0 = self.pca0.explained_variance_ratio_
        var_ratio1 = self.pca1.explained_variance_ratio_
        features['factor_variance_ratio_change'] = float(np.sum(np.abs(var_ratio1 - var_ratio0)))

        # Individual variance ratios for first 3 components
        for i in range(min(3, self.n_factors)):
            features[f'factor_var_ratio_p0_comp{i}'] = float(var_ratio0[i]) if i < len(var_ratio0) else 0.0
            features[f'factor_var_ratio_p1_comp{i}'] = float(var_ratio1[i]) if i < len(var_ratio1) else 0.0

        return features

    def _extract_regression_features(self) -> Dict[str, float]:
        """Extract features based on factor regression relationships."""
        features = {}

        if self.n_factors > 1:
            # Period 0 regression
            X0 = self.factors0[:, 1:]
            y0 = self.factors0[:, 0]
            model0 = OLS(y0, np.column_stack([np.ones(len(y0)), X0])).fit()
            features['factor_regression_r2_period0'] = float(model0.rsquared)

            # Period 1 regression
            X1 = self.factors1[:, 1:]
            y1 = self.factors1[:, 0]
            model1 = OLS(y1, np.column_stack([np.ones(len(y1)), X1])).fit()
            features['factor_regression_r2_period1'] = float(model1.rsquared)

            # R-squared difference
            features['factor_regression_r2_diff'] = float(model1.rsquared - model0.rsquared)

            # Chow test for parameter stability
            n0, n1 = len(y0), len(y1)
            rss_unrestricted = model0.ssr + model1.ssr

            # Combined regression
            X_combined = np.vstack([
                np.column_stack([np.ones(len(y0)), X0]),
                np.column_stack([np.ones(len(y1)), X1])
            ])
            y_combined = np.concatenate([y0, y1])
            model_restricted = OLS(y_combined, X_combined).fit()
            rss_restricted = model_restricted.ssr

            # Chow test statistic
            k = X0.shape[1] + 1
            if n0 + n1 - 2 * k > 0:
                chow_stat = ((rss_restricted - rss_unrestricted) / k) / (rss_unrestricted / (n0 + n1 - 2 * k))
                chow_pvalue = 1 - stats.f.cdf(chow_stat, k, n0 + n1 - 2 * k)
            else:
                chow_stat = np.nan
                chow_pvalue = np.nan

            features['factor_break_chow_stat'] = float(chow_stat)
            features['factor_break_wald_stat'] = float(chow_stat)
            features['factor_break_wald_pvalue'] = float(chow_pvalue)
        else:
            features['factor_regression_r2_period0'] = np.nan
            features['factor_regression_r2_period1'] = np.nan
            features['factor_regression_r2_diff'] = np.nan
            features['factor_break_chow_stat'] = np.nan
            features['factor_break_wald_stat'] = np.nan
            features['factor_break_wald_pvalue'] = np.nan

        return features

    def _extract_loading_features(self) -> Dict[str, float]:
        """Extract features based on factor loadings."""
        features = {}

        loadings0 = self.pca0.components_.T
        loadings1 = self.pca1.components_.T

        # Procrustes distance between loadings
        U, _, Vt = np.linalg.svd(loadings1.T @ loadings0)
        R = U @ Vt
        loadings1_aligned = loadings1 @ R.T

        features['factor_loadings_distance'] = float(np.linalg.norm(loadings0 - loadings1_aligned, 'fro'))

        # Principal angle between factor spaces
        try:
            Q0, _ = np.linalg.qr(loadings0)
            Q1, _ = np.linalg.qr(loadings1)
            _, S, _ = np.linalg.svd(Q0.T @ Q1)
            S = np.clip(S, -1, 1)
            principal_angles = np.arccos(S)
            features['factor_space_angle'] = float(np.max(principal_angles))
            features['factor_space_angle_mean'] = float(np.mean(principal_angles))
        except:
            features['factor_space_angle'] = np.nan
            features['factor_space_angle_mean'] = np.nan

        return features

    def _extract_correlation_features(self) -> Dict[str, float]:
        """Extract features based on factor correlations."""
        features = {}

        if self.n_factors > 1:
            corr0 = np.corrcoef(self.factors0.T)
            corr1 = np.corrcoef(self.factors1.T)

            # Extract upper triangular elements
            triu_idx = np.triu_indices(self.n_factors, k=1)
            corr_diff = np.abs(corr1[triu_idx] - corr0[triu_idx])

            features['factor_correlation_change'] = float(np.mean(corr_diff))
            features['factor_correlation_change_max'] = float(np.max(corr_diff)) if len(corr_diff) > 0 else 0.0
        else:
            features['factor_correlation_change'] = 0.0
            features['factor_correlation_change_max'] = 0.0

        return features

    def _extract_factor_count_features(self, vals0: np.ndarray, vals1: np.ndarray) -> Dict[str, float]:
        """Extract features based on number of factors needed to explain variance."""
        features = {}

        # Try different embedding dimensions
        for embed_dim in [10, 15, 20]:
            embed0 = self.create_embedded_matrix(vals0, embed_dim=embed_dim)
            embed1 = self.create_embedded_matrix(vals1, embed_dim=embed_dim)

            if embed0 is None or embed1 is None or len(embed0) < 10 or len(embed1) < 10:
                features[f'n_factors_95var_period0_embed{embed_dim}'] = np.nan
                features[f'n_factors_95var_period1_embed{embed_dim}'] = np.nan
                features[f'n_factors_difference_embed{embed_dim}'] = np.nan
                features[f'n_factors_90var_period0_embed{embed_dim}'] = np.nan
                features[f'n_factors_90var_period1_embed{embed_dim}'] = np.nan
                continue

            try:
                # Standardize
                embed0_std = (embed0 - np.mean(embed0, axis=0)) / (np.std(embed0, axis=0) + 1e-8)
                embed1_std = (embed1 - np.mean(embed1, axis=0)) / (np.std(embed1, axis=0) + 1e-8)

                # Fit PCA with 95% variance
                pca0 = PCA(n_components=0.95)
                pca1 = PCA(n_components=0.95)

                pca0.fit(embed0_std)
                pca1.fit(embed1_std)

                features[f'n_factors_95var_period0_embed{embed_dim}'] = int(pca0.n_components_)
                features[f'n_factors_95var_period1_embed{embed_dim}'] = int(pca1.n_components_)
                features[f'n_factors_difference_embed{embed_dim}'] = int(pca1.n_components_ - pca0.n_components_)

                # Also try 90% variance
                pca0_90 = PCA(n_components=0.90)
                pca1_90 = PCA(n_components=0.90)

                pca0_90.fit(embed0_std)
                pca1_90.fit(embed1_std)

                features[f'n_factors_90var_period0_embed{embed_dim}'] = int(pca0_90.n_components_)
                features[f'n_factors_90var_period1_embed{embed_dim}'] = int(pca1_90.n_components_)

            except Exception as e:
                features[f'n_factors_95var_period0_embed{embed_dim}'] = np.nan
                features[f'n_factors_95var_period1_embed{embed_dim}'] = np.nan
                features[f'n_factors_difference_embed{embed_dim}'] = np.nan
                features[f'n_factors_90var_period0_embed{embed_dim}'] = np.nan
                features[f'n_factors_90var_period1_embed{embed_dim}'] = np.nan

        return features

    def _extract_dynamic_features(self, data: Union[pd.DataFrame, TimeSeriesData],
                                  window_size: int = 20, step_size: int = 5) -> Dict[str, float]:
        """Extract features based on dynamic factor evolution."""
        features = {}

        # Extract full series
        if isinstance(data, TimeSeriesData):
            full_series = np.concatenate([data.period_0_values, data.period_1_values])
            break_point = len(data.period_0_values)
        else:
            full_series = data['value'].dropna().values
            break_point = len(data[data['period'] == 0])

        n = len(full_series)
        if n < window_size * 2:
            return {
                'factor_stability_score': np.nan,
                'factor_evolution_rate': np.nan,
                'factor_break_location_score': np.nan
            }

        # Track factor evolution
        factor_distances = []
        window_centers = []
        prev_pca = None

        for i in range(0, n - window_size, step_size):
            window = full_series[i:i + window_size]
            embedded = self.create_embedded_matrix(window, embed_dim=min(10, window_size // 2))

            if embedded is None or len(embedded) < 10:
                continue

            try:
                pca = PCA(n_components=2)
                pca.fit(embedded)

                if prev_pca is not None:
                    # Compare loadings
                    dist = np.linalg.norm(prev_pca.components_ - pca.components_, 'fro')
                    factor_distances.append(dist)
                    window_centers.append(i + window_size // 2)

                prev_pca = pca
            except:
                continue

        if len(factor_distances) > 0:
            factor_distances = np.array(factor_distances)
            window_centers = np.array(window_centers)

            features['factor_stability_score'] = float(np.mean(factor_distances))
            features['factor_evolution_rate'] = float(np.std(factor_distances))

            # Location of maximum change
            max_change_idx = np.argmax(factor_distances)
            max_change_location = window_centers[max_change_idx]
            location_score = 1.0 - abs(max_change_location - break_point) / (n / 2)
            features['factor_break_location_score'] = float(location_score)
        else:
            features['factor_stability_score'] = np.nan
            features['factor_evolution_rate'] = np.nan
            features['factor_break_location_score'] = np.nan

        return features

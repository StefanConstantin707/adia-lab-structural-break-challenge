# src/features/extract_distribution_combined_features.py
from typing import Dict, Union, Optional, List
import pandas as pd
import numpy as np
import scipy.stats as st
import warnings

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class DistributionCombinedFeatureExtractor(BaseFeatureExtractor):
    """
    Extract distribution-based features, residual-distribution features, and raw distribution-distance features
    for a time series split into two periods.
    Fixed version with proper sample size handling for statistical tests.
    """

    def __init__(
            self,
            lags: int = 1,
            min_length: Optional[int] = None,
            cache_name: str = 'distribution_combined_features',
            force_recompute: bool = False,
            min_sample_size: int = 8  # Minimum sample size for advanced statistical tests
    ):
        """
        Args:
            lags: number of lags for AR residual extraction in residual-distribution comparison.
            min_length: minimal length AFTER lags required in each segment for residual tests.
                        If None, defaults to lags + 2.
            cache_name: name for caching features.
            force_recompute: if True, ignore cache and always recompute.
            min_sample_size: minimum sample size required for advanced statistical tests.
        """
        super().__init__(cache_name=cache_name, force_recompute=force_recompute, lags=lags,
                         min_length=min_length, min_sample_size=min_sample_size)
        self.lags = lags
        self.min_length = min_length if min_length is not None else lags + 2
        self.min_sample_size = min_sample_size

    def get_feature_names(self) -> List[str]:
        """
        Return the list of all feature names produced by this extractor.
        """
        names: List[str] = []

        # Distribution descriptive features
        names += [
            'p0_mean', 'p0_std', 'p0_skew', 'p0_kurt',
            'p1_mean', 'p1_std', 'p1_skew', 'p1_kurt',
            'mean_diff', 'std_diff', 'mean_ratio', 'std_ratio',
            'ttest_stat', 'ttest_pvalue',
            'levene_stat', 'levene_pvalue',
            'mannwhitney_pvalue', 'ks_pvalue',
            'p0_length', 'p1_length', 'length_ratio'
        ]

        # Residual-distribution comparison features
        names += [
            'resid_pre_length', 'resid_post_length',
            'resid_ks_stat', 'resid_ks_pvalue',
            'resid_mannwhitney_stat', 'resid_mannwhitney_pvalue',
            'resid_ttest_stat', 'resid_ttest_pvalue',
            'resid_levene_stat', 'resid_levene_pvalue',
            'resid_wasserstein', 'resid_energy_distance',
            'resid_cramervonmises_stat', 'resid_cramervonmises_pvalue',
            'resid_epps_singleton_stat', 'resid_epps_singleton_pvalue'
        ]

        # Raw distribution-distance features
        names += [
            'raw_p0_length', 'raw_p1_length',
            'raw_wasserstein', 'raw_energy_distance',
            'raw_ks_stat', 'raw_ks_pvalue',
            'raw_cramervonmises_stat', 'raw_cramervonmises_pvalue',
            'raw_epps_singleton_stat', 'raw_epps_singleton_pvalue'
        ]
        return names

    def _extract_period_values(self, data: Union[pd.DataFrame, TimeSeriesData]) -> (np.ndarray, np.ndarray):
        """
        Return (period_0_array, period_1_array) as float numpy arrays.
        """
        if isinstance(data, TimeSeriesData):
            p0 = np.asarray(data.period_0_values, dtype=float)
            p1 = np.asarray(data.period_1_values, dtype=float)
        elif isinstance(data, pd.DataFrame):
            if 'value' not in data.columns or 'period' not in data.columns:
                raise ValueError("DataFrame input must have 'value' and 'period' columns.")
            p0 = data.loc[data['period'] == 0, 'value'].dropna().astype(float).values
            p1 = data.loc[data['period'] == 1, 'value'].dropna().astype(float).values
        else:
            raise TypeError("Input must be a pandas DataFrame or TimeSeriesData instance.")
        return p0, p1

    def _build_lagged_matrix(self, series: np.ndarray, lags: int) -> (Optional[np.ndarray], Optional[np.ndarray]):
        """
        For series length N, returns (X, y) with:
          y = series[lags:]
          X: shape (N - lags, lags + 1): col 0 ones; col j is series[lags - j : N - j].
        If N <= lags, returns (None, None).
        """
        N = len(series)
        if N <= lags:
            return None, None
        y = series[lags:]
        X = np.ones((N - lags, lags + 1), dtype=float)
        for j in range(1, lags + 1):
            X[:, j] = series[lags - j: N - j]
        return X, y

    def _ols_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit OLS via normal equations; return residuals array.
        """
        XT_X = X.T.dot(X)
        try:
            beta = np.linalg.solve(XT_X, X.T.dot(y))
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XT_X).dot(X.T.dot(y))
        resid = y - X.dot(beta)
        return resid

    def _safe_statistical_test(self, test_func, data1, data2, test_name="unknown", **kwargs):
        """
        Safely execute a statistical test with proper error handling and sample size checks.
        """
        try:
            # Check minimum sample sizes
            if len(data1) < self.min_sample_size or len(data2) < self.min_sample_size:
                return np.nan, np.nan

            # Check for constant arrays (zero variance)
            if np.var(data1) == 0 or np.var(data2) == 0:
                return np.nan, np.nan

            # Suppress warnings for this specific test
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*One or more sample arguments is too small.*")

                result = test_func(data1, data2, **kwargs)

                # Handle different return types
                if hasattr(result, 'statistic') and hasattr(result, 'pvalue'):
                    return float(result.statistic), float(result.pvalue)
                elif isinstance(result, tuple) and len(result) >= 2:
                    return float(result[0]), float(result[1])
                else:
                    return np.nan, np.nan

        except Exception as e:
            # For debugging, you can uncomment the next line
            # print(f"Statistical test {test_name} failed: {e}")
            return np.nan, np.nan

    def _safe_distance_metric(self, metric_func, data1, data2, metric_name="unknown"):
        """
        Safely execute a distance metric with proper error handling.
        """
        try:
            # Check minimum sample sizes
            if len(data1) < 2 or len(data2) < 2:
                return np.nan

            # Suppress warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)

                result = metric_func(data1, data2)
                return float(result)

        except Exception as e:
            # For debugging, you can uncomment the next line
            # print(f"Distance metric {metric_name} failed: {e}")
            return np.nan

    def _compute_distribution_features(self, p0: np.ndarray, p1: np.ndarray, features: Dict[str, float]):
        """
        Fill in distribution descriptive and comparative test features into `features` dict.
        """

        # Basic stats period 0
        if len(p0) > 0:
            features['p0_mean'] = float(np.mean(p0))
            features['p0_std'] = float(np.std(p0))
            features['p0_skew'] = float(pd.Series(p0).skew())
            features['p0_kurt'] = float(pd.Series(p0).kurt())
        else:
            features['p0_mean'] = 0.0
            features['p0_std'] = 0.0
            features['p0_skew'] = 0.0
            features['p0_kurt'] = 0.0

        # Basic stats period 1
        if len(p1) > 0:
            features['p1_mean'] = float(np.mean(p1))
            features['p1_std'] = float(np.std(p1))
            features['p1_skew'] = float(pd.Series(p1).skew())
            features['p1_kurt'] = float(pd.Series(p1).kurt())
        else:
            features['p1_mean'] = 0.0
            features['p1_std'] = 0.0
            features['p1_skew'] = 0.0
            features['p1_kurt'] = 0.0

        # Differences/ratios
        features['mean_diff'] = features['p1_mean'] - features['p0_mean']
        features['std_diff'] = features['p1_std'] - features['p0_std']
        features['mean_ratio'] = float(features['p1_mean'] / (features['p0_mean'] + 1e-8))
        features['std_ratio'] = float(features['p1_std'] / (features['p0_std'] + 1e-8))

        # Lengths
        len0 = len(p0)
        len1 = len(p1)

        features['p0_length'] = len0
        features['p1_length'] = len1
        features['length_ratio'] = float(len1 / (len0 + 1))

        # Comparative tests if possible
        if len0 > 1 and len1 > 1:

            # T-test for mean difference
            t_stat, t_pvalue = self._safe_statistical_test(st.ttest_ind, p0, p1, "ttest")
            features['ttest_stat'] = t_stat
            features['ttest_pvalue'] = t_pvalue

            # Levene test for variance
            lev_stat, lev_pvalue = self._safe_statistical_test(st.levene, p0, p1, "levene")
            features['levene_stat'] = lev_stat
            features['levene_pvalue'] = lev_pvalue

            # Mann-Whitney U test (returns statistic, pvalue)
            mw_stat, mw_pvalue = self._safe_statistical_test(st.mannwhitneyu, p0, p1, "mannwhitney")
            features['mannwhitney_pvalue'] = mw_pvalue

            # KS test
            ks_stat, ks_pvalue = self._safe_statistical_test(st.ks_2samp, p0, p1, "ks_2samp")
            features['ks_pvalue'] = ks_pvalue
        else:
            features['ttest_stat'] = np.nan
            features['ttest_pvalue'] = np.nan
            features['levene_stat'] = np.nan
            features['levene_pvalue'] = np.nan
            features['mannwhitney_pvalue'] = np.nan
            features['ks_pvalue'] = np.nan

    def _compute_residual_distribution_features(self, p0: np.ndarray, p1: np.ndarray, features: Dict[str, float]):
        """
        Fit AR(lags) separately on pre- and post- segments to obtain residuals,
        then compare residual distributions via tests and distances.
        """
        # Record raw lengths
        features['resid_pre_length'] = len(p0)
        features['resid_post_length'] = len(p1)

        # Build lagged matrices
        X_pre, y_pre = self._build_lagged_matrix(p0, self.lags)
        X_post, y_post = self._build_lagged_matrix(p1, self.lags)

        ok_pre = (X_pre is not None and len(y_pre) >= 2 and len(p0) >= self.min_length)
        ok_post = (X_post is not None and len(y_post) >= 2 and len(p1) >= self.min_length)

        # If insufficient data, fill NaNs
        if not (ok_pre and ok_post):
            for name in [
                'resid_ks_stat', 'resid_ks_pvalue',
                'resid_mannwhitney_stat', 'resid_mannwhitney_pvalue',
                'resid_ttest_stat', 'resid_ttest_pvalue',
                'resid_levene_stat', 'resid_levene_pvalue',
                'resid_wasserstein', 'resid_energy_distance',
                'resid_cramervonmises_stat', 'resid_cramervonmises_pvalue',
                'resid_epps_singleton_stat', 'resid_epps_singleton_pvalue'
            ]:
                features[name] = np.nan
            return

        # Obtain residuals
        resid_pre = self._ols_residuals(X_pre, y_pre)
        resid_post = self._ols_residuals(X_post, y_post)

        # KS test
        ks_stat, ks_pvalue = self._safe_statistical_test(st.ks_2samp, resid_pre, resid_post, "resid_ks")
        features['resid_ks_stat'] = ks_stat
        features['resid_ks_pvalue'] = ks_pvalue

        # Mann-Whitney U
        mw_stat, mw_pvalue = self._safe_statistical_test(
            st.mannwhitneyu, resid_pre, resid_post, "resid_mannwhitney", alternative='two-sided'
        )
        features['resid_mannwhitney_stat'] = mw_stat
        features['resid_mannwhitney_pvalue'] = mw_pvalue

        # t-test (Welch)
        t_stat, t_pvalue = self._safe_statistical_test(
            st.ttest_ind, resid_pre, resid_post, "resid_ttest", equal_var=False, nan_policy='omit'
        )
        features['resid_ttest_stat'] = t_stat
        features['resid_ttest_pvalue'] = t_pvalue

        # Levene
        lev_stat, lev_pvalue = self._safe_statistical_test(st.levene, resid_pre, resid_post, "resid_levene")
        features['resid_levene_stat'] = lev_stat
        features['resid_levene_pvalue'] = lev_pvalue

        # Wasserstein distance
        features['resid_wasserstein'] = self._safe_distance_metric(
            st.wasserstein_distance, resid_pre, resid_post, "resid_wasserstein"
        )

        # Energy distance
        features['resid_energy_distance'] = self._safe_distance_metric(
            st.energy_distance, resid_pre, resid_post, "resid_energy"
        )

        # Cramér-von Mises (check if available)
        try:
            from scipy.stats import cramervonmises_2samp
            cvm_stat, cvm_pvalue = self._safe_statistical_test(
                cramervonmises_2samp, resid_pre, resid_post, "resid_cramervonmises"
            )
            features['resid_cramervonmises_stat'] = cvm_stat
            features['resid_cramervonmises_pvalue'] = cvm_pvalue
        except ImportError:
            features['resid_cramervonmises_stat'] = np.nan
            features['resid_cramervonmises_pvalue'] = np.nan

        # Epps-Singleton (with enhanced sample size checking)
        try:
            from scipy.stats import epps_singleton_2samp
            es_stat, es_pvalue = self._safe_statistical_test(
                epps_singleton_2samp, resid_pre, resid_post, "resid_epps_singleton"
            )
            features['resid_epps_singleton_stat'] = es_stat
            features['resid_epps_singleton_pvalue'] = es_pvalue
        except ImportError:
            features['resid_epps_singleton_stat'] = np.nan
            features['resid_epps_singleton_pvalue'] = np.nan

    def _compute_raw_distribution_distance_features(self, p0: np.ndarray, p1: np.ndarray, features: Dict[str, float]):
        """
        Compute distribution-distance metrics on raw series pre vs post.
        """
        # Record raw lengths
        features['raw_p0_length'] = len(p0)
        features['raw_p1_length'] = len(p1)

        # If too small, fill NaNs
        if len(p0) < 2 or len(p1) < 2:
            for name in [
                'raw_wasserstein', 'raw_energy_distance',
                'raw_ks_stat', 'raw_ks_pvalue',
                'raw_cramervonmises_stat', 'raw_cramervonmises_pvalue',
                'raw_epps_singleton_stat', 'raw_epps_singleton_pvalue'
            ]:
                features[name] = np.nan
            return

        # Wasserstein distance
        features['raw_wasserstein'] = self._safe_distance_metric(
            st.wasserstein_distance, p0, p1, "raw_wasserstein"
        )

        # Energy distance
        features['raw_energy_distance'] = self._safe_distance_metric(
            st.energy_distance, p0, p1, "raw_energy"
        )

        # KS test
        ks_stat, ks_pvalue = self._safe_statistical_test(st.ks_2samp, p0, p1, "raw_ks")
        features['raw_ks_stat'] = ks_stat
        features['raw_ks_pvalue'] = ks_pvalue

        # Cramér-von Mises
        try:
            from scipy.stats import cramervonmises_2samp
            cvm_stat, cvm_pvalue = self._safe_statistical_test(
                cramervonmises_2samp, p0, p1, "raw_cramervonmises"
            )
            features['raw_cramervonmises_stat'] = cvm_stat
            features['raw_cramervonmises_pvalue'] = cvm_pvalue
        except ImportError:
            features['raw_cramervonmises_stat'] = np.nan
            features['raw_cramervonmises_pvalue'] = np.nan

        # Epps-Singleton (with enhanced sample size checking)
        try:
            from scipy.stats import epps_singleton_2samp
            es_stat, es_pvalue = self._safe_statistical_test(
                epps_singleton_2samp, p0, p1, "raw_epps_singleton"
            )
            features['raw_epps_singleton_stat'] = es_stat
            features['raw_epps_singleton_pvalue'] = es_pvalue
        except ImportError:
            features['raw_epps_singleton_stat'] = np.nan
            features['raw_epps_singleton_pvalue'] = np.nan

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """
        Compute all distribution-related features for a single time series.
        """
        features: Dict[str, float] = {}

        # Extract period arrays
        p0, p1 = self._extract_period_values(data)

        # 1) Distribution descriptive & tests
        self._compute_distribution_features(p0, p1, features)

        # 2) Residual-distribution comparison
        self._compute_residual_distribution_features(p0, p1, features)

        # 3) Raw distribution-distance features
        self._compute_raw_distribution_distance_features(p0, p1, features)

        return features
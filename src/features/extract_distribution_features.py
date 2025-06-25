# src/features/extract_distribution_combined_features.py
from typing import Dict, Union, Optional, List
import pandas as pd
import numpy as np
import scipy.stats as st

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class DistributionCombinedFeatureExtractor(BaseFeatureExtractor):
    """
    Extract distribution-based features, residual-distribution features, and raw distribution-distance features
    for a time series split into two periods.
    """

    def __init__(
        self,
        lags: int = 1,
        min_length: Optional[int] = None,
        cache_name: str = 'distribution_combined_features',
        force_recompute: bool = False
    ):
        """
        Args:
            lags: number of lags for AR residual extraction in residual-distribution comparison.
            min_length: minimal length AFTER lags required in each segment for residual tests.
                        If None, defaults to lags + 2.
            cache_name: name for caching features.
            force_recompute: if True, ignore cache and always recompute.
        """
        super().__init__(cache_name=cache_name, force_recompute=force_recompute, lags=lags, min_length=min_length)
        self.lags = lags
        self.min_length = min_length if min_length is not None else lags + 2

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
            try:
                t_stat, p_value = st.ttest_ind(p0, p1)
                features['ttest_stat'] = float(t_stat)
                features['ttest_pvalue'] = float(p_value)
            except Exception:
                features['ttest_stat'] = np.nan
                features['ttest_pvalue'] = np.nan

            # Levene test for variance
            try:
                lev_stat, lev_pvalue = st.levene(p0, p1)
                features['levene_stat'] = float(lev_stat)
                features['levene_pvalue'] = float(lev_pvalue)
            except Exception:
                features['levene_stat'] = np.nan
                features['levene_pvalue'] = np.nan

            # Mann-Whitney U test
            try:
                _, mw_p = st.mannwhitneyu(p0, p1)
                features['mannwhitney_pvalue'] = float(mw_p)
            except Exception:
                features['mannwhitney_pvalue'] = np.nan

            # KS test
            try:
                ks_res = st.ks_2samp(p0, p1)
                features['ks_pvalue'] = float(ks_res.pvalue)
            except Exception:
                features['ks_pvalue'] = np.nan
        else:
            features['ttest_stat'] = 0.0
            features['ttest_pvalue'] = 1.0
            features['levene_stat'] = 0.0
            features['levene_pvalue'] = 1.0
            features['mannwhitney_pvalue'] = 1.0
            features['ks_pvalue'] = 1.0

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
        try:
            ks_res = st.ks_2samp(resid_pre, resid_post)
            features['resid_ks_stat'] = float(ks_res.statistic)
            features['resid_ks_pvalue'] = float(ks_res.pvalue)
        except Exception:
            features['resid_ks_stat'] = np.nan
            features['resid_ks_pvalue'] = np.nan

        # Mann-Whitney U
        try:
            mw_res = st.mannwhitneyu(resid_pre, resid_post, alternative='two-sided')
            features['resid_mannwhitney_stat'] = float(mw_res.statistic)
            features['resid_mannwhitney_pvalue'] = float(mw_res.pvalue)
        except Exception:
            features['resid_mannwhitney_stat'] = np.nan
            features['resid_mannwhitney_pvalue'] = np.nan

        # t-test (Welch)
        try:
            t_res = st.ttest_ind(resid_pre, resid_post, equal_var=False, nan_policy='omit')
            features['resid_ttest_stat'] = float(t_res.statistic)
            features['resid_ttest_pvalue'] = float(t_res.pvalue)
        except Exception:
            features['resid_ttest_stat'] = np.nan
            features['resid_ttest_pvalue'] = np.nan

        # Levene
        try:
            lev_res = st.levene(resid_pre, resid_post)
            features['resid_levene_stat'] = float(lev_res.statistic)
            features['resid_levene_pvalue'] = float(lev_res.pvalue)
        except Exception:
            features['resid_levene_stat'] = np.nan
            features['resid_levene_pvalue'] = np.nan

        # Wasserstein
        try:
            w = st.wasserstein_distance(resid_pre, resid_post)
            features['resid_wasserstein'] = float(w)
        except Exception:
            features['resid_wasserstein'] = np.nan

        # Energy distance
        try:
            e = st.energy_distance(resid_pre, resid_post)
            features['resid_energy_distance'] = float(e)
        except Exception:
            features['resid_energy_distance'] = np.nan

        # Cramér-von Mises
        try:
            from scipy.stats import cramervonmises_2samp
            cvm_res = cramervonmises_2samp(resid_pre, resid_post)
            features['resid_cramervonmises_stat'] = float(cvm_res.statistic)
            features['resid_cramervonmises_pvalue'] = float(cvm_res.pvalue)
        except Exception:
            features['resid_cramervonmises_stat'] = np.nan
            features['resid_cramervonmises_pvalue'] = np.nan

        # Epps-Singleton
        try:
            from scipy.stats import epps_singleton_2samp
            es_res = epps_singleton_2samp(resid_pre, resid_post)
            features['resid_epps_singleton_stat'] = float(es_res.statistic)
            features['resid_epps_singleton_pvalue'] = float(es_res.pvalue)
        except Exception:
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

        # Wasserstein
        try:
            w = st.wasserstein_distance(p0, p1)
            features['raw_wasserstein'] = float(w)
        except Exception:
            features['raw_wasserstein'] = np.nan

        # Energy distance
        try:
            e = st.energy_distance(p0, p1)
            features['raw_energy_distance'] = float(e)
        except Exception:
            features['raw_energy_distance'] = np.nan

        # KS test
        try:
            ks_res = st.ks_2samp(p0, p1)
            features['raw_ks_stat'] = float(ks_res.statistic)
            features['raw_ks_pvalue'] = float(ks_res.pvalue)
        except Exception:
            features['raw_ks_stat'] = np.nan
            features['raw_ks_pvalue'] = np.nan

        # Cramér-von Mises
        try:
            from scipy.stats import cramervonmises_2samp
            cvm_res = cramervonmises_2samp(p0, p1)
            features['raw_cramervonmises_stat'] = float(cvm_res.statistic)
            features['raw_cramervonmises_pvalue'] = float(cvm_res.pvalue)
        except Exception:
            features['raw_cramervonmises_stat'] = np.nan
            features['raw_cramervonmises_pvalue'] = np.nan

        # Epps-Singleton
        try:
            from scipy.stats import epps_singleton_2samp
            es_res = epps_singleton_2samp(p0, p1)
            features['raw_epps_singleton_stat'] = float(es_res.statistic)
            features['raw_epps_singleton_pvalue'] = float(es_res.pvalue)
        except Exception:
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

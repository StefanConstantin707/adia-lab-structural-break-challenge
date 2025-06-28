from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
import scipy.stats as st

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class CusumBreakpointFeatureExtractor(BaseFeatureExtractor):
    """
    Extract CUSUM and CUSUMSQ features for a known breakpoint in a lagged regression.
    """

    def __init__(self,
                 lags: int = 1,
                 cache_name: str = 'cusum_breakpoint_features',
                 force_recompute: bool = False):
        """
        Initialize the feature extractor.

        Args:
            lags: number of lags in regression
            cache_name: Name for the cache file.
            force_recompute: If True, ignore cache and recompute all features.
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            lags=lags
        )
        self.lags = lags

    def get_feature_names(self) -> list:
        """Get list of all feature names produced by this extractor."""
        return [
            'n_pre', 'n_post', 'n_full', 'n_params',
            'sigma_hat', 'k_eff', 'u_eff',
            'cusum_stat', 'cusum_pvalue',
            'cusumsq_stat', 'cusumsq_pvalue'
        ]

    def _extract_period_values(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pre- and post-break series arrays."""
        if isinstance(data, TimeSeriesData):
            series_pre = data.period_0_values.astype(float)
            series_post = data.period_1_values.astype(float)
        elif isinstance(data, pd.DataFrame):
            if 'value' not in data.columns or 'period' not in data.columns:
                raise ValueError("DataFrame input must have 'value' and 'period' columns.")
            series_pre = data.loc[data['period'] == 0, 'value'].dropna().astype(float).values
            series_post = data.loc[data['period'] == 1, 'value'].dropna().astype(float).values
        else:
            raise TypeError("Input must be a pandas DataFrame or TimeSeriesData-like instance.")
        return series_pre, series_post

    def _build_lagged_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build lagged design matrix X and vector y. Returns (None, None) if insufficient length."""
        N = len(series)
        lags = self.lags

        if N <= lags:
            return None, None

        y = series[lags:]
        X = np.ones((N - lags, lags + 1), dtype=float)

        for j in range(1, lags + 1):
            X[:, j] = series[lags - j: N - j]

        return X, y

    def _ols_fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fit OLS: returns beta, residuals, rss."""
        XT_X = X.T.dot(X)
        try:
            beta = np.linalg.solve(XT_X, X.T.dot(y))
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XT_X).dot(X.T.dot(y))

        resid = y - X.dot(beta)
        rss = float(resid.dot(resid))

        return beta, resid, rss

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute CUSUM/CUSUMSQ features."""
        features: Dict[str, float] = {}
        series_pre, series_post = self._extract_period_values(data)

        # Build full series
        series_full = np.concatenate([series_pre, series_post])

        # Build lagged matrices
        X_full, y_full = self._build_lagged_matrix(series_full)
        X_pre, y_pre = self._build_lagged_matrix(series_pre)
        X_post, y_post = self._build_lagged_matrix(series_post)

        # Sample sizes
        n_pre = 0 if y_pre is None else len(y_pre)
        n_post = 0 if y_post is None else len(y_post)
        n_full = 0 if y_full is None else len(y_full)

        m = self.lags + 1
        features.update({
            'n_pre': float(n_pre),
            'n_post': float(n_post),
            'n_full': float(n_full),
            'n_params': float(m)
        })

        # Check sufficient data
        if X_full is None or n_full <= m:
            # insufficient data
            nan_vals = {key: np.nan for key in ['sigma_hat','k_eff','u_eff','cusum_stat','cusum_pvalue','cusumsq_stat','cusumsq_pvalue']}
            features.update(nan_vals)
            return features

        # Fit full OLS
        beta_full, resid_full, rss_full = self._ols_fit(X_full, y_full)
        df_resid = n_full - m
        sigma_hat = np.sqrt(rss_full / df_resid) if df_resid > 0 else np.nan
        features['sigma_hat'] = float(sigma_hat) if not np.isnan(sigma_hat) else np.nan

        # Effective break index in residuals
        K = len(series_pre)
        k_eff = K - self.lags
        if not (isinstance(k_eff, (int, np.integer)) and 1 <= k_eff <= n_full - 1):
            nan_vals = {'k_eff': np.nan, 'u_eff': np.nan, 'cusum_stat': np.nan, 'cusum_pvalue': np.nan, 'cusumsq_stat': np.nan, 'cusumsq_pvalue': np.nan}
            features.update(nan_vals)
            return features

        features['k_eff'] = float(k_eff)
        u = k_eff / n_full
        features['u_eff'] = float(u)

        # CUSUM statistic
        resid_pre = resid_full[:k_eff]
        S = float(np.sum(resid_pre))

        if np.isnan(sigma_hat) or sigma_hat <= 0:
            T_cusum = np.nan
            p_cusum = np.nan
        else:
            denom = sigma_hat * np.sqrt(n_full * u * (1 - u))
            if denom <= 0:
                T_cusum = np.nan
                p_cusum = np.nan
            else:
                T_cusum = S / denom
                p_cusum = 2 * (1 - st.norm.cdf(abs(T_cusum)))

        features['cusum_stat'] = float(T_cusum) if not np.isnan(T_cusum) else np.nan
        features['cusum_pvalue'] = float(p_cusum) if not np.isnan(p_cusum) else np.nan

        # CUSUMSQ statistic
        sumsq_pre = float(np.sum(resid_pre**2))
        if rss_full <= 0 or n_full + 2 <= 0:
            Z_cusumsq = np.nan
            p_cusumsq = np.nan
        else:
            C = sumsq_pre / rss_full
            var_C = 2 * u * (1 - u) / (n_full + 2)
            if var_C <= 0:
                Z_cusumsq = np.nan
                p_cusumsq = np.nan
            else:
                Z_cusumsq = (C - u) / np.sqrt(var_C)
                p_cusumsq = 2 * (1 - st.norm.cdf(abs(Z_cusumsq)))

        features['cusumsq_stat'] = float(Z_cusumsq) if not np.isnan(Z_cusumsq) else np.nan
        features['cusumsq_pvalue'] = float(p_cusumsq) if not np.isnan(p_cusumsq) else np.nan
        return features

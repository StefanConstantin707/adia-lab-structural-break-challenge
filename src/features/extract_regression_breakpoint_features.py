from typing import Dict, Union, Tuple, List
import pandas as pd
import numpy as np
import scipy.stats as st

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class RegressionBreakpointFeatureExtractor(BaseFeatureExtractor):
    """
    Extract regression-based breakpoint features (Chow test) from a time series split into two periods.
    """

    def __init__(self,
                 lags: int = 1,
                 cache_name: str = 'regression_breakpoint_features',
                 force_recompute: bool = False):
        """
        Initialize the feature extractor.

        Args:
            lags: Number of lagged terms to include (>=1).
            cache_name: Name for the cache file.
            force_recompute: If True, ignore cache and recompute all features.
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            lags=lags
        )
        self.lags = lags

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        names: List[str] = [
            'n_pre', 'n_post', 'n_full', 'n_params',
            'rss_full', 'rss_pre', 'rss_post',
            'chow_f_stat', 'chow_pvalue',
            'coef_diff_norm'
        ]
        # coefficient names
        for prefix in ['full', 'pre', 'post']:
            for i in range(self.lags + 1):
                names.append(f'coef_{prefix}_{i}')
        return names

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
            raise TypeError("Input must be a pandas DataFrame or TimeSeriesData instance.")
        return series_pre, series_post

    def _build_lagged_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build lagged design matrix X and dependent y. Returns (None, None) if insufficient length."""
        N = len(series)
        lags = self.lags
        if N <= lags:
            return None, None
        y = series[lags:]
        X = np.ones((N - lags, lags + 1), dtype=float)
        for j in range(1, lags + 1):
            X[:, j] = series[lags - j: N - j]
        return X, y

    def _ols_fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit OLS via normal equations; return beta and rss."""
        XT_X = X.T.dot(X)
        try:
            beta = np.linalg.solve(XT_X, X.T.dot(y))
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XT_X).dot(X.T.dot(y))
        resid = y - X.dot(beta)
        rss = float(resid.dot(resid))
        return beta, rss

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute regression breakpoint features for a single time series."""
        features: Dict[str, float] = {}
        series_pre, series_post = self._extract_period_values(data)

        # Build full concatenated series in time order
        series_full = np.concatenate([series_pre, series_post])

        # Build lagged matrices
        X_full, y_full = self._build_lagged_matrix(series_full)
        X_pre, y_pre = self._build_lagged_matrix(series_pre)
        X_post, y_post = self._build_lagged_matrix(series_post)

        # Sample sizes
        n_pre = 0 if y_pre is None else len(y_pre)
        n_post = 0 if y_post is None else len(y_post)
        n_full = 0 if y_full is None else len(y_full)

        features['n_pre'] = float(n_pre)
        features['n_post'] = float(n_post)
        features['n_full'] = float(n_full)
        m = self.lags + 1
        features['n_params'] = float(m)

        # Check for sufficient data
        if X_full is None or X_pre is None or X_post is None:
            # insufficient data: fill NaNs
            features.update({
                'rss_full': np.nan,
                'rss_pre': np.nan,
                'rss_post': np.nan,
                'chow_f_stat': np.nan,
                'chow_pvalue': np.nan,
                'coef_diff_norm': np.nan
            })
            for prefix in ['full', 'pre', 'post']:
                for i in range(m):
                    features[f'coef_{prefix}_{i}'] = np.nan
            return features

        # Fit OLS
        beta_full, rss_full = self._ols_fit(X_full, y_full)
        beta_pre, rss_pre = self._ols_fit(X_pre, y_pre)
        beta_post, rss_post = self._ols_fit(X_post, y_post)

        features['rss_full'] = float(rss_full)
        features['rss_pre'] = float(rss_pre)
        features['rss_post'] = float(rss_post)

        # Store coefficients
        for i in range(m):
            features[f'coef_full_{i}'] = float(beta_full[i])
            features[f'coef_pre_{i}'] = float(beta_pre[i])
            features[f'coef_post_{i}'] = float(beta_post[i])

        # Compute Chow F-statistic and p-value
        denom_df = (n_pre + n_post - 2 * m)
        num = rss_full - (rss_pre + rss_post)

        if denom_df <= 0 or num <= 0:
            features['chow_f_stat'] = np.nan
            features['chow_pvalue'] = np.nan
        else:
            F_stat = (num / m) / ((rss_pre + rss_post) / denom_df)
            p_value = st.f.sf(F_stat, m, denom_df)
            features['chow_f_stat'] = float(F_stat)
            features['chow_pvalue'] = float(p_value)

        # Norm of coefficient difference
        diff = beta_pre - beta_post
        features['coef_diff_norm'] = float(np.linalg.norm(diff))
        return features

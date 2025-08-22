from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
import scipy.stats as st

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class CusumBreakpointFeatureExtractor(BaseFeatureExtractor):
    """
    Extract CUSUM and CUSUMSQ features for a known breakpoint in a lagged regression.
    Now includes assumption testing and robustness features.
    """

    def __init__(self,
                 lags: int = 1,
                 cache_name: str = 'cusum_breakpoint_features',
                 force_recompute: bool = False,
                 no_cache: bool = False,
                 test_assumptions: bool = True,
                 adaptive_lags: bool = False,
                 max_lags: int = 15,
                 pacf_alpha: float = 0.05):
        """
        Initialize the feature extractor.

        Args:
            lags: number of lags in regression (used if adaptive_lags=False)
            cache_name: Name for the cache file.
            force_recompute: If True, ignore cache and recompute all features.
            no_cache: If True, skip all cache operations.
            test_assumptions: If True, include assumption testing features.
            adaptive_lags: If True, use PACF to determine optimal lags for each series
            max_lags: Maximum number of lags to consider for PACF analysis
            pacf_alpha: Significance level for PACF confidence intervals
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            lags=lags,
            no_cache=no_cache,
            test_assumptions=test_assumptions,
            adaptive_lags=adaptive_lags,
            max_lags=max_lags,
            pacf_alpha=pacf_alpha
        )
        self.lags = lags
        self.test_assumptions = test_assumptions
        self.adaptive_lags = adaptive_lags
        self.max_lags = max_lags
        self.pacf_alpha = pacf_alpha

    def get_feature_names(self) -> list:
        """Get list of all feature names produced by this extractor."""
        base_features = [
            'n_pre', 'n_post', 'n_full', 'n_params',
            'sigma_hat', 'k_eff', 'u_eff',
            'cusum_stat', 'cusum_pvalue',
            'cusumsq_stat', 'cusumsq_pvalue'
        ]

        enhanced_features = [
            'cusum_log_pvalue', 'cusum_significant',
            'cusumsq_log_pvalue', 'cusumsq_significant',
            'cusum_abs_stat', 'cusumsq_abs_stat'
        ]

        if self.adaptive_lags:
            enhanced_features.append('optimal_lags')

        if self.test_assumptions:
            assumption_features = [
                'residual_normality_pval', 'residual_normality_stat',
                'residual_skewness', 'residual_kurtosis', 'residual_excess_kurtosis',
                'assumptions_valid', 'heavy_tails', 'highly_skewed',
                'ljungbox_pval', 'ljungbox_stat', 'serial_correlation',
                'variance_ratio_pval', 'heteroscedasticity'
            ]
            return base_features + enhanced_features + assumption_features

        return base_features + enhanced_features

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

    def _count_significant_pacf_lags(self, series: np.ndarray) -> int:
        """Count number of significant PACF lags using confidence intervals."""
        try:
            from statsmodels.tsa.stattools import pacf

            # Ensure we have enough data for PACF analysis
            min_length = max(self.max_lags + 5, 20)
            if len(series) < min_length:
                return max(1, min(3, len(series) // 4))  # Fallback to small number

            # Compute PACF with confidence intervals
            vals, conf_int = pacf(series, nlags=self.max_lags, alpha=self.pacf_alpha)

            sig_lags = 0
            # Skip lag=0 (always 1.0), check lags 1 to max_lags
            for lag in range(1, len(vals)):
                if lag < len(conf_int):
                    lo, hi = conf_int[lag]
                    # Check if confidence interval excludes zero
                    if hi < 0 or lo > 0:
                        sig_lags += 1

            # Return at least 1 lag, at most max_lags
            return max(1, min(sig_lags, self.max_lags))

        except Exception as e:
            # If PACF computation fails, return a reasonable default
            return max(1, min(3, len(series) // 10))

    def _determine_optimal_lags(self, series_full: np.ndarray) -> int:
        """Determine optimal number of lags for this series."""
        if self.adaptive_lags:
            return self._count_significant_pacf_lags(series_full)
        else:
            return self.lags

    def _build_lagged_matrix(self, series: np.ndarray, lags: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Build lagged design matrix X and vector y. Returns (None, None) if insufficient length."""
        N = len(series)
        if lags is None:
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

    def _test_residual_assumptions(self, residuals: np.ndarray, X: np.ndarray = None) -> Dict[str, float]:
        """Test assumptions on residuals and return diagnostic features."""
        features = {}

        try:
            # Normality tests
            jb_stat, jb_pval = st.jarque_bera(residuals)
            features['residual_normality_stat'] = float(jb_stat)
            features['residual_normality_pval'] = float(jb_pval)

            # Residual characteristics
            skewness = st.skew(residuals)
            kurtosis = st.kurtosis(residuals)  # Excess kurtosis (normal = 0)

            features['residual_skewness'] = float(skewness)
            features['residual_kurtosis'] = float(kurtosis + 3)  # Convert to raw kurtosis
            features['residual_excess_kurtosis'] = float(kurtosis)

            # Assumption compliance flags
            features['assumptions_valid'] = float(jb_pval > 0.05)
            features['heavy_tails'] = float(kurtosis > 3)  # Excess kurtosis > 3
            features['highly_skewed'] = float(abs(skewness) > 1)

            # Serial correlation test (Ljung-Box on first 5 lags)
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(residuals, lags=min(5, len(residuals) // 4), return_df=True)
                lb_pval = lb_result['lb_pvalue'].iloc[-1]  # Last lag p-value
                lb_stat = lb_result['lb_stat'].iloc[-1]

                features['ljungbox_pval'] = float(lb_pval)
                features['ljungbox_stat'] = float(lb_stat)
                features['serial_correlation'] = float(lb_pval < 0.05)
            except:
                features['ljungbox_pval'] = np.nan
                features['ljungbox_stat'] = np.nan
                features['serial_correlation'] = np.nan

            # Simple variance ratio test (first half vs second half)
            n = len(residuals)
            if n > 20:
                half = n // 2
                var1 = np.var(residuals[:half])
                var2 = np.var(residuals[half:])

                if var1 > 0 and var2 > 0:
                    f_stat = var2 / var1
                    # Two-tailed F-test
                    f_pval = 2 * min(st.f.cdf(f_stat, half - 1, n - half - 1),
                                     1 - st.f.cdf(f_stat, half - 1, n - half - 1))

                    features['variance_ratio_pval'] = float(f_pval)
                    features['heteroscedasticity'] = float(f_pval < 0.05)
                else:
                    features['variance_ratio_pval'] = np.nan
                    features['heteroscedasticity'] = np.nan
            else:
                features['variance_ratio_pval'] = np.nan
                features['heteroscedasticity'] = np.nan

        except Exception as e:
            # Fill with NaN if assumption testing fails
            assumption_keys = [
                'residual_normality_stat', 'residual_normality_pval',
                'residual_skewness', 'residual_kurtosis', 'residual_excess_kurtosis',
                'assumptions_valid', 'heavy_tails', 'highly_skewed',
                'ljungbox_pval', 'ljungbox_stat', 'serial_correlation',
                'variance_ratio_pval', 'heteroscedasticity'
            ]
            for key in assumption_keys:
                features[key] = np.nan

        return features

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute CUSUM/CUSUMSQ features with enhanced robustness measures."""
        features: Dict[str, float] = {}
        series_pre, series_post = self._extract_period_values(data)

        # Build full series
        series_full = np.concatenate([series_pre, series_post])

        # Determine optimal number of lags for this series
        optimal_lags = self._determine_optimal_lags(series_full)

        # Add optimal lags as a feature if using adaptive lags
        if self.adaptive_lags:
            features['optimal_lags'] = float(optimal_lags)

        # Build lagged matrices using optimal lags
        X_full, y_full = self._build_lagged_matrix(series_full, lags=optimal_lags)
        X_pre, y_pre = self._build_lagged_matrix(series_pre, lags=optimal_lags)
        X_post, y_post = self._build_lagged_matrix(series_post, lags=optimal_lags)

        # Sample sizes
        n_pre = 0 if y_pre is None else len(y_pre)
        n_post = 0 if y_post is None else len(y_post)
        n_full = 0 if y_full is None else len(y_full)

        m = optimal_lags + 1  # Use optimal lags instead of self.lags
        features.update({
            'n_pre': float(n_pre),
            'n_post': float(n_post),
            'n_full': float(n_full),
            'n_params': float(m)
        })

        # Check sufficient data
        if X_full is None or n_full <= m:
            # insufficient data - fill all features with NaN
            nan_keys = ['sigma_hat', 'k_eff', 'u_eff', 'cusum_stat', 'cusum_pvalue', 'cusumsq_stat', 'cusumsq_pvalue']
            nan_keys += ['cusum_log_pvalue', 'cusum_significant', 'cusumsq_log_pvalue', 'cusumsq_significant']
            nan_keys += ['cusum_abs_stat', 'cusumsq_abs_stat']

            if self.test_assumptions:
                nan_keys += [
                    'residual_normality_pval', 'residual_normality_stat',
                    'residual_skewness', 'residual_kurtosis', 'residual_excess_kurtosis',
                    'assumptions_valid', 'heavy_tails', 'highly_skewed',
                    'ljungbox_pval', 'ljungbox_stat', 'serial_correlation',
                    'variance_ratio_pval', 'heteroscedasticity'
                ]

            nan_vals = {key: np.nan for key in nan_keys}
            features.update(nan_vals)
            return features

        # Fit full OLS
        beta_full, resid_full, rss_full = self._ols_fit(X_full, y_full)
        df_resid = n_full - m
        sigma_hat = np.sqrt(rss_full / df_resid) if df_resid > 0 else np.nan
        features['sigma_hat'] = float(sigma_hat) if not np.isnan(sigma_hat) else np.nan

        # Test assumptions on residuals if requested
        if self.test_assumptions and len(resid_full) > 10:
            assumption_features = self._test_residual_assumptions(resid_full, X_full)
            features.update(assumption_features)

        # Effective break index in residuals
        K = len(series_pre)
        k_eff = K - optimal_lags  # Use optimal lags instead of self.lags
        if not (isinstance(k_eff, (int, np.integer)) and 1 <= k_eff <= n_full - 1):
            nan_keys = ['k_eff', 'u_eff', 'cusum_stat', 'cusum_pvalue', 'cusumsq_stat', 'cusumsq_pvalue']
            nan_keys += ['cusum_log_pvalue', 'cusum_significant', 'cusumsq_log_pvalue', 'cusumsq_significant']
            nan_keys += ['cusum_abs_stat', 'cusumsq_abs_stat']
            nan_vals = {key: np.nan for key in nan_keys}
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

        # Enhanced CUSUM features
        features['cusum_abs_stat'] = float(abs(T_cusum)) if not np.isnan(T_cusum) else np.nan
        features['cusum_log_pvalue'] = float(-np.log10(p_cusum)) if not np.isnan(p_cusum) and p_cusum > 0 else np.nan
        features['cusum_significant'] = float(p_cusum < 0.05) if not np.isnan(p_cusum) else np.nan

        # CUSUMSQ statistic
        sumsq_pre = float(np.sum(resid_pre ** 2))
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

        # Enhanced CUSUMSQ features
        features['cusumsq_abs_stat'] = float(abs(Z_cusumsq)) if not np.isnan(Z_cusumsq) else np.nan
        features['cusumsq_log_pvalue'] = float(-np.log10(p_cusumsq)) if not np.isnan(p_cusumsq) and p_cusumsq > 0 else np.nan
        features['cusumsq_significant'] = float(p_cusumsq < 0.05) if not np.isnan(p_cusumsq) else np.nan

        return features
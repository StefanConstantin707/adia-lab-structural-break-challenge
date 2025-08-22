# src/features/extract_advanced_volatility_features.py
from typing import Dict, Union, List
import pandas as pd
import numpy as np
import warnings
from scipy import stats

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class AdvancedVolatilityBreakFeatureExtractor(BaseFeatureExtractor):
    """
    Extract advanced features to distinguish structural breaks from natural volatility clustering.

    This extractor addresses the core problem: CUSUM-squared detects both real structural breaks
    AND natural financial volatility patterns (GARCH effects, volatility clustering).

    Features include:
    - GARCH-conditioned break tests
    - Volatility persistence analysis
    - Multi-scale break detection
    - Break magnitude and characteristics
    """

    def __init__(self,
                 garch_max_p: int = 2,
                 garch_max_q: int = 2,
                 volatility_window: int = 20,
                 scales: List[int] = None,
                 cache_name: str = 'advanced_volatility_features',
                 check_same: bool = True,
                 force_recompute: bool = False,
                 no_cache: bool = False):
        """
        Initialize the advanced volatility break feature extractor.

        Args:
            garch_max_p: Maximum GARCH(p,q) p parameter to try
            garch_max_q: Maximum GARCH(p,q) q parameter to try
            volatility_window: Window size for rolling volatility calculations
            scales: List of time scales for multi-scale analysis
            cache_name: Name for the cache file
            force_recompute: If True, ignore cache and recompute all features
            no_cache: If True, skip all cache operations
        """
        if scales is None:
            scales = [5, 10, 20]

        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            no_cache=no_cache,
            check_same=check_same,
            garch_max_p=garch_max_p,
            garch_max_q=garch_max_q,
            volatility_window=volatility_window,
            scales=scales
        )

        self.garch_max_p = garch_max_p
        self.garch_max_q = garch_max_q
        self.volatility_window = volatility_window
        self.scales = scales

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names produced by this extractor."""
        base_features = [
            # GARCH-conditioned features
            'garch_fitted', 'garch_aic', 'garch_bic',
            'garch_param_break', 'garch_lr_stat', 'garch_lr_pvalue',
            'garch_resid_ks_stat', 'garch_resid_ks_pvalue',
            'garch_resid_significant',

            # Volatility persistence features
            'vol_autocorr_lag1', 'vol_autocorr_lag5', 'vol_autocorr_lag10',
            'vol_mean_reversion_pre', 'vol_mean_reversion_post',
            'vol_mean_reversion_change', 'vol_mean_reversion_break',
            'high_vol_regime_duration', 'vol_regime_stability',

            # Break characteristics
            'break_magnitude', 'break_magnitude_log', 'break_gradualness',
            'break_reversibility', 'break_persistence_score',

            # Multi-scale features
            'break_consistency_score', 'dominant_break_scale'
        ]

        # Add scale-specific features
        scale_features = []
        for scale in self.scales:
            scale_features.extend([
                f'scale_{scale}_break_detected',
                f'scale_{scale}_effect_size',
                f'scale_{scale}_pvalue'
            ])

        return base_features + scale_features

    def _fit_garch_model(self, series: np.ndarray) -> Dict[str, float]:
        """Fit GARCH model and return model statistics."""
        try:
            from arch import arch_model

            # Remove mean for GARCH fitting
            demeaned_series = series - np.mean(series)

            best_aic = np.inf
            best_model = None

            # Try different GARCH specifications
            for p in range(1, self.garch_max_p + 1):
                for q in range(1, self.garch_max_q + 1):
                    try:
                        garch = arch_model(
                            demeaned_series,
                            vol='GARCH',
                            p=p,
                            q=q,
                            mean='Zero',
                            rescale=False
                        )
                        fitted = garch.fit(disp='off', show_warning=False)

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                    except:
                        continue

            if best_model is not None:
                return {
                    'fitted': True,
                    'model': best_model,
                    'aic': float(best_model.aic),
                    'bic': float(best_model.bic),
                    'loglikelihood': float(best_model.loglikelihood)
                }
            else:
                return {'fitted': False}

        except ImportError:
            warnings.warn("arch package not available. GARCH features will be NaN.")
            return {'fitted': False}
        except Exception:
            return {'fitted': False}

    def _test_garch_parameter_stability(self, series: np.ndarray, split_point: int) -> Dict[str, float]:
        """Test if GARCH parameters are stable across split point."""
        features = {
            'garch_param_break': np.nan,
            'garch_lr_stat': np.nan,
            'garch_lr_pvalue': np.nan
        }

        try:
            pre_series = series[:split_point]
            post_series = series[split_point:]

            # Need sufficient data for GARCH fitting
            if len(pre_series) < 20 or len(post_series) < 20:
                return features

            # Fit GARCH to each period
            pre_garch = self._fit_garch_model(pre_series)
            post_garch = self._fit_garch_model(post_series)
            full_garch = self._fit_garch_model(series)

            if (pre_garch['fitted'] and post_garch['fitted'] and full_garch['fitted']):
                # Likelihood ratio test
                ll_separate = pre_garch['loglikelihood'] + post_garch['loglikelihood']
                ll_full = full_garch['loglikelihood']

                lr_stat = 2 * (ll_separate - ll_full)

                # Approximate p-value (df = number of GARCH parameters)
                df = 3  # omega, alpha, beta
                p_value = 1 - stats.chi2.cdf(lr_stat, df) if lr_stat > 0 else 1.0

                features.update({
                    'garch_param_break': float(p_value < 0.05),
                    'garch_lr_stat': float(lr_stat),
                    'garch_lr_pvalue': float(p_value)
                })
        except:
            pass

        return features

    def _test_garch_residuals(self, garch_model, series: np.ndarray, split_point: int) -> Dict[str, float]:
        """Test for breaks in GARCH standardized residuals."""
        features = {
            'garch_resid_ks_stat': np.nan,
            'garch_resid_ks_pvalue': np.nan,
            'garch_resid_significant': np.nan
        }

        try:
            # Get standardized residuals
            std_resid = garch_model['model'].std_resid

            if len(std_resid) > split_point + 10:
                # Adjust split point for residuals (accounting for GARCH lags)
                model_start = len(series) - len(std_resid)
                adj_split = split_point - model_start

                if adj_split > 5 and adj_split < len(std_resid) - 5:
                    pre_resid = std_resid[:adj_split]
                    post_resid = std_resid[adj_split:]

                    # KS test on standardized residuals
                    ks_stat, ks_pval = stats.ks_2samp(pre_resid, post_resid)

                    features.update({
                        'garch_resid_ks_stat': float(ks_stat),
                        'garch_resid_ks_pvalue': float(ks_pval),
                        'garch_resid_significant': float(ks_pval < 0.05)
                    })
        except:
            pass

        return features

    def _analyze_volatility_persistence(self, series: np.ndarray) -> Dict[str, float]:
        """Analyze volatility persistence patterns."""
        features = {
            'vol_autocorr_lag1': np.nan,
            'vol_autocorr_lag5': np.nan,
            'vol_autocorr_lag10': np.nan,
            'high_vol_regime_duration': np.nan,
            'vol_regime_stability': np.nan
        }

        try:
            if len(series) < self.volatility_window * 2:
                return features

            # Rolling volatility
            rolling_vol = pd.Series(series).rolling(self.volatility_window, min_periods=5).std()
            vol_changes = rolling_vol.diff().dropna()

            if len(vol_changes) > 20:
                # Volatility autocorrelations
                features['vol_autocorr_lag1'] = float(vol_changes.autocorr(lag=1))

                if len(vol_changes) > 10:
                    features['vol_autocorr_lag5'] = float(vol_changes.autocorr(lag=5))

                if len(vol_changes) > 20:
                    features['vol_autocorr_lag10'] = float(vol_changes.autocorr(lag=10))

            # Volatility regime analysis
            if len(rolling_vol.dropna()) > 30:
                vol_clean = rolling_vol.dropna()
                high_vol_threshold = vol_clean.quantile(0.75)

                # High volatility regime duration
                high_vol_indicator = (vol_clean > high_vol_threshold).astype(int)
                regime_changes = high_vol_indicator.diff().abs().sum()

                if regime_changes > 0:
                    avg_duration = len(high_vol_indicator) / regime_changes
                    features['high_vol_regime_duration'] = float(avg_duration)
                    features['vol_regime_stability'] = float(1.0 / regime_changes * len(high_vol_indicator))

        except:
            pass

        return features

    def _test_volatility_mean_reversion(self, series: np.ndarray, split_point: int) -> Dict[str, float]:
        """Test for changes in volatility mean reversion properties."""
        features = {
            'vol_mean_reversion_pre': np.nan,
            'vol_mean_reversion_post': np.nan,
            'vol_mean_reversion_change': np.nan,
            'vol_mean_reversion_break': np.nan
        }

        try:
            pre_series = series[:split_point]
            post_series = series[split_point:]

            if len(pre_series) < 30 or len(post_series) < 30:
                return features

            def compute_vol_ar1_coeff(data):
                """Compute AR(1) coefficient for volatility series."""
                vol_series = pd.Series(data).rolling(10, min_periods=5).std().dropna()
                if len(vol_series) < 15:
                    return np.nan

                vol_lagged = vol_series.shift(1).dropna()
                vol_current = vol_series[1:]

                if len(vol_current) < 10:
                    return np.nan

                slope, _, _, _, _ = stats.linregress(vol_lagged, vol_current)
                return slope

            pre_beta = compute_vol_ar1_coeff(pre_series)
            post_beta = compute_vol_ar1_coeff(post_series)

            if not (np.isnan(pre_beta) or np.isnan(post_beta)):
                beta_change = abs(post_beta - pre_beta)

                features.update({
                    'vol_mean_reversion_pre': float(pre_beta),
                    'vol_mean_reversion_post': float(post_beta),
                    'vol_mean_reversion_change': float(beta_change),
                    'vol_mean_reversion_break': float(beta_change > 0.3)
                })

        except:
            pass

        return features

    def _analyze_break_characteristics(self, series: np.ndarray, split_point: int) -> Dict[str, float]:
        """Analyze characteristics of the potential break."""
        features = {
            'break_magnitude': np.nan,
            'break_magnitude_log': np.nan,
            'break_gradualness': np.nan,
            'break_reversibility': np.nan,
            'break_persistence_score': np.nan
        }

        try:
            pre_series = series[:split_point]
            post_series = series[split_point:]

            if len(pre_series) < 10 or len(post_series) < 10:
                return features

            # Break magnitude (volatility ratio)
            pre_vol = np.std(pre_series)
            post_vol = np.std(post_series)

            if pre_vol > 0:
                vol_ratio = post_vol / pre_vol
                features['break_magnitude'] = float(abs(vol_ratio - 1))
                features['break_magnitude_log'] = float(abs(np.log(vol_ratio)))

            # Break gradualness (transition period analysis)
            transition_window = min(10, len(pre_series) // 4, len(post_series) // 4)

            if transition_window >= 3:
                transition_start = max(0, split_point - transition_window)
                transition_end = min(len(series), split_point + transition_window)
                transition_data = series[transition_start:transition_end]

                if len(transition_data) > 3:
                    x = np.arange(len(transition_data))
                    slope, _, _, _, _ = stats.linregress(x, transition_data)
                    features['break_gradualness'] = float(abs(slope))

            # Break reversibility (persistence)
            if len(post_series) >= 20:
                early_post = post_series[:len(post_series) // 2]
                late_post = post_series[len(post_series) // 2:]

                early_vol = np.std(early_post)
                late_vol = np.std(late_post)

                if early_vol > 0:
                    # Compute persistence score
                    early_diff = abs(early_vol - pre_vol)
                    late_diff = abs(late_vol - pre_vol)

                    if early_diff > 0:
                        persistence_score = 1 - (late_diff / early_diff)
                        features['break_persistence_score'] = float(max(0, persistence_score))

                    reversion_ratio = late_diff / early_diff if early_diff > 0 else 1.0
                    features['break_reversibility'] = float(reversion_ratio)

        except:
            pass

        return features

    def _analyze_multiscale_breaks(self, series: np.ndarray, split_point: int) -> Dict[str, float]:
        """Analyze breaks at multiple time scales."""
        features = {
            'break_consistency_score': np.nan,
            'dominant_break_scale': np.nan
        }

        scale_results = []

        try:
            for scale in self.scales:
                scale_features = {
                    f'scale_{scale}_break_detected': np.nan,
                    f'scale_{scale}_effect_size': np.nan,
                    f'scale_{scale}_pvalue': np.nan
                }

                if len(series) < scale * 4:
                    features.update(scale_features)
                    continue

                # Aggregate data at this scale
                def aggregate_at_scale(data, scale_size):
                    n_windows = len(data) // scale_size
                    aggregated = []
                    for i in range(n_windows):
                        window = data[i * scale_size:(i + 1) * scale_size]
                        aggregated.append(np.std(window))
                    return np.array(aggregated)

                # Aggregate full series
                agg_series = aggregate_at_scale(series, scale)
                agg_split = split_point // scale

                if agg_split > 2 and agg_split < len(agg_series) - 2:
                    agg_pre = agg_series[:agg_split]
                    agg_post = agg_series[agg_split:]

                    if len(agg_pre) > 2 and len(agg_post) > 2:
                        # Test for difference at this scale
                        t_stat, p_val = stats.ttest_ind(agg_pre, agg_post)
                        effect_size = abs(np.mean(agg_post) - np.mean(agg_pre)) / np.sqrt(
                            (np.var(agg_pre) + np.var(agg_post)) / 2)

                        scale_features.update({
                            f'scale_{scale}_break_detected': float(p_val < 0.05),
                            f'scale_{scale}_effect_size': float(effect_size),
                            f'scale_{scale}_pvalue': float(p_val)
                        })

                        scale_results.append({
                            'scale': scale,
                            'detected': p_val < 0.05,
                            'effect_size': effect_size,
                            'pvalue': p_val
                        })

                features.update(scale_features)

            # Compute consistency score
            if scale_results:
                detections = [r['detected'] for r in scale_results]
                consistency_score = np.mean(detections)
                features['break_consistency_score'] = float(consistency_score)

                # Find dominant scale (largest effect size)
                if any(r['detected'] for r in scale_results):
                    detected_scales = [r for r in scale_results if r['detected']]
                    dominant_scale = max(detected_scales, key=lambda x: x['effect_size'])['scale']
                    features['dominant_break_scale'] = float(dominant_scale)

        except:
            pass

        return features

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute all advanced volatility break features."""
        features = {}

        try:
            # Extract data
            if isinstance(data, TimeSeriesData):
                series = data.values
                split_point = data.boundary_point
            else:
                # Handle DataFrame input
                series = data['value'].values
                # Find split point
                periods = data['period'].values
                split_point = np.where(periods == 1)[0][0] if 1 in periods else len(periods)

            # Ensure sufficient data
            if len(series) < 50 or split_point < 20 or split_point > len(series) - 20:
                # Fill all features with NaN for insufficient data
                for feature_name in self.get_feature_names():
                    features[feature_name] = np.nan
                return features

            # 1. GARCH analysis
            garch_result = self._fit_garch_model(series)
            features['garch_fitted'] = float(garch_result['fitted'])

            if garch_result['fitted']:
                features['garch_aic'] = garch_result['aic']
                features['garch_bic'] = garch_result['bic']

                # GARCH parameter stability test
                garch_stability = self._test_garch_parameter_stability(series, split_point)
                features.update(garch_stability)

                # GARCH residual analysis
                garch_residual = self._test_garch_residuals(garch_result, series, split_point)
                features.update(garch_residual)
            else:
                features.update({
                    'garch_aic': np.nan,
                    'garch_bic': np.nan,
                    'garch_param_break': np.nan,
                    'garch_lr_stat': np.nan,
                    'garch_lr_pvalue': np.nan,
                    'garch_resid_ks_stat': np.nan,
                    'garch_resid_ks_pvalue': np.nan,
                    'garch_resid_significant': np.nan
                })

            # 2. Volatility persistence analysis
            persistence_features = self._analyze_volatility_persistence(series)
            features.update(persistence_features)

            # 3. Volatility mean reversion analysis
            mean_reversion_features = self._test_volatility_mean_reversion(series, split_point)
            features.update(mean_reversion_features)

            # 4. Break characteristics
            break_char_features = self._analyze_break_characteristics(series, split_point)
            features.update(break_char_features)

            # 5. Multi-scale analysis
            multiscale_features = self._analyze_multiscale_breaks(series, split_point)
            features.update(multiscale_features)

        except Exception as e:
            warnings.warn(f"Error in advanced volatility feature extraction: {e}")
            # Fill all features with NaN on error
            for feature_name in self.get_feature_names():
                features[feature_name] = np.nan

        return features
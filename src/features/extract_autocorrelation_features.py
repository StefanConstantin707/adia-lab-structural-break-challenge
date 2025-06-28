from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class AutocorrelationFeatureExtractor(BaseFeatureExtractor):
    """
    Extract autocorrelation and partial autocorrelation features from a time series split into two periods.
    """

    def __init__(self,
                 max_lag: int = 100,
                 cache_name: str = 'autocorrelation_features',
                 force_recompute: bool = False):
        """
        Initialize the feature extractor.

        Args:
            max_lag: Maximum lag to consider for ACF/PACF.
            cache_name: Name for the cache file.
            force_recompute: If True, ignore cache and recompute all features.
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            max_lag=max_lag
        )
        self.max_lag = max_lag

    def get_feature_names(self) -> list:
        """Get list of all feature names produced by this extractor."""
        names = []
        # Features for periods 0 and 1
        for p in [0, 1]:
            prefix = f'p{p}'
            for lag in [1, 2, 5, 10]:
                names.append(f'{prefix}_acf_lag{lag}')
                names.append(f'{prefix}_pacf_lag{lag}')
            names.extend([
                f'{prefix}_acf_sum_abs',
                f'{prefix}_acf_mean',
                f'{prefix}_acf_std',
                f'{prefix}_acf_first_zero',
                f'{prefix}_num_sig_acf_lags',
                f'{prefix}_lb_stat_lag10',
                f'{prefix}_lb_pvalue_lag10'
            ])
        # Difference/distance features
        diff_names = [
            'diff_acf_euclid', 'diff_acf_l1', 'diff_acf_sum_abs',
            'diff_pacf_euclid', 'diff_pacf_l1', 'diff_pacf_sum_abs',
            'diff_acf_first_zero', 'diff_num_sig_acf_lags',
            'diff_lb_stat_lag10', 'diff_lb_pvalue_lag10'
        ]
        names.extend(diff_names)
        return names

    def _extract_period_values(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract values for period 0 and 1 from input data."""
        if isinstance(data, TimeSeriesData):
            vals0 = data.period_0_values
            vals1 = data.period_1_values
        elif isinstance(data, pd.DataFrame):
            if 'value' not in data.columns or 'period' not in data.columns:
                raise ValueError("DataFrame input must have 'value' and 'period' columns.")
            vals0 = data.loc[data['period'] == 0, 'value'].dropna().values
            vals1 = data.loc[data['period'] == 1, 'value'].dropna().values
        else:
            raise TypeError("Input must be a pandas DataFrame or TimeSeriesData instance.")
        return vals0, vals1

    def _compute_acf_pacf(self, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ACF and PACF arrays with fallback."""
        n = len(vals)
        nlags = min(self.max_lag, n - 1)
        try:
            acf_vals = acf(vals, nlags=nlags, fft=True, missing='drop')
        except Exception:
            acf_vals = acf(vals, nlags=nlags, fft=False, missing='drop')
        try:
            pacf_vals = pacf(vals, nlags=nlags, method='ywunbiased')
        except Exception:
            pacf_vals = pacf(vals, nlags=nlags, method='ld')
        return acf_vals, pacf_vals

    def _compute_segment_feats(self, vals: np.ndarray, prefix: str) -> Dict[str, Union[float, np.ndarray]]:
        """Compute ACF/PACF features for a single segment."""
        seg_feats: Dict[str, Union[float, np.ndarray]] = {}
        n = len(vals)
        # Handle short series
        if n < 3:
            for lag in [1, 2, 5, 10]:
                seg_feats[f"{prefix}_acf_lag{lag}"] = 0.0
                seg_feats[f"{prefix}_pacf_lag{lag}"] = 0.0
            seg_feats[f"{prefix}_acf_sum_abs"] = 0.0
            seg_feats[f"{prefix}_acf_mean"] = 0.0
            seg_feats[f"{prefix}_acf_std"] = 0.0
            seg_feats[f"{prefix}_acf_first_zero"] = float(self.max_lag)
            seg_feats[f"{prefix}_num_sig_acf_lags"] = 0.0
            seg_feats[f"{prefix}_lb_stat_lag10"] = np.nan
            seg_feats[f"{prefix}_lb_pvalue_lag10"] = np.nan
            seg_feats[f"{prefix}_acf_full"] = np.zeros(1)
            seg_feats[f"{prefix}_pacf_full"] = np.zeros(1)
            return seg_feats

        # Compute ACF and PACF
        acf_vals, pacf_vals = self._compute_acf_pacf(vals)
        nlags = len(acf_vals) - 1

        # Lag-specific features
        for lag in [1, 2, 5, 10]:
            if lag <= nlags:
                seg_feats[f"{prefix}_acf_lag{lag}"] = float(acf_vals[lag])
                seg_feats[f"{prefix}_pacf_lag{lag}"] = float(pacf_vals[lag])
            else:
                seg_feats[f"{prefix}_acf_lag{lag}"] = 0.0
                seg_feats[f"{prefix}_pacf_lag{lag}"] = 0.0

        # Summary over ACF lags 1..nlags
        acf_slice = acf_vals[1: nlags+1]
        seg_feats[f"{prefix}_acf_sum_abs"] = float(np.sum(np.abs(acf_slice)))
        seg_feats[f"{prefix}_acf_mean"] = float(np.mean(acf_slice))
        seg_feats[f"{prefix}_acf_std"] = float(np.std(acf_slice))

        # First zero-crossing
        zero_crossings = np.where(np.diff(np.sign(acf_slice)) != 0)[0]
        if len(zero_crossings) > 0:
            seg_feats[f"{prefix}_acf_first_zero"] = float(zero_crossings[0] + 1)
        else:
            seg_feats[f"{prefix}_acf_first_zero"] = float(nlags)

        # Significant lags count
        conf_bound = 1.96 / np.sqrt(n)
        num_sig = np.sum(np.abs(acf_slice) > conf_bound)
        seg_feats[f"{prefix}_num_sig_acf_lags"] = float(num_sig)

        # Ljung-Box test
        lb_lag = 10
        if n > lb_lag:
            try:
                lb_res = acorr_ljungbox(vals, lags=[lb_lag], return_df=True)
                seg_feats[f"{prefix}_lb_stat_lag10"] = float(lb_res['lb_stat'].iloc[0])
                seg_feats[f"{prefix}_lb_pvalue_lag10"] = float(lb_res['lb_pvalue'].iloc[0])
            except Exception:
                seg_feats[f"{prefix}_lb_stat_lag10"] = np.nan
                seg_feats[f"{prefix}_lb_pvalue_lag10"] = np.nan
        else:
            seg_feats[f"{prefix}_lb_stat_lag10"] = np.nan
            seg_feats[f"{prefix}_lb_pvalue_lag10"] = np.nan

        # Store full arrays for distance computations
        seg_feats[f"{prefix}_acf_full"] = acf_vals
        seg_feats[f"{prefix}_pacf_full"] = pacf_vals

        return seg_feats

    def _compute_diff_feats(self, feats0: Dict[str, Union[float, np.ndarray]],
                            feats1: Dict[str, Union[float, np.ndarray]]) -> Dict[str, float]:
        """Compute difference and distance features between two segments."""
        features: Dict[str, float] = {}
        # ACF distances
        acf0 = feats0.get("p0_acf_full")
        acf1 = feats1.get("p1_acf_full")

        if acf0 is not None and acf1 is not None:
            len0, len1 = len(acf0), len(acf1)
            L = max(len0, len1)
            a0 = np.pad(acf0, (0, L - len0), 'constant')
            a1 = np.pad(acf1, (0, L - len1), 'constant')
            features['diff_acf_euclid'] = float(np.linalg.norm(a1 - a0))
            features['diff_acf_l1'] = float(np.sum(np.abs(a1 - a0)))
            features['diff_acf_sum_abs'] = feats1.get('p1_acf_sum_abs', 0.0) - feats0.get('p0_acf_sum_abs', 0.0)
        else:
            features['diff_acf_euclid'] = 0.0
            features['diff_acf_l1'] = 0.0
            features['diff_acf_sum_abs'] = 0.0

        # PACF distances
        pacf0 = feats0.get("p0_pacf_full")
        pacf1 = feats1.get("p1_pacf_full")

        if pacf0 is not None and pacf1 is not None:
            len0, len1 = len(pacf0), len(pacf1)
            L = max(len0, len1)
            p0 = np.pad(pacf0, (0, L - len0), 'constant')
            p1 = np.pad(pacf1, (0, L - len1), 'constant')
            features['diff_pacf_euclid'] = float(np.linalg.norm(p1 - p0))
            features['diff_pacf_l1'] = float(np.sum(np.abs(p1 - p0)))
            features['diff_pacf_sum_abs'] = float(np.sum(np.abs(p1[1:]) - np.abs(p0[1:]))) if L > 1 else 0.0
        else:
            features['diff_pacf_euclid'] = 0.0
            features['diff_pacf_l1'] = 0.0
            features['diff_pacf_sum_abs'] = 0.0

        # First zero-crossing difference
        fz0 = feats0.get('p0_acf_first_zero', self.max_lag)
        fz1 = feats1.get('p1_acf_first_zero', self.max_lag)
        features['diff_acf_first_zero'] = float(fz1 - fz0)

        # Significant lags count difference
        ns0 = feats0.get('p0_num_sig_acf_lags', 0.0)
        ns1 = feats1.get('p1_num_sig_acf_lags', 0.0)
        features['diff_num_sig_acf_lags'] = float(ns1 - ns0)

        # Ljung-Box difference
        lb0_stat = feats0.get('p0_lb_stat_lag10', np.nan)
        lb1_stat = feats1.get('p1_lb_stat_lag10', np.nan)
        lb0_p = feats0.get('p0_lb_pvalue_lag10', np.nan)
        lb1_p = feats1.get('p1_lb_pvalue_lag10', np.nan)

        try:
            features['diff_lb_stat_lag10'] = float(lb1_stat - lb0_stat)
        except Exception:
            features['diff_lb_stat_lag10'] = np.nan
        try:
            features['diff_lb_pvalue_lag10'] = float(lb1_p - lb0_p)
        except Exception:
            features['diff_lb_pvalue_lag10'] = np.nan
        return features

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute all autocorrelation features for a single time series."""
        features: Dict[str, float] = {}
        period_0, period_1 = self._extract_period_values(data)

        if len(period_0) < 3 or len(period_1) < 3:
            return self._get_default_features()

        # Compute per-segment features
        feats0 = self._compute_segment_feats(period_0, prefix="p0")
        feats1 = self._compute_segment_feats(period_1, prefix="p1")

        # Transfer scalar features
        for k, v in feats0.items():
            if isinstance(v, np.ndarray):
                continue
            features[k] = v

        for k, v in feats1.items():
            if isinstance(v, np.ndarray):
                continue
            features[k] = v

        # Compute and merge diff features
        diff_feats = self._compute_diff_feats(feats0, feats1)
        features.update(diff_feats)

        return features

    def _get_default_features(self, throw_error: bool = True) -> Dict[str, float]:
        """Return default features when extraction fails."""
        default_features = {}

        if throw_error:
            raise Exception(f"{self.__class__.__name__.lower()} feature extraction failed at id: {self.series_id}")

        # Add default values for all expected features
        feature_names = self.get_feature_names()

        for name in feature_names:
            if 'pvalue' in name:
                default_features[name] = 1.0
            elif 'ratio' in name:
                default_features[name] = 1.0
            elif 'length' in name:
                default_features[name] = 0
            else:
                default_features[name] = 0.0

        return default_features

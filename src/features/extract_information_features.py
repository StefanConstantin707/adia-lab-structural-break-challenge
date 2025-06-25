import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple
from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy


class InformationFeatureExtractor(BaseFeatureExtractor):
    """
    Extract information-theoretic features: mutual information at lags and permutation entropy.
    """
    def __init__(self,
                 lags: List[int] = [1, 2, 5],
                 perm_order: int = 3,
                 perm_delay: int = 1,
                 cache_name: str = 'information_features',
                 force_recompute: bool = False):
        """
        Initialize information-theoretic feature extractor.

        Args:
            lags: list of lags for mutual information
            perm_order: order for permutation entropy
            perm_delay: delay for permutation entropy
            cache_name: cache identifier
            force_recompute: ignore cache if True
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            lags=lags,
            perm_order=perm_order,
            perm_delay=perm_delay
        )
        self.lags = lags
        self.perm_order = perm_order
        self.perm_delay = perm_delay

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        names = []
        for lag in self.lags:
            names.extend([
                f'p0_mi_lag{lag}',
                f'p1_mi_lag{lag}',
                f'mi_lag{lag}_diff'
            ])
        names.extend([
            'p0_perm_entropy', 'p1_perm_entropy', 'perm_entropy_diff'
        ])
        return names

    def _extract_period_values(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract arrays for period 0 and 1."""
        if isinstance(data, TimeSeriesData):
            period_0 = data.period_0_values
            period_1 = data.period_1_values
        elif isinstance(data, pd.DataFrame):
            if 'value' not in data.columns or 'period' not in data.columns:
                raise ValueError("DataFrame input must have 'value' and 'period' columns.")
            period_0 = data.loc[data['period'] == 0, 'value'].dropna().astype(float).values
            period_1 = data.loc[data['period'] == 1, 'value'].dropna().astype(float).values
        else:
            raise TypeError("Input must be a pandas DataFrame or TimeSeriesData instance.")
        return period_0, period_1

    def _mutual_info_lags(self, segment: np.ndarray) -> Dict[int, float]:
        """Compute mutual information between segment and its lagged versions."""
        mi_features: Dict[int, float] = {}
        for lag in self.lags:
            if len(segment) > lag + 1:
                current = segment[lag:]
                lagged = segment[:-lag]
                try:
                    mi = mutual_info_regression(lagged.reshape(-1, 1), current, discrete_features=False)[0]
                    mi_features[lag] = float(mi)
                except Exception:
                    mi_features[lag] = 0.0
            else:
                mi_features[lag] = 0.0
        return mi_features

    def _permutation_entropy(self, ts: np.ndarray) -> float:
        """Calculate permutation entropy for a time series."""
        order = self.perm_order
        delay = self.perm_delay
        n = len(ts)
        if n < order + delay:
            return 0.0
        patterns = []
        for i in range(n - delay * (order - 1)):
            window = ts[i:(i + delay * (order - 1) + 1):delay]
            # get ordinal pattern
            sorted_idx = tuple(np.argsort(window))
            patterns.append(sorted_idx)
        if not patterns:
            return 0.0
        unique, counts = np.unique(patterns, return_counts=True)
        probs = counts / counts.sum()
        return float(entropy(probs, base=2))

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute information-theoretic features for a single time series."""
        features: Dict[str, float] = {}
        period_0, period_1 = self._extract_period_values(data)
        # Mutual information
        mi0 = self._mutual_info_lags(period_0)
        mi1 = self._mutual_info_lags(period_1)
        for lag in self.lags:
            features[f'p0_mi_lag{lag}'] = mi0.get(lag, 0.0)
            features[f'p1_mi_lag{lag}'] = mi1.get(lag, 0.0)
            features[f'mi_lag{lag}_diff'] = mi1.get(lag, 0.0) - mi0.get(lag, 0.0)
        # Permutation entropy
        pe0 = self._permutation_entropy(period_0)
        pe1 = self._permutation_entropy(period_1)
        features['p0_perm_entropy'] = pe0
        features['p1_perm_entropy'] = pe1
        features['perm_entropy_diff'] = pe1 - pe0
        return features

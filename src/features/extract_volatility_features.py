import pandas as pd
import numpy as np
from typing import Union, Dict, Tuple, List
from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class VolatilityFeatureExtractor(BaseFeatureExtractor):
    """
    Extract volatility and higher-order moment features comparing pre/post segments.
    """

    def __init__(self,
                 min_returns_length: int = 3,
                 rolling_window_max: int = 10,
                 cache_name: str = 'volatility_features',
                 force_recompute: bool = False):
        """
        Initialize volatility feature extractor.

        Args:
            min_returns_length: minimum length of returns to compute features
            rolling_window_max: maximum window size for vol-of-vol calculations
            cache_name: cache identifier
            force_recompute: ignore cache if True
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            min_returns_length=min_returns_length,
            rolling_window_max=rolling_window_max
        )
        self.min_returns_length = min_returns_length
        self.rolling_window_max = rolling_window_max

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        names: List[str] = []
        for prefix in ['p0', 'p1']:
            names.extend([
                f'{prefix}_realized_vol',
                f'{prefix}_vol_of_vol',
                f'{prefix}_returns_skew',
                f'{prefix}_returns_kurt',
                f'{prefix}_downside_vol',
                f'{prefix}_upside_vol',
                f'{prefix}_vol_asymmetry'
            ])

        names.extend([
            'realized_vol_diff',
            'vol_of_vol_diff',
            'returns_skew_diff',
            'vol_asymmetry_diff'
        ])
        return names

    def _extract_period_values(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract arrays for period 0 and 1."""
        if isinstance(data, TimeSeriesData):
            period_0 = data.period_0_values.astype(float)
            period_1 = data.period_1_values.astype(float)
        elif isinstance(data, pd.DataFrame):
            if 'value' not in data.columns or 'period' not in data.columns:
                raise ValueError("DataFrame input must have 'value' and 'period' columns.")
            period_0 = data.loc[data['period'] == 0, 'value'].dropna().astype(float).values
            period_1 = data.loc[data['period'] == 1, 'value'].dropna().astype(float).values
        else:
            raise TypeError("Input must be a pandas DataFrame or TimeSeriesData instance.")
        return period_0, period_1

    def _volatility_features_segment(self, segment: np.ndarray) -> Dict[str, float]:
        """Compute volatility features for one segment."""
        features: Dict[str, float] = {}

        # Need returns length >= min_returns_length
        if len(segment) < self.min_returns_length:
            # Fill zeros
            features.update({
                'realized_vol': 0.0,
                'vol_of_vol': 0.0,
                'returns_skew': 0.0,
                'returns_kurt': 0.0,
                'downside_vol': 0.0,
                'upside_vol': 0.0,
                'vol_asymmetry': 0.0
            })
            return features

        # Compute returns
        returns = np.diff(segment)
        if len(returns) < 1:
            return {
                'realized_vol': 0.0,
                'vol_of_vol': 0.0,
                'returns_skew': 0.0,
                'returns_kurt': 0.0,
                'downside_vol': 0.0,
                'upside_vol': 0.0,
                'vol_asymmetry': 0.0
            }

        # Realized volatility: std of returns
        realized_vol = float(np.std(returns))

        # Volatility of volatility: rolling std of returns volatility
        # Determine window: at most rolling_window_max, at most half returns length
        window = min(self.rolling_window_max, len(returns) // 2)
        if window >= 2:
            rolling_vol = pd.Series(returns).rolling(window).std().dropna()
            vol_of_vol = float(np.std(rolling_vol)) if len(rolling_vol) > 1 else 0.0
        else:
            vol_of_vol = 0.0

        # Higher moments: skew, kurtosis
        returns_series = pd.Series(returns)
        returns_skew = float(returns_series.skew()) if len(returns) > 2 else 0.0
        returns_kurt = float(returns_series.kurt()) if len(returns) > 3 else 0.0

        # Downside and upside volatility
        downside = returns[returns < 0]
        upside = returns[returns > 0]
        downside_vol = float(np.std(downside)) if len(downside) > 1 else 0.0
        upside_vol = float(np.std(upside)) if len(upside) > 1 else 0.0
        vol_asymmetry = float(upside_vol - downside_vol)
        features.update({
            'realized_vol': realized_vol,
            'vol_of_vol': vol_of_vol,
            'returns_skew': returns_skew,
            'returns_kurt': returns_kurt,
            'downside_vol': downside_vol,
            'upside_vol': upside_vol,
            'vol_asymmetry': vol_asymmetry
        })
        return features

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute volatility features for a single time series."""
        features: Dict[str, float] = {}
        period_0, period_1 = self._extract_period_values(data)

        # Segment features
        seg0 = self._volatility_features_segment(period_0)
        seg1 = self._volatility_features_segment(period_1)

        # Prefix keys
        for key, val in seg0.items():
            features[f'p0_{key}'] = val
        for key, val in seg1.items():
            features[f'p1_{key}'] = val

        # Differences
        features['realized_vol_diff'] = features.get('p1_realized_vol', 0.0) - features.get('p0_realized_vol', 0.0)
        features['vol_of_vol_diff'] = features.get('p1_vol_of_vol', 0.0) - features.get('p0_vol_of_vol', 0.0)
        features['returns_skew_diff'] = features.get('p1_returns_skew', 0.0) - features.get('p0_returns_skew', 0.0)
        features['vol_asymmetry_diff'] = features.get('p1_vol_asymmetry', 0.0) - features.get('p0_vol_asymmetry', 0.0)

        return features

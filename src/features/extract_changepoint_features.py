import pandas as pd
import numpy as np
from typing import Union, Dict
from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class ChangepointFeatureExtractor(BaseFeatureExtractor):
    """
    Extract change-point detection algorithm features comparing two periods.
    """

    def __init__(self,
                 cache_name: str = 'changepoint_features',
                 force_recompute: bool = False):
        """
        Initialize the feature extractor.

        Args:
            cache_name: Name for the cache file.
            force_recompute: If True, ignore cache and recompute all features.
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute
        )

    def get_feature_names(self) -> list:
        """Get list of all feature names produced by this extractor."""
        names = [
            'pelt_cost_improvement', 'pelt_cost_ratio'
        ]
        for model in ['normal', 'variance']:
            names.append(f'pelt_{model}_improvement')
        return names

    def _extract_period_values(self, data: Union[pd.DataFrame, TimeSeriesData]) -> tuple:
        """Extract full series, period0, period1, and boundary index."""
        if isinstance(data, TimeSeriesData):
            period_0 = data.period_0_values
            period_1 = data.period_1_values
        else:
            if not {'period', 'value'}.issubset(data.columns):
                raise ValueError("DataFrame input must have 'value' and 'period' columns.")
            period_0 = data.loc[data['period'] == 0, 'value'].dropna().values
            period_1 = data.loc[data['period'] == 1, 'value'].dropna().values
        full_series = np.concatenate([period_0, period_1])
        boundary_idx = len(period_0)
        return full_series, period_0, period_1, boundary_idx

    def _cost_function(self, segment: np.ndarray, model: str = 'normal') -> float:
        """Compute cost for a segment under specified model."""
        n = len(segment)
        if n < 2:
            return float('inf')
        if model == 'normal':
            mu = np.mean(segment)
            sigma2 = np.var(segment, ddof=1)
            if sigma2 <= 0:
                return float('inf')
            cost = n * (np.log(2 * np.pi * sigma2) + 1) / 2
        elif model == 'variance':
            var = np.var(segment, ddof=1)
            cost = n * np.log(var + 1e-8)
        else:
            raise ValueError(f"Unsupported cost model: {model}")
        return cost

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute all change-point features for a single time series."""
        features: Dict[str, float] = {}
        full_series, period_0, period_1, _ = self._extract_period_values(data)
        # Base PELT-like improvement for default model
        cost_full = self._cost_function(full_series, model='normal')
        cost_split = self._cost_function(period_0, model='normal') + self._cost_function(period_1, model='normal')
        features['pelt_cost_improvement'] = float(cost_full - cost_split)
        features['pelt_cost_ratio'] = float(cost_split / (cost_full + 1e-8))
        # For each model
        for model in ['normal', 'variance']:
            cost_full_m = self._cost_function(full_series, model=model)
            cost_split_m = self._cost_function(period_0, model=model) + self._cost_function(period_1, model=model)
            features[f'pelt_{model}_improvement'] = float(cost_full_m - cost_split_m)
        return features

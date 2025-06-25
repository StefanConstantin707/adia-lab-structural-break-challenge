import pandas as pd
import numpy as np
from typing import Union, Dict, Tuple
from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class RollingFeatureExtractor(BaseFeatureExtractor):
    """
    Extract features based on rolling window statistics around a known breakpoint.
    """

    def __init__(self,
                 min_window: int = 3,
                 max_window_fraction: float = 0.25,
                 cache_name: str = 'rolling_features',
                 force_recompute: bool = False):
        """
        Initialize rolling feature extractor.

        Args:
            min_window: minimum window length for rolling
            max_window_fraction: max fraction of full series length to use as window
            cache_name: cache identifier
            force_recompute: ignore cache if True
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            min_window=min_window,
            max_window_fraction=max_window_fraction
        )
        self.min_window = min_window
        self.max_window_fraction = max_window_fraction

    def get_feature_names(self) -> list:
        """Return list of feature names."""
        return [
            'rolling_mean_change', 'rolling_std_change',
            'rolling_mean_ratio', 'rolling_std_ratio'
        ]

    def _get_full_series_and_boundary(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Tuple[np.ndarray, int]:
        """Extract full series array and boundary index between period 0 and 1."""
        if isinstance(data, TimeSeriesData):
            series_pre = data.period_0_values
            series_post = data.period_1_values
            full_series = np.concatenate([series_pre, series_post])
            boundary_idx = len(series_pre)
        else:
            # assume DataFrame sorted by time index
            if 'period' not in data.columns or 'value' not in data.columns:
                raise ValueError("DataFrame input must have 'value' and 'period' columns.")

            sorted_data = data.sort_index()
            full_series = sorted_data['value'].dropna().astype(float).values

            # boundary count of period 0
            boundary_idx = int(sorted_data.loc[sorted_data['period'] == 0].shape[0])
        return full_series, boundary_idx

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute rolling features for a single time series."""
        features: Dict[str, float] = {}
        full_series, boundary_idx = self._get_full_series_and_boundary(data)
        n = len(full_series)

        # Determine window size: at most max_window_fraction * n, at least min_window
        window = int(min(max(self.min_window, int(n * self.max_window_fraction)), n // 4))
        if n < 2 * self.min_window or window < self.min_window:
            # Insufficient data for rolling
            return {name: 0.0 for name in self.get_feature_names()}

        # Compute rolling mean and std with center=True
        series_pd = pd.Series(full_series)
        rolling_mean = series_pd.rolling(window, center=True).mean()
        rolling_std = series_pd.rolling(window, center=True).std()

        # Define pre and post index ranges around boundary
        pre_start = max(0, boundary_idx - window)
        pre_end = boundary_idx  # exclusive
        post_start = boundary_idx
        post_end = min(n, boundary_idx + window)

        # Ensure non-empty
        if pre_start < pre_end and post_start < post_end:
            pre_indices = list(range(pre_start, pre_end))
            post_indices = list(range(post_start, post_end))

            pre_mean_avg = rolling_mean.iloc[pre_indices].mean()
            post_mean_avg = rolling_mean.iloc[post_indices].mean()
            pre_std_avg = rolling_std.iloc[pre_indices].mean()
            post_std_avg = rolling_std.iloc[post_indices].mean()

            # Compute features, handle division by zero
            features['rolling_mean_change'] = float(post_mean_avg - pre_mean_avg)
            features['rolling_std_change'] = float(post_std_avg - pre_std_avg)
            features['rolling_mean_ratio'] = float(post_mean_avg / (pre_mean_avg + 1e-8))
            features['rolling_std_ratio'] = float(post_std_avg / (pre_std_avg + 1e-8))
        else:
            # fallback zeros
            features.update({name: 0.0 for name in self.get_feature_names()})

        return features
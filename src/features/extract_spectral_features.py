import numpy as np
from scipy import signal
import pandas as pd
from typing import Union, Dict, Tuple, List
from src.data.dataLoader import TimeSeriesData
from src.features.base_feature_extractor import BaseFeatureExtractor


class SpectralFeatureExtractor(BaseFeatureExtractor):
    """
    Extract frequency domain features comparing pre/post segments: spectral centroid, rolloff,
    bandwidth, dominant frequency, and cross-spectral distances.
    """

    def __init__(self,
                 rolloff_percent: float = 0.85,
                 min_length: int = 4,
                 common_freqs_count: int = None,
                 cache_name: str = 'spectral_features',
                 force_recompute: bool = False):
        """
        Initialize spectral feature extractor.

        Args:
            rolloff_percent: cumulative energy threshold for spectral rolloff (e.g., 0.85)
            min_length: minimum segment length to compute features
            common_freqs_count: number of points for aligned frequency grid; if None, use min length of two PSDs
            cache_name: cache identifier
            force_recompute: ignore cache if True
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            rolloff_percent=rolloff_percent,
            min_length=min_length,
            common_freqs_count=common_freqs_count
        )
        self.rolloff_percent = rolloff_percent
        self.min_length = min_length
        self.common_freqs_count = common_freqs_count

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        names: List[str] = []
        # segment features
        for prefix in ['p0', 'p1']:
            names.extend([
                f'{prefix}_spectral_centroid',
                f'{prefix}_spectral_rolloff',
                f'{prefix}_spectral_bandwidth',
                f'{prefix}_dominant_freq'
            ])
        # differences
        names.extend([
            'spectral_centroid_diff',
            'spectral_bandwidth_diff',
            'dominant_freq_diff'
        ])
        # cross-spectral
        names.extend([
            'spectral_euclidean_distance',
            'spectral_kl_divergence'
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

    def _spectral_features_segment(self, segment: np.ndarray) -> Dict[str, float]:
        """Compute spectral features for one segment."""
        if len(segment) < self.min_length:
            return {
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'spectral_bandwidth': 0.0,
                'dominant_freq': 0.0
            }

        # Compute PSD via periodogram
        freqs, psd = signal.periodogram(segment)
        total_power = np.sum(psd) + 1e-8

        # Spectral centroid
        centroid = float(np.sum(freqs * psd) / total_power)

        # Spectral rolloff: frequency where cumsum >= threshold
        cumsum = np.cumsum(psd)
        threshold = self.rolloff_percent * cumsum[-1]
        idx = np.where(cumsum >= threshold)[0]
        rolloff = float(freqs[idx[0]] if len(idx) > 0 else freqs[-1])

        # Spectral bandwidth
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total_power))

        # Dominant frequency
        dominant_freq = float(freqs[np.argmax(psd)])
        return {
            'spectral_centroid': centroid,
            'spectral_rolloff': rolloff,
            'spectral_bandwidth': bandwidth,
            'dominant_freq': dominant_freq
        }

    def _compute_cross_spectral(self, period_0: np.ndarray, period_1: np.ndarray) -> Dict[str, float]:
        """Compute cross-spectral distance measures if feasible."""
        if len(period_0) < self.min_length or len(period_1) < self.min_length:
            return {
                'spectral_euclidean_distance': 0.0,
                'spectral_kl_divergence': 0.0
            }

        freqs0, psd0 = signal.periodogram(period_0)
        freqs1, psd1 = signal.periodogram(period_1)

        # Determine common frequency grid
        if self.common_freqs_count is not None:
            count = self.common_freqs_count
        else:
            count = min(len(freqs0), len(freqs1))

        common_freqs = np.linspace(0, max(freqs0.max(), freqs1.max()), count)
        psd0_i = np.interp(common_freqs, freqs0, psd0)
        psd1_i = np.interp(common_freqs, freqs1, psd1)

        # Euclidean distance
        euclid = float(np.linalg.norm(psd1_i - psd0_i))

        # KL divergence: sum psd1 * log(psd1/psd0)
        kl = float(np.sum(psd1_i * np.log((psd1_i + 1e-8) / (psd0_i + 1e-8))))

        return {
            'spectral_euclidean_distance': euclid,
            'spectral_kl_divergence': kl
        }

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute spectral features for a single time series."""
        features: Dict[str, float] = {}
        period_0, period_1 = self._extract_period_values(data)

        # Segment features
        seg0 = self._spectral_features_segment(period_0)
        seg1 = self._spectral_features_segment(period_1)

        # Prefix and store
        features['p0_spectral_centroid'] = seg0['spectral_centroid']
        features['p0_spectral_rolloff'] = seg0['spectral_rolloff']
        features['p0_spectral_bandwidth'] = seg0['spectral_bandwidth']
        features['p0_dominant_freq'] = seg0['dominant_freq']
        features['p1_spectral_centroid'] = seg1['spectral_centroid']
        features['p1_spectral_rolloff'] = seg1['spectral_rolloff']
        features['p1_spectral_bandwidth'] = seg1['spectral_bandwidth']
        features['p1_dominant_freq'] = seg1['dominant_freq']

        # Differences
        features['spectral_centroid_diff'] = features['p1_spectral_centroid'] - features['p0_spectral_centroid']
        features['spectral_bandwidth_diff'] = features['p1_spectral_bandwidth'] - features['p0_spectral_bandwidth']
        features['dominant_freq_diff'] = features['p1_dominant_freq'] - features['p0_dominant_freq']

        # Cross-spectral
        cross_feats = self._compute_cross_spectral(period_0, period_1)
        features.update(cross_feats)

        return features

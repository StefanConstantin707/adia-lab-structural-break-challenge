# src/features/base_feature_extractor.py
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Optional, Any
import pandas as pd
import numpy as np
import hashlib
import warnings
from pathlib import Path
import json

from config import CACHE_DIR
from src.data.dataLoader import TimeSeriesData


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors with caching support.

    All feature extractors should inherit from this class and implement
    the abstract methods.
    """

    def __init__(self,
                 cache_name: str = None,
                 force_recompute: bool = False,
                 cache_dir: str = None,
                 **kwargs):
        """
        Initialize the base feature extractor.

        Args:
            cache_name: Name for the cache file. If None, uses class name
            force_recompute: If True, ignore cache and recompute all features
            cache_dir: Custom cache directory. If None, uses CACHE_DIR from config
            **kwargs: Additional parameters stored in self.params
        """
        self.force_recompute = force_recompute
        self.params = kwargs

        # Setup cache
        if cache_name is None:
            cache_name = self.__class__.__name__.lower()

        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / f"{cache_name}.parquet"

        # Cache metadata file to store parameters
        self.cache_meta_file = self.cache_dir / f"{cache_name}_metadata.json"

        self.cache_verified = False
        self.cache_df = None
        self._verification_samples = 2

        # Load cache if exists
        if not force_recompute:
            self._load_cache()

    def _load_cache(self):
        """Load cache and metadata if they exist."""
        if self.cache_file.exists() and self.cache_meta_file.exists():
            try:
                # Load metadata
                with open(self.cache_meta_file, 'r') as f:
                    cache_metadata = json.load(f)

                # Check if parameters match
                if self._check_params_match(cache_metadata.get('params', {})):
                    self.cache_df = pd.read_parquet(self.cache_file)
                    print(f"Loaded cache from {self.cache_file} with {len(self.cache_df)} entries")
                else:
                    warnings.warn("Cache parameters don't match current parameters. Will recompute.")
                    self.cache_df = None
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}. Will recompute.")
                self.cache_df = None

    def _check_params_match(self, cached_params: Dict[str, Any]) -> bool:
        """Check if cached parameters match current parameters."""
        # Get comparable parameters (exclude non-comparable ones)
        current_params = self._get_comparable_params()

        # Compare parameters
        if set(current_params.keys()) != set(cached_params.keys()):
            return False

        for key, value in current_params.items():
            if key not in cached_params:
                return False

            cached_value = cached_params[key]

            # Handle special comparisons
            if isinstance(value, np.ndarray):
                if not np.array_equal(value, np.array(cached_value)):
                    return False
            elif value != cached_value:
                return False

        return True

    def _get_comparable_params(self) -> Dict[str, Any]:
        """Get parameters that should be compared for cache validity."""
        # Default implementation - override in subclasses if needed
        return {k: v for k, v in self.params.items()
                if not callable(v) and k not in ['force_recompute']}

    def _save_cache(self):
        """Save the cache dataframe and metadata to disk."""
        if self.cache_df is not None and len(self.cache_df) > 0:
            # Save data
            self.cache_df.to_parquet(self.cache_file)

            # Save metadata
            metadata = {
                'params': self._get_comparable_params(),
                'feature_names': list(self.cache_df.columns),
                'n_entries': len(self.cache_df),
                'class_name': self.__class__.__name__
            }

            with open(self.cache_meta_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            print(f"Saved cache to {self.cache_file} with {len(self.cache_df)} entries")

    def _get_cache_key(self, data: Union[pd.DataFrame, TimeSeriesData]) -> str:
        """
        Generate unique cache key for the time series.

        Override this method if you want custom cache key generation.
        """
        if isinstance(data, TimeSeriesData):
            # Use the ts_id if available
            if hasattr(data, 'series_id') and data.series_id is not None:
                return str(data.series_id)
            # Otherwise create hash from values
            vals = np.concatenate([data.period_0_values, data.period_1_values])
        else:
            vals = data['value'].values

        # Create hash from values and parameters
        hash_obj = hashlib.md5()
        hash_obj.update(vals.tobytes())

        # Include relevant parameters in hash
        param_str = json.dumps(self._get_comparable_params(), sort_keys=True, default=str)
        hash_obj.update(param_str.encode())

        return hash_obj.hexdigest()

    def _extract_period_values(self, data: Union[pd.DataFrame, TimeSeriesData]) -> tuple:
        """Extract period values from input data."""
        if isinstance(data, TimeSeriesData):
            return data.period_0_values, data.period_1_values
        elif isinstance(data, pd.DataFrame):
            if 'value' not in data.columns or 'period' not in data.columns:
                raise ValueError("DataFrame must have 'value' and 'period' columns.")
            vals0 = data.loc[data['period'] == 0, 'value'].dropna().values
            vals1 = data.loc[data['period'] == 1, 'value'].dropna().values
            return vals0, vals1
        else:
            raise TypeError("Input must be DataFrame or TimeSeriesData instance.")

    def _verify_cache_entry(self, cache_key: str, computed_features: Dict[str, float]) -> bool:
        """Verify that cached features match computed features."""
        if self.cache_df is None or cache_key not in self.cache_df.index:
            return False

        cached_row = self.cache_df.loc[cache_key]

        # Check that all computed features exist in cache
        for key in computed_features.keys():
            if key not in cached_row:
                print(f"Feature {key} missing from cache")
                return False

        # Compare each feature value
        for key, computed_val in computed_features.items():
            cached_val = cached_row[key]

            # Handle NaN comparison
            if pd.isna(computed_val) and pd.isna(cached_val):
                continue
            elif pd.isna(computed_val) or pd.isna(cached_val):
                print(f"NaN mismatch for {key}: cached={cached_val}, computed={computed_val}")
                return False

            # Check if values are close enough
            if not np.isclose(computed_val, cached_val, rtol=1e-9, atol=1e-12):
                print(f"Value mismatch for {key}: cached={cached_val}, computed={computed_val}")
                return False

        return True

    def _verify_cache_integrity(self, sample_data: List[Union[pd.DataFrame, TimeSeriesData]]) -> bool:
        """Verify cache integrity using a sample of data."""
        if not sample_data:
            return True

        print(f"Verifying cache integrity using {len(sample_data)} samples...")

        for i, data in enumerate(sample_data):
            cache_key = self._get_cache_key(data)

            if cache_key in self.cache_df.index:
                computed_features = self._compute_features(data)

                if not self._verify_cache_entry(cache_key, computed_features):
                    print(f"Cache verification failed at sample {i + 1}")
                    return False

        print("Cache verification successful!")
        return True

    @abstractmethod
    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """
        Compute all features for a single time series.

        This method must be implemented by all subclasses.

        Args:
            data: Time series data

        Returns:
            Dictionary of computed features
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that this extractor produces.

        This method must be implemented by all subclasses.

        Returns:
            List of feature names
        """
        pass

    def extract_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """
        Extract features with caching support.

        Args:
            data: Time series data

        Returns:
            Dictionary of features
        """
        # Get cache key
        cache_key = self._get_cache_key(data)

        # Check cache first
        if not self.force_recompute and self.cache_df is not None and cache_key in self.cache_df.index:
            # On first access, verify cache with some samples
            if not self.cache_verified:
                # Get random sample of cached entries for verification
                sample_indices = np.random.choice(
                    len(self.cache_df),
                    size=min(self._verification_samples, len(self.cache_df)),
                    replace=False
                )

                # For now, just verify this entry
                computed_features = self._compute_features(data)

                if self._verify_cache_entry(cache_key, computed_features):
                    self.cache_verified = True
                    return self.cache_df.loc[cache_key].to_dict()
                else:
                    warnings.warn("Cache verification failed! Deleting cache and recomputing...")
                    self._clear_cache()
                    return self._compute_and_cache_features(data, cache_key)

            return self.cache_df.loc[cache_key].to_dict()

        # Compute features
        return self._compute_and_cache_features(data, cache_key)

    def _compute_and_cache_features(self, data: Union[pd.DataFrame, TimeSeriesData],
                                    cache_key: str) -> Dict[str, float]:
        """Compute features and update cache."""
        features = self._compute_features(data)

        # Update cache
        if self.cache_df is None:
            self.cache_df = pd.DataFrame([features], index=[cache_key])
        else:
            # Ensure all columns exist
            for col in features.keys():
                if col not in self.cache_df.columns:
                    self.cache_df[col] = np.nan

            self.cache_df.loc[cache_key] = features

        # Save cache periodically (every 100 entries)
        if len(self.cache_df) % 100 == 0:
            self._save_cache()

        return features

    def _clear_cache(self):
        """Clear cache files and memory."""
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.cache_meta_file.exists():
            self.cache_meta_file.unlink()
        self.cache_df = None
        self.cache_verified = False

    def batch_extract_features(self, data_list: List[Union[pd.DataFrame, TimeSeriesData]],
                               show_progress: bool = True) -> pd.DataFrame:
        """
        Extract features for a batch of time series.

        Args:
            data_list: List of time series data
            show_progress: Whether to show progress

        Returns:
            DataFrame with features for all time series
        """
        all_features = []
        all_keys = []

        for i, data in enumerate(data_list):
            if show_progress and i % 100 == 0:
                print(f"Processing {i}/{len(data_list)}...")

            cache_key = self._get_cache_key(data)
            features = self.extract_features(data)

            all_features.append(features)
            all_keys.append(cache_key)

        # Save final cache
        self._save_cache()

        return pd.DataFrame(all_features, index=all_keys)

    def get_cached_features(self) -> Optional[pd.DataFrame]:
        """Get all cached features if available."""
        return self.cache_df.copy() if self.cache_df is not None else None

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        info = {
            'cache_file': str(self.cache_file),
            'cache_exists': self.cache_file.exists(),
            'cache_loaded': self.cache_df is not None,
            'cache_verified': self.cache_verified,
            'force_recompute': self.force_recompute
        }

        if self.cache_df is not None:
            info.update({
                'n_entries': len(self.cache_df),
                'n_features': len(self.cache_df.columns),
                'feature_names': list(self.cache_df.columns)
            })

        return info

    def clear_cache(self):
        """Public method to clear cache."""
        self._clear_cache()
        print(f"Cache cleared for {self.__class__.__name__}")
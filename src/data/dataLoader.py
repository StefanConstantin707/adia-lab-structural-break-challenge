# src/data/dataloader.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import logging
from dataclasses import dataclass
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesData:
    """Container for a single time series with metadata"""
    series_id: int
    values: np.ndarray
    time_points: np.ndarray
    boundary_point: int
    period_0_values: np.ndarray
    period_1_values: np.ndarray
    has_break: Optional[bool] = None

    @property
    def length(self) -> int:
        return len(self.values)

    @property
    def period_0_length(self) -> int:
        return len(self.period_0_values)

    @property
    def period_1_length(self) -> int:
        return len(self.period_1_values)


class StructuralBreakDataLoader:
    """
    Data loader for the ADIA Lab Structural Break Challenge.

    Handles loading, preprocessing, and accessing time series data
    with structural break labels.
    """

    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self._train_data_dict = {}
        self._test_data_dict = {}

    def load_data(self, use_crunch: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Load data either using crunch module or from local files.

        Args:
            use_crunch: If True, use crunch.load_data(), else load from local files

        Returns:
            Tuple of (X_train, y_train, X_test)
        """
        if use_crunch:
            try:
                import crunch
                crunch_obj = crunch.load_notebook()
                self.X_train, self.y_train, self.X_test = crunch_obj.load_data()
                logger.info("Data loaded successfully using crunch")
            except ImportError:
                logger.warning("Crunch module not available, loading from local files")
                self._load_from_files()
        else:
            self._load_from_files()

        # Process data into dictionary format for easier access
        self._process_data()

        return self.X_train, self.y_train, self.X_test

    def _load_from_files(self):
        """Load data from local parquet files"""
        try:
            self.X_train = pd.read_parquet(self.data_dir / "X_train.parquet")
            self.y_train = pd.read_parquet(self.data_dir / "y_train.parquet")
            if isinstance(self.y_train, pd.DataFrame):
                self.y_train = self.y_train.squeeze()
            self.X_test = pd.read_parquet(self.data_dir / "X_test.reduced.parquet")
            logger.info("Data loaded successfully from local files")
        except FileNotFoundError as e:
            logger.error(f"Data files not found: {e}")
            raise

    def _process_data(self):
        """Process raw data into TimeSeriesData objects"""
        # Process training data
        for series_id in self.X_train.index.get_level_values('id').unique():
            series_data = self.X_train.loc[series_id]

            # Find boundary point (where period changes from 0 to 1)
            periods = series_data['period'].values
            boundary_idx = np.where(periods == 1)[0][0] if 1 in periods else len(periods)

            # Split into periods
            period_0_mask = periods == 0
            period_1_mask = periods == 1

            ts_data = TimeSeriesData(
                series_id=series_id,
                values=series_data['value'].values,
                time_points=series_data.index.values,
                boundary_point=boundary_idx,
                period_0_values=series_data.loc[period_0_mask, 'value'].values,
                period_1_values=series_data.loc[period_1_mask, 'value'].values,
                has_break=self.y_train.loc[series_id] if series_id in self.y_train.index else None
            )

            self._train_data_dict[series_id] = ts_data

        # Process test data
        for series_id in self.X_test.index.get_level_values('id').unique():
            series_data = self.X_test.loc[series_id]

            # Find boundary point
            periods = series_data['period'].values
            boundary_idx = np.where(periods == 1)[0][0] if 1 in periods else len(periods)

            # Split into periods
            period_0_mask = periods == 0
            period_1_mask = periods == 1

            ts_data = TimeSeriesData(
                series_id=series_id,
                values=series_data['value'].values,
                time_points=series_data.index.values,
                boundary_point=boundary_idx,
                period_0_values=series_data.loc[period_0_mask, 'value'].values,
                period_1_values=series_data.loc[period_1_mask, 'value'].values
            )

            self._test_data_dict[series_id] = ts_data

    def get_train_series(self, series_id: int) -> TimeSeriesData:
        """Get a specific training time series"""
        return self._train_data_dict.get(series_id)

    def get_test_series(self, series_id: int) -> TimeSeriesData:
        """Get a specific test time series"""
        return self._test_data_dict.get(series_id)

    def get_all_train_series(self) -> Dict[int, TimeSeriesData]:
        """Get all training time series"""
        return self._train_data_dict

    def get_all_test_series(self) -> Dict[int, TimeSeriesData]:
        """Get all test time series"""
        return self._test_data_dict

    def get_train_ids(self) -> List[int]:
        """Get all training series IDs"""
        return list(self._train_data_dict.keys())

    def get_test_ids(self) -> List[int]:
        """Get all test series IDs"""
        return list(self._test_data_dict.keys())

    def get_positive_examples(self) -> Dict[int, TimeSeriesData]:
        """Get all training examples with structural breaks"""
        return {
            sid: ts for sid, ts in self._train_data_dict.items()
            if ts.has_break
        }

    def get_negative_examples(self) -> Dict[int, TimeSeriesData]:
        """Get all training examples without structural breaks"""
        return {
            sid: ts for sid, ts in self._train_data_dict.items()
            if not ts.has_break
        }

    def get_statistics(self) -> Dict[str, any]:
        """Get basic statistics about the dataset"""
        positive_examples = self.get_positive_examples()
        negative_examples = self.get_negative_examples()

        stats = {
            'n_train_series': len(self._train_data_dict),
            'n_test_series': len(self._test_data_dict),
            'n_positive': len(positive_examples),
            'n_negative': len(negative_examples),
            'positive_ratio': len(positive_examples) / len(self._train_data_dict),
            'series_lengths': [ts.length for ts in self._train_data_dict.values()],
            'boundary_positions': [ts.boundary_point for ts in self._train_data_dict.values()]
        }

        # Add length statistics
        lengths = stats['series_lengths']
        stats['min_length'] = min(lengths)
        stats['max_length'] = max(lengths)
        stats['mean_length'] = np.mean(lengths)
        stats['std_length'] = np.std(lengths)

        return stats

    def create_train_val_split(self, val_ratio: float = 0.2, random_state: int = 42) -> Tuple[Dict[int, TimeSeriesData], Dict[int, TimeSeriesData]]:
        """
        Create train/validation split preserving class balance.

        Args:
            val_ratio: Fraction of data to use for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dict, val_dict)
        """
        np.random.seed(random_state)

        positive_ids = [sid for sid, ts in self._train_data_dict.items() if ts.has_break]
        negative_ids = [sid for sid, ts in self._train_data_dict.items() if not ts.has_break]

        # Shuffle
        np.random.shuffle(positive_ids)
        np.random.shuffle(negative_ids)

        # Split
        n_pos_val = int(len(positive_ids) * val_ratio)
        n_neg_val = int(len(negative_ids) * val_ratio)

        val_ids = positive_ids[:n_pos_val] + negative_ids[:n_neg_val]
        train_ids = positive_ids[n_pos_val:] + negative_ids[n_neg_val:]

        train_dict = {sid: self._train_data_dict[sid] for sid in train_ids}
        val_dict = {sid: self._train_data_dict[sid] for sid in val_ids}

        return train_dict, val_dict

    def save_processed_data(self, filepath: Union[str, Path]):
        """Save processed data to disk for faster loading"""
        data = {
            'train_data_dict': self._train_data_dict,
            'test_data_dict': self._test_data_dict,
            'y_train': self.y_train
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Processed data saved to {filepath}")

    def load_processed_data(self, filepath: Union[str, Path]):
        """Load previously processed data"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self._train_data_dict = data['train_data_dict']
        self._test_data_dict = data['test_data_dict']
        self.y_train = data['y_train']
        logger.info(f"Processed data loaded from {filepath}")

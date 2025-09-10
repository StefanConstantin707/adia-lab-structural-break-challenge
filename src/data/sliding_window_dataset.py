# src/data/sliding_window_dataset.py
"""
Create sliding window dataset for neural network training on structural breaks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import torch
from statsmodels.sandbox.distributions.genpareto import shape
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging

from src.data.dataLoader import TimeSeriesData, StructuralBreakDataLoader

logger = logging.getLogger(__name__)


@dataclass
class WindowSample:
    """Single training sample: a window of time series data"""
    window_data: np.ndarray  # Shape: (window_size,)
    has_break: bool  # Whether this window contains a break
    break_position: Optional[int]  # Position of break within window (None if no break)
    series_id: str  # Original series ID
    window_start: int  # Start position in original series


class SlidingWindowDataset(Dataset):
    """PyTorch Dataset for sliding window structural break detection"""

    def __init__(self, samples: List[WindowSample], normalize: bool = True):
        self.samples = samples
        self.normalize = normalize

        if normalize:
            # Fit scaler on all windows
            all_windows = np.vstack([s.window_data.reshape(1, -1) for s in samples])
            self.scaler = StandardScaler()
            self.scaler.fit(all_windows)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Normalize window if requested
        window = sample.window_data.copy()
        if self.normalize:
            window = self.scaler.transform(window.reshape(1, -1)).flatten()

        # Create position encoding (optional, can be used as additional channel)
        positions = np.arange(len(window)) / len(window)  # Normalized positions 0-1

        # Break position encoding (-1 if no break, otherwise normalized position)
        break_pos_encoded = -1.0 if sample.break_position is None else sample.break_position / len(window)

        return {
            'window': torch.FloatTensor(window),
            'positions': torch.FloatTensor(positions),
            'has_break': torch.FloatTensor([1.0 if sample.has_break else 0.0]),
            'break_position': torch.FloatTensor([break_pos_encoded]),
            'series_id': sample.series_id,
            'window_start': sample.window_start
        }


def create_sliding_window_dataset(
        train_data_dict: Dict,
        window_size: int = 100,
        stride: int = 20,
        min_break_distance: int = 10,
        balance_classes: bool = True,
        max_windows_per_series: int = 50
) -> Tuple[List[WindowSample], Dict]:
    """
    Create sliding window dataset from time series data

    Args:
        train_data_dict: Dictionary of {series_id: TimeSeriesData}
        window_size: Size of each sliding window
        stride: Step size for sliding window
        min_break_distance: Minimum distance from window edge to consider break "in window"
        balance_classes: Whether to balance break vs no-break windows
        max_windows_per_series: Maximum windows to extract per series (for efficiency)

    Returns:
        List of WindowSample objects and statistics dict
    """

    all_samples = []
    stats = {
        'total_windows': 0,
        'break_windows': 0,
        'no_break_windows': 0,
        'series_processed': 0,
        'series_skipped': 0
    }

    for series_id, ts_obj in train_data_dict.items():
        try:
            # Get the full time series
            full_series = np.concatenate([ts_obj.period_0_values, ts_obj.period_1_values])

            if len(full_series) < window_size:
                stats['series_skipped'] += 1
                continue

            # Determine break position in full series
            break_pos_in_full = len(ts_obj.period_0_values)  # Break is between periods

            # Generate sliding windows
            series_samples = []

            for start_pos in range(0, len(full_series) - window_size + 1, stride):

                end_pos = start_pos + window_size


                if (start_pos + min_break_distance <= break_pos_in_full <=
                        end_pos - min_break_distance):
                    break_position = break_pos_in_full - start_pos
                else:
                    continue

                window_data = np.zeros((window_size * 2,))
                window_data[:window_size] = full_series[start_pos:end_pos]
                one = np.ones((break_pos_in_full - start_pos,))
                window_data[window_size: (window_size + break_pos_in_full - start_pos)] = one

                sample = WindowSample(
                    window_data=window_data,
                    has_break=ts_obj.has_break,
                    break_position=break_position,
                    series_id=series_id,
                    window_start=start_pos
                )

                series_samples.append(sample)

                # Limit windows per series for efficiency
                if len(series_samples) >= max_windows_per_series:
                    break

            all_samples.extend(series_samples)
            stats['series_processed'] += 1

            if stats['series_processed'] % 100 == 0:
                print(f"Processed {stats['series_processed']} series, "
                      f"generated {len(all_samples)} windows so far...")

        except Exception as e:
            logger.warning(f"Failed to process series {series_id}: {e}")
            stats['series_skipped'] += 1
            continue

    # Count break vs no-break windows
    break_samples = [s for s in all_samples if s.has_break]
    no_break_samples = [s for s in all_samples if not s.has_break]

    stats['break_windows'] = len(break_samples)
    stats['no_break_windows'] = len(no_break_samples)
    stats['total_windows'] = len(all_samples)

    print(f"\nDataset Statistics:")
    print(f"Total windows: {stats['total_windows']}")
    print(f"Break windows: {stats['break_windows']}")
    print(f"No-break windows: {stats['no_break_windows']}")
    print(f"Break ratio: {stats['break_windows'] / stats['total_windows']:.3f}")

    # Balance classes if requested
    if balance_classes:
        break_samples_nu = len(break_samples)

        req_no_break = int(break_samples_nu / 3 * 7)

        # Randomly sample to balance
        np.random.seed(42)

        no_break_samples = np.random.choice(no_break_samples, size=req_no_break, replace=False).tolist()

        balanced_samples = break_samples + no_break_samples
        np.random.shuffle(balanced_samples)

        print(f"\nAfter balancing:")
        print(f"Total windows: {len(balanced_samples)}")
        print(f"Break windows: {len(break_samples)}")
        print(f"No-break windows: {len(no_break_samples)}")

        return balanced_samples, stats

    return all_samples, stats

def create_simple_testing_dataset(
        test_data_dict: Dict,
        window_size: int = 100,
) -> Tuple[List[WindowSample], Dict]:
    """
    Create sliding window dataset from time series data

    Args:
        train_data_dict: Dictionary of {series_id: TimeSeriesData}
        window_size: Size of each sliding window
        stride: Step size for sliding window
        min_break_distance: Minimum distance from window edge to consider break "in window"
        balance_classes: Whether to balance break vs no-break windows
        max_windows_per_series: Maximum windows to extract per series (for efficiency)

    Returns:
        List of WindowSample objects and statistics dict
    """

    all_samples = []
    stats = {
        'total_windows': 0,
        'break_windows': 0,
        'no_break_windows': 0,
        'series_processed': 0,
        'series_skipped': 0
    }

    for series_id, ts_obj in test_data_dict.items():
        try:
            # Determine break position in full series
            break_pos_in_full = len(ts_obj.period_0_values)  # Break is between periods

            # Generate sliding windows
            series_samples = []

            start_pos = ts_obj.period_0_values.shape[0] - window_size // 2

            data = np.zeros((window_size * 2,))
            data[:window_size] = np.concatenate((ts_obj.period_0_values[-window_size//2:], ts_obj.period_1_values[:window_size//2]), axis=0)
            one = np.ones((break_pos_in_full - start_pos,))
            data[window_size: (window_size + break_pos_in_full - start_pos)] = one



            sample = WindowSample(
                window_data=data,
                has_break=ts_obj.has_break,
                break_position=break_pos_in_full,
                series_id=series_id,
                window_start=ts_obj.period_0_values.shape[0] - window_size//2,
            )

            series_samples.append(sample)

            all_samples.extend(series_samples)
            stats['series_processed'] += 1

            if stats['series_processed'] % 100 == 0:
                print(f"Processed {stats['series_processed']} series, "
                      f"generated {len(all_samples)} windows so far...")

        except Exception as e:
            logger.warning(f"Failed to process series {series_id}: {e}")
            stats['series_skipped'] += 1
            continue

    # Count break vs no-break windows
    break_samples = [s for s in all_samples if s.has_break]
    no_break_samples = [s for s in all_samples if not s.has_break]

    stats['break_windows'] = len(break_samples)
    stats['no_break_windows'] = len(no_break_samples)
    stats['total_windows'] = len(all_samples)

    print(f"\nDataset Statistics:")
    print(f"Total windows: {stats['total_windows']}")
    print(f"Break windows: {stats['break_windows']}")
    print(f"No-break windows: {stats['no_break_windows']}")
    print(f"Break ratio: {stats['break_windows'] / stats['total_windows']:.3f}")

    return all_samples, stats

def create_simple_testing_dataset2(
        test_data_dict: Dict,
        window_size: int = 100,
) -> torch.Tensor:
    """
    Create sliding window dataset from time series data

    Args:
        train_data_dict: Dictionary of {series_id: TimeSeriesData}
        window_size: Size of each sliding window
        stride: Step size for sliding window
        min_break_distance: Minimum distance from window edge to consider break "in window"
        balance_classes: Whether to balance break vs no-break windows
        max_windows_per_series: Maximum windows to extract per series (for efficiency)

    Returns:
        List of WindowSample objects and statistics dict
    """

    all_samples = []
    stats = {
        'total_windows': 0,
        'break_windows': 0,
        'no_break_windows': 0,
        'series_processed': 0,
        'series_skipped': 0
    }

    len_data = len(test_data_dict)
    data = torch.empty(size=(len_data, window_size))

    i = 0
    for series_id, ts_obj in test_data_dict.items():

        len_pre_post = window_size // 2

        data[i, :len_pre_post] = ts_obj.period_0_values[-len_pre_post:]
        data[i, len_pre_post:] = ts_obj.period_1_values[:len_pre_post]

        i += 1

    return data


def create_test_windows(
        ts_obj: TimeSeriesData,
        window_size: int = 100,
        stride: int = 10,
        scaler: Optional[StandardScaler] = None
) -> List[Dict]:
    """
    Create all possible test windows for a single time series at inference time

    Args:
        ts_obj: TimeSeriesData object
        window_size: Size of sliding windows
        stride: Step size for sliding windows
        scaler: Fitted scaler from training (optional)

    Returns:
        List of window dictionaries for prediction
    """
    full_series = np.concatenate([ts_obj.period_0_values, ts_obj.period_1_values])

    if len(full_series) < window_size:
        # Handle short series - pad or return single window
        if len(full_series) == 0:
            return []

        # Pad with edge values
        padded_series = np.pad(full_series,
                               (0, window_size - len(full_series)),
                               mode='edge')
        full_series = padded_series

    test_windows = []

    for start_pos in range(0, len(full_series) - window_size + 1, stride):
        window_data = full_series[start_pos:start_pos + window_size]

        # Normalize if scaler provided
        if scaler is not None:
            window_data = scaler.transform(window_data.reshape(1, -1)).flatten()

        # Create position encoding
        positions = np.arange(len(window_data)) / len(window_data)

        test_windows.append({
            'window': torch.FloatTensor(window_data),
            'positions': torch.FloatTensor(positions),
            'window_start': start_pos,
            'series_id': ts_obj.series_id
        })

    return test_windows


def analyze_dataset_distribution(samples: List[WindowSample]) -> Dict:
    """Analyze the distribution of the created dataset"""

    analysis = {
        'series_distribution': {},
        'break_position_distribution': [],
        'window_length_stats': {},
        'class_balance': {}
    }

    # Series distribution
    series_counts = {}
    for sample in samples:
        series_counts[sample.series_id] = series_counts.get(sample.series_id, 0) + 1

    analysis['series_distribution'] = {
        'unique_series': len(series_counts),
        'avg_windows_per_series': np.mean(list(series_counts.values())),
        'std_windows_per_series': np.std(list(series_counts.values())),
        'min_windows_per_series': min(series_counts.values()),
        'max_windows_per_series': max(series_counts.values())
    }

    # Break position distribution
    break_positions = [s.break_position for s in samples if s.break_position is not None]
    if break_positions:
        analysis['break_position_distribution'] = {
            'mean_position': np.mean(break_positions),
            'std_position': np.std(break_positions),
            'min_position': min(break_positions),
            'max_position': max(break_positions),
            'positions': break_positions
        }

    # Window length stats
    window_lengths = [len(s.window_data) for s in samples]
    analysis['window_length_stats'] = {
        'mean_length': np.mean(window_lengths),
        'std_length': np.std(window_lengths),
        'min_length': min(window_lengths),
        'max_length': max(window_lengths)
    }

    # Class balance
    break_count = sum(1 for s in samples if s.has_break)
    total_count = len(samples)
    analysis['class_balance'] = {
        'break_count': break_count,
        'no_break_count': total_count - break_count,
        'break_ratio': break_count / total_count,
        'total_samples': total_count
    }

    return analysis


def create_dataset_for_training(data_dict: Dict = None, data_handler: StructuralBreakDataLoader = None, window_size=100, stride=20, normalize=False):
    """
    Complete pipeline to create sliding window dataset
    """
    print("Creating sliding window dataset...")

    if data_dict is None:
        # Get training data
        data_dict = data_handler.get_all_train_series()
    elif data_handler is None:
        data_dict = data_dict
    else:
        raise ValueError("data_handler or data_dict must be None")

    # Create sliding window samples
    samples, stats = create_sliding_window_dataset(
        data_dict,
        window_size=window_size,
        stride=stride,
        balance_classes=True,
        max_windows_per_series=10000
    )

    # Analyze dataset
    analysis = analyze_dataset_distribution(samples)
    print(f"\nDataset Analysis:")
    print(f"Unique series: {analysis['series_distribution']['unique_series']}")
    print(f"Avg windows per series: {analysis['series_distribution']['avg_windows_per_series']:.1f}")
    print(f"Class balance: {analysis['class_balance']['break_ratio']:.3f}")

    # Create PyTorch dataset
    dataset = SlidingWindowDataset(samples, normalize=normalize)

    print(f"Created dataset with {len(dataset)} samples")

    return dataset, analysis


def create_dataset_for_testing(data_dict: Dict = None, data_handler: StructuralBreakDataLoader = None, window_size=100, normalize=False):
    """
    Complete pipeline to create sliding window dataset
    """
    print("Creating sliding window dataset...")

    if data_dict is None:
        # Get training data
        data_dict = data_handler.get_all_train_series()
    elif data_handler is None:
        data_dict = data_dict
    else:
        raise ValueError("data_handler or data_dict must be None")

    # Create sliding window samples
    samples, stats = create_simple_testing_dataset(
        data_dict,
        window_size=window_size,
    )

    # Analyze dataset
    analysis = analyze_dataset_distribution(samples)
    print(f"\nDataset Analysis:")
    print(f"Unique series: {analysis['series_distribution']['unique_series']}")
    print(f"Avg windows per series: {analysis['series_distribution']['avg_windows_per_series']:.1f}")
    print(f"Class balance: {analysis['class_balance']['break_ratio']:.3f}")

    # Create PyTorch dataset
    dataset = SlidingWindowDataset(samples, normalize=normalize)

    print(f"Created dataset with {len(dataset)} samples")

    return dataset, analysis


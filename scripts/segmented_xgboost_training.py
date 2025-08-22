# scripts/segmented_xgboost_training_proper.py
"""
XGBoost training with proper augmentation - only augment training folds, never test fold
"""
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from src.data.dataLoader import StructuralBreakDataLoader, TimeSeriesData
from src.features.extract_distribution_features import DistributionCombinedFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_SAMPLES_TO_USE = 1000  # Use only 1000 examples


@dataclass
class SegmentedTimeSeriesData:
    """Container for a segmented time series with metadata"""
    original_series_id: int
    segment_id: int
    values: np.ndarray
    time_points: np.ndarray
    boundary_point: int
    has_break: bool
    segment_start: int
    segment_end: int


class TimeSeriesAugmenter:
    """
    Augments time series by creating multiple views/segments
    """

    def __init__(self, augmentation_factor: int = 3, strategy: str = 'window_shift'):
        """
        Args:
            augmentation_factor: How many augmented versions to create per series
            strategy: 'window_shift', 'subsample', or 'noise'
        """
        self.augmentation_factor = augmentation_factor
        self.strategy = strategy

    def augment_time_series(self, ts_data: TimeSeriesData) -> List[TimeSeriesData]:
        """
        Create augmented versions of a single time series
        All augmented versions keep the SAME label as the original
        """
        if self.strategy == 'window_shift':
            return self._augment_window_shift(ts_data)
        elif self.strategy == 'subsample':
            return self._augment_subsample(ts_data)
        elif self.strategy == 'noise':
            return self._augment_noise(ts_data)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _augment_window_shift(self, ts_data: TimeSeriesData) -> List[TimeSeriesData]:
        """Create different windows around the boundary point"""
        augmented = []

        # Always include the original
        augmented.append(ts_data)

        # Create shifted windows
        series_length = len(ts_data.values)
        boundary = ts_data.boundary_point

        for i in range(1, self.augmentation_factor):
            # Calculate window size (80-100% of original)
            window_size = int(series_length * (0.8 + 0.2 * np.random.random()))

            # Random offset for the window
            max_offset = max(0, series_length - window_size)
            if max_offset > 0:
                offset = np.random.randint(0, max_offset)
            else:
                offset = 0

            # Ensure boundary stays in the window
            if offset > boundary:
                offset = max(0, boundary - window_size // 2)
            if offset + window_size <= boundary:
                offset = min(boundary - window_size // 2, series_length - window_size)

            # Extract window
            start = max(0, offset)
            end = min(series_length, start + window_size)

            window_values = ts_data.values[start:end]
            new_boundary = boundary - start

            # Ensure boundary is valid
            if new_boundary < 0 or new_boundary >= len(window_values):
                # Skip this augmentation
                continue

            augmented_ts = TimeSeriesData(
                series_id=f"{ts_data.series_id}_aug{i}",
                values=window_values,
                time_points=np.arange(len(window_values)),
                boundary_point=new_boundary,
                period_0_values=window_values[:new_boundary],
                period_1_values=window_values[new_boundary:],
                has_break=ts_data.has_break  # KEEP ORIGINAL LABEL
            )
            augmented.append(augmented_ts)

        return augmented

    def _augment_subsample(self, ts_data: TimeSeriesData) -> List[TimeSeriesData]:
        """Subsample the time series at different rates"""
        augmented = []
        augmented.append(ts_data)  # Original

        for i in range(1, self.augmentation_factor):
            # Subsample rate (keep 70-95% of points)
            keep_rate = 0.7 + 0.25 * np.random.random()
            n_keep = int(len(ts_data.values) * keep_rate)

            # Random indices to keep (but always keep boundary region)
            boundary = ts_data.boundary_point
            boundary_region = list(range(max(0, boundary - 10), min(len(ts_data.values), boundary + 10)))

            other_indices = [j for j in range(len(ts_data.values)) if j not in boundary_region]
            n_other = n_keep - len(boundary_region)

            if n_other > 0:
                selected_other = np.random.choice(other_indices, size=min(n_other, len(other_indices)), replace=False)
                keep_indices = sorted(boundary_region + list(selected_other))
            else:
                keep_indices = boundary_region

            # Subsample
            subsampled_values = ts_data.values[keep_indices]

            # Find new boundary position
            new_boundary = np.searchsorted(keep_indices, boundary)

            augmented_ts = TimeSeriesData(
                series_id=f"{ts_data.series_id}_aug{i}",
                values=subsampled_values,
                time_points=np.arange(len(subsampled_values)),
                boundary_point=new_boundary,
                period_0_values=subsampled_values[:new_boundary],
                period_1_values=subsampled_values[new_boundary:],
                has_break=ts_data.has_break  # KEEP ORIGINAL LABEL
            )
            augmented.append(augmented_ts)

        return augmented

    def _augment_noise(self, ts_data: TimeSeriesData) -> List[TimeSeriesData]:
        """Add small noise to the time series"""
        augmented = []
        augmented.append(ts_data)  # Original

        for i in range(1, self.augmentation_factor):
            # Add small Gaussian noise
            noise_level = 0.01 * np.std(ts_data.values)
            noise = np.random.normal(0, noise_level, size=len(ts_data.values))
            noisy_values = ts_data.values + noise

            augmented_ts = TimeSeriesData(
                series_id=f"{ts_data.series_id}_aug{i}",
                values=noisy_values,
                time_points=ts_data.time_points,
                boundary_point=ts_data.boundary_point,
                period_0_values=noisy_values[:ts_data.boundary_point],
                period_1_values=noisy_values[ts_data.boundary_point:],
                has_break=ts_data.has_break  # KEEP ORIGINAL LABEL
            )
            augmented.append(augmented_ts)

        return augmented


def evaluate_with_proper_cv(augmentation_factor: int = 3,
                            augmentation_strategy: str = 'window_shift',
                            n_folds: int = 5):
    """
    Proper CV: Split data into folds FIRST, then augment only training folds
    """
    # Load data
    data_handler = StructuralBreakDataLoader()
    data_handler.load_data(use_crunch=False)
    all_train_data = data_handler.get_all_train_series()

    # Sample subset
    np.random.seed(42)
    all_ids = list(all_train_data.keys())
    np.random.shuffle(all_ids)
    sampled_ids = all_ids[:N_SAMPLES_TO_USE]

    print(f"Using {len(sampled_ids)} time series")

    # Extract features for ALL data (before augmentation)
    print("Extracting features from original data...")
    feature_extractor = DistributionCombinedFeatureExtractor(lags=5)

    features_dict = {}
    labels_dict = {}

    for series_id in tqdm(sampled_ids):
        ts_data = all_train_data[series_id]
        try:
            features = feature_extractor.extract_features(ts_data)
            features_dict[series_id] = features
            labels_dict[series_id] = ts_data.has_break
        except Exception as e:
            logger.warning(f"Failed to extract features for {series_id}: {e}")

    # Convert to arrays for sklearn
    series_ids = list(features_dict.keys())
    X = pd.DataFrame([features_dict[sid] for sid in series_ids], index=series_ids)
    y = pd.Series([labels_dict[sid] for sid in series_ids], index=series_ids)

    print(f"Original dataset: {len(X)} samples")
    print(f"Positive class ratio: {y.mean():.2%}")

    # Perform K-fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores_augmented = []
    cv_scores_baseline = []
    train_scores_augmented = []
    train_scores_baseline = []

    augmenter = TimeSeriesAugmenter(
        augmentation_factor=augmentation_factor,
        strategy=augmentation_strategy
    )

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'=' * 60}")

        # Get train and test IDs
        train_ids = [series_ids[i] for i in train_idx]
        test_ids = [series_ids[i] for i in test_idx]

        print(f"Train: {len(train_ids)} series, Test: {len(test_ids)} series")

        # Test set (NEVER augmented)
        X_test = X.loc[test_ids]
        y_test = y.loc[test_ids]

        # Baseline: Train without augmentation
        X_train_baseline = X.loc[train_ids]
        y_train_baseline = y.loc[train_ids]

        # Train baseline model
        model_baseline = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        model_baseline.fit(X_train_baseline, y_train_baseline)

        # Evaluate baseline
        baseline_train_score = model_baseline.score(X_train_baseline, y_train_baseline)
        baseline_test_score = model_baseline.score(X_test, y_test)

        cv_scores_baseline.append(baseline_test_score)
        train_scores_baseline.append(baseline_train_score)

        print(f"Baseline - Train: {baseline_train_score:.4f}, Test: {baseline_test_score:.4f}")

        # Augmented: Augment ONLY the training data
        print(f"Augmenting training data ({augmentation_factor}x)...")

        augmented_features = []
        augmented_labels = []
        augmented_ids = []

        for train_id in tqdm(train_ids, desc="Augmenting"):
            ts_data = all_train_data[train_id]

            # Generate augmented versions
            augmented_series = augmenter.augment_time_series(ts_data)

            # Extract features from each augmented version
            for aug_ts in augmented_series:
                try:
                    features = feature_extractor.extract_features(aug_ts)
                    augmented_features.append(features)
                    augmented_labels.append(aug_ts.has_break)
                    augmented_ids.append(aug_ts.series_id)
                except Exception as e:
                    logger.warning(f"Failed to extract features for augmented {aug_ts.series_id}: {e}")

        # Create augmented training set
        X_train_augmented = pd.DataFrame(augmented_features, index=augmented_ids)
        y_train_augmented = pd.Series(augmented_labels, index=augmented_ids)

        print(f"Augmented training set: {len(X_train_augmented)} samples")

        # Train augmented model
        model_augmented = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        model_augmented.fit(X_train_augmented, y_train_augmented)

        # Evaluate augmented model on ORIGINAL test set
        augmented_train_score = model_augmented.score(X_train_augmented, y_train_augmented)
        augmented_test_score = model_augmented.score(X_test, y_test)

        cv_scores_augmented.append(augmented_test_score)
        train_scores_augmented.append(augmented_train_score)

        print(f"Augmented - Train: {augmented_train_score:.4f}, Test: {augmented_test_score:.4f}")

    # Compute final statistics
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")

    print("\nBaseline (No Augmentation):")
    print(f"  CV Test Score:  {np.mean(cv_scores_baseline):.4f} ± {np.std(cv_scores_baseline):.4f}")
    print(f"  CV Train Score: {np.mean(train_scores_baseline):.4f} ± {np.std(train_scores_baseline):.4f}")
    print(f"  Overfitting Gap: {np.mean(train_scores_baseline) - np.mean(cv_scores_baseline):.4f}")

    print(f"\nAugmented ({augmentation_factor}x with {augmentation_strategy}):")
    print(f"  CV Test Score:  {np.mean(cv_scores_augmented):.4f} ± {np.std(cv_scores_augmented):.4f}")
    print(f"  CV Train Score: {np.mean(train_scores_augmented):.4f} ± {np.std(train_scores_augmented):.4f}")
    print(f"  Overfitting Gap: {np.mean(train_scores_augmented) - np.mean(cv_scores_augmented):.4f}")

    print(f"\nImprovement from Augmentation:")
    print(f"  Test Score Improvement: {np.mean(cv_scores_augmented) - np.mean(cv_scores_baseline):+.4f}")
    print(
        f"  Overfitting Reduction: {(np.mean(train_scores_baseline) - np.mean(cv_scores_baseline)) - (np.mean(train_scores_augmented) - np.mean(cv_scores_augmented)):+.4f}")

    return {
        'baseline_test': np.mean(cv_scores_baseline),
        'baseline_train': np.mean(train_scores_baseline),
        'augmented_test': np.mean(cv_scores_augmented),
        'augmented_train': np.mean(train_scores_augmented),
    }


def compare_augmentation_strategies():
    """
    Compare different augmentation strategies
    """
    strategies = ['window_shift', 'subsample', 'noise']
    augmentation_factors = [2, 3, 5]

    results = {}

    for strategy in strategies:
        for factor in augmentation_factors:
            print(f"\n{'=' * 80}")
            print(f"Testing: {strategy} with {factor}x augmentation")
            print(f"{'=' * 80}")

            result = evaluate_with_proper_cv(
                augmentation_factor=factor,
                augmentation_strategy=strategy,
                n_folds=5
            )

            results[f"{strategy}_{factor}x"] = result

    # Summary table
    print(f"\n{'=' * 80}")
    print("COMPARISON OF ALL STRATEGIES")
    print(f"{'=' * 80}")
    print(f"{'Strategy':<20} {'Test AUC':<12} {'Overfit Gap':<12} {'Improvement':<12}")
    print(f"{'-' * 80}")

    baseline_score = None
    for name, result in results.items():
        test_score = result['augmented_test']
        overfit = result['augmented_train'] - result['augmented_test']

        if baseline_score is None:
            baseline_score = result['baseline_test']

        improvement = test_score - baseline_score

        print(f"{name:<20} {test_score:<12.4f} {overfit:<12.4f} {improvement:+12.4f}")

    return results


if __name__ == "__main__":
    print(f"Evaluating augmentation with proper CV (no data leakage)")
    print(f"Using {N_SAMPLES_TO_USE} samples")
    print(f"{'=' * 60}\n")

    # Test single configuration
    evaluate_with_proper_cv(augmentation_factor=3, augmentation_strategy='window_shift')

    # Uncomment to compare all strategies
    # compare_augmentation_strategies()
# scripts/evaluate_model.py
"""Script to evaluate model performance"""
import sys

sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path
import logging
from typing import List, Iterable

from main import infer
from src.data.dataLoader import StructuralBreakDataLoader
from src.utils.visualization import plot_model_performance, plot_time_series_with_break

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset_generator(X_data: pd.DataFrame, series_ids: List[int]) -> Iterable[pd.DataFrame]:
    """
    Create a generator that yields individual series DataFrames.

    Args:
        X_data: MultiIndex DataFrame with all data
        series_ids: List of series IDs to yield

    Yields:
        pd.DataFrame: Individual series data
    """
    for series_id in series_ids:
        yield X_data.loc[series_id]


def evaluate_model(model_path: str = "resources"):
    """Evaluate model on validation set"""

    # Load data
    loader = StructuralBreakDataLoader("../data")
    X_train, y_train, X_test = loader.load_data(use_crunch=False)

    # Create train/val split
    train_dict, val_dict = loader.create_train_val_split(val_ratio=0.2)

    # Prepare validation data
    val_ids = list(val_dict.keys())
    X_val = X_train.loc[val_ids]
    y_val = y_train.loc[val_ids]

    # Create generator for validation data
    val_generator = create_dataset_generator(X_train, val_ids)

    # Get predictions using the generator pattern
    logger.info("Getting predictions...")
    predictions_generator = infer(val_generator, model_path)

    # Consume the first yield (readiness signal)
    next(predictions_generator)

    # Collect all predictions
    predictions = []
    for i, prediction in enumerate(predictions_generator):
        predictions.append(prediction)
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1} series...")

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'prediction': predictions
    }, index=val_ids)

    y_pred = predictions_df['prediction'].values

    # Calculate metrics
    auc_score = roc_auc_score(y_val, y_pred)
    logger.info(f"Validation AUC: {auc_score:.4f}")

    # Binary predictions for classification report
    y_pred_binary = (y_pred > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_binary, target_names=['No Break', 'Break']))

    # Plot performance
    plot_model_performance(y_val.values, y_pred, save_path="experiments/results/roc_curve.png")

    return auc_score


def quick_test_inference():
    """Quick test to verify inference is working"""
    # Load a small sample
    loader = StructuralBreakDataLoader("../data")
    X_train, y_train, _ = loader.load_data(use_crunch=False)

    # Get first 5 series
    test_ids = X_train.index.get_level_values('id').unique()[:5]

    # Create generator
    test_generator = create_dataset_generator(X_train, test_ids)

    # Run inference
    predictions_gen = infer(test_generator, "../resources")

    # Get predictions
    next(predictions_gen)  # Skip readiness signal
    predictions = list(predictions_gen)

    print(f"Test predictions: {predictions}")
    print(f"Number of predictions: {len(predictions)}")

    return predictions


if __name__ == "__main__":
    # First do a quick test
    print("Running quick test...")
    try:
        quick_test_inference()
        print("Quick test passed!\n")
    except Exception as e:
        print(f"Quick test failed: {e}\n")

    # Run full evaluation
    print("Running full evaluation...")
    evaluate_model("../resources")
# /main.py
"""
Main submission file for ADIA Lab Structural Break Challenge
This file must contain train() and infer() functions as required by CrunchDAO
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Iterable
import scipy.stats as st

from src.features.extract_autocorrelation_features import extract_autocorrelation_features
from src.features.extract_distribution_features import extract_distribution_features

# Set random seed for reproducibility (REQUIRED for deterministic output)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_directory_path: str,
):
    """
    Train the structural break detection model.

    Args:
        X_train: Training data with MultiIndex (id, time) and columns [value, period]
        y_train: Boolean series indicating structural breaks
        model_directory_path: Path to save the trained model
    """
    logger.info("Starting training process...")

    try:
        from xgboost import XGBClassifier
        use_xgboost = True
    except ImportError:
        logger.warning("XGBoost not available, using Random Forest instead")
        from sklearn.ensemble import RandomForestClassifier
        use_xgboost = False

    # Extract features for all training series
    feature_list = []
    labels = []
    series_ids = []

    # Get unique series IDs
    unique_ids = X_train.index.get_level_values('id').unique()

    for series_id in unique_ids:
        try:
            series_data = X_train.loc[series_id]
            distribution_features = extract_distribution_features(series_data)
            autocorrelation_features = extract_autocorrelation_features(series_data)

            total_features = distribution_features | autocorrelation_features

            # Convert to list for DataFrame
            feature_list.append(total_features)
            labels.append(y_train.loc[series_id])
            series_ids.append(series_id)
        except Exception as e:
            logger.warning(f"Error processing series {series_id}: {e}")
            continue

    # Create feature DataFrame
    feature_df = pd.DataFrame(feature_list, index=series_ids)

    # Train model
    if use_xgboost:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_SEED
        )

    # Fit the model
    model.fit(feature_df, labels)

    # Save model and metadata
    model_data = {
        'model': model,
        'feature_names': list(feature_df.columns),
        'feature_stats': {
            'mean': feature_df.mean().to_dict(),
            'std': feature_df.std().to_dict()
        },
        'model_type': 'xgboost' if use_xgboost else 'random_forest'
    }

    model_path = os.path.join(model_directory_path, 'model.joblib')
    joblib.dump(model_data, model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Training completed. Features used: {len(feature_df.columns)}")
    logger.info(f"Model type: {model_data['model_type']}")


def infer(
        X_test: Iterable[pd.DataFrame],
        model_directory_path: str,
):
    """
    Make predictions on test data using a generator pattern.

    Args:
        X_test: Iterable of test DataFrames
        model_directory_path: Path to load the trained model

    Yields:
        float: Prediction for each dataset (probability between 0 and 1)
    """
    logger.info("Starting inference process...")

    # Load model
    model_path = os.path.join(model_directory_path, 'model.joblib')

    try:
        with open(model_path, 'rb') as f:
            model_data = joblib.load(f)

        model = model_data['model']
        feature_names = model_data['feature_names']
        logger.info(f"Model loaded successfully. Type: {model_data.get('model_type', 'unknown')}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Falling back to baseline t-test approach")
        model = None
        feature_names = None

    # Mark as ready
    yield

    # Process each dataset
    for dataset in X_test:
        try:
            if model is not None and feature_names is not None:
                # Extract features
                distribution_features = extract_distribution_features(dataset)
                autocorrelation_features = extract_autocorrelation_features(dataset)

                total_features = distribution_features | autocorrelation_features

                # Create feature vector in correct order
                feature_vector = [total_features.get(fname, 0) for fname in feature_names]

                # Predict probability
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba([feature_vector])[0, 1]
                else:
                    # For models without predict_proba
                    prob = float(model.predict([feature_vector])[0])
            else:
                # Fallback: Use t-test baseline
                period_0_values = dataset[dataset["period"] == 0]["value"]
                period_1_values = dataset[dataset["period"] == 1]["value"]

                if len(period_0_values) > 1 and len(period_1_values) > 1:
                    _, p_value = st.ttest_ind(period_0_values, period_1_values)
                    # Convert p-value to probability (smaller p-value = higher probability of break)
                    prob = 1.0 - p_value
                else:
                    prob = 0.5  # Default if not enough data

            # Ensure probability is in [0, 1]
            prob = np.clip(prob, 0.0, 1.0)

            yield float(prob)

        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            # Yield default probability on error
            yield 0.5


# For local testing
if __name__ == "__main__":
    print("Main submission file ready")
    print("Use 'crunch test' to test locally")
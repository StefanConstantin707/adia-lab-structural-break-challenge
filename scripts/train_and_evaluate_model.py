# scripts/optimize_xgboost.py
"""Script to evaluate model performance"""
from sklearn.model_selection import StratifiedKFold, cross_validate
from tqdm import tqdm
from xgboost import XGBClassifier

from src.features.extract_autocorrelation_features import AutocorrelationFeatureExtractor
from src.features.extract_changepoint_features import ChangepointFeatureExtractor
from src.features.extract_cnn_autoencoder_components import CNNBreakDetectionExtractor
from src.features.extract_cusum_breakpoint_features import CusumBreakpointFeatureExtractor
from src.features.extract_factor_break_features import FactorBreakFeatureExtractor

import numpy as np
import pandas as pd

import logging

from src.data.dataLoader import StructuralBreakDataLoader
from src.features.extract_information_features import InformationFeatureExtractor
from src.features.extract_regression_breakpoint_features import RegressionBreakpointFeatureExtractor
from src.features.extract_rolling_features import RollingFeatureExtractor
from src.features.extract_spectral_features import SpectralFeatureExtractor
from src.features.extract_volatility_features import VolatilityFeatureExtractor
from src.features.extract_distribution_features import DistributionCombinedFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_extract_features():
    """
    Load data via StructuralBreakDataLoader, extract features for each series,
    and return feature_df (DataFrame) and label_series (Series).
    """
    data_handler = StructuralBreakDataLoader()
    data_handler.load_data(use_crunch=False)

    train_data_dict = data_handler.get_all_train_series()
    feature_list = []
    labels = []
    series_ids = []

    print(len(train_data_dict))

    volatilityExtractor = VolatilityFeatureExtractor()
    spectralExtractor = SpectralFeatureExtractor()
    rollingExtractor = RollingFeatureExtractor()
    regressionBreakpointExtractor = RegressionBreakpointFeatureExtractor()
    informationExtractor = InformationFeatureExtractor()
    factorBreakExtractor = FactorBreakFeatureExtractor()
    cusumBreakpointExtractor = CusumBreakpointFeatureExtractor()
    changepointExtractor = ChangepointFeatureExtractor()
    autocorrelationExtractor = AutocorrelationFeatureExtractor()
    distributionCombinedExtractor = DistributionCombinedFeatureExtractor()

    # cnnBreakDetectionExtractor = CNNBreakDetectionExtractor(check_same=False)

    for series_id, ts_obj in tqdm(train_data_dict.items()):
        try:
            # 0.626
            volatility_features = volatilityExtractor.extract_features(ts_obj)

            # 0.582
            spectral_features = spectralExtractor.extract_features(ts_obj)

            # 0.6
            rolling_features = rollingExtractor.extract_features(ts_obj)

            # 0.618
            regression_breakpoint_features = regressionBreakpointExtractor.extract_features(ts_obj)

            # 0.544
            information_features = informationExtractor.extract_features(ts_obj)

            # 0.522
            factor_break_features = factorBreakExtractor.extract_features(ts_obj)

            # 0.64
            cusum_breakpoint_features = cusumBreakpointExtractor.extract_features(ts_obj)

            # 0.573
            changepoint_features = changepointExtractor.extract_features(ts_obj)

            # 0.517
            autocorrelation_features = autocorrelationExtractor.extract_features(ts_obj)

            # 0.69
            distribution_combined_features = distributionCombinedExtractor.extract_features(ts_obj)

            # cnn_break_detection_features = cnnBreakDetectionExtractor.extract_features(ts_obj)

            total_feats = (volatility_features | spectral_features | rolling_features | regression_breakpoint_features |
                           information_features | factor_break_features | cusum_breakpoint_features |
                           changepoint_features | autocorrelation_features | distribution_combined_features)

        except Exception as e:
            logger.warning(f"Failed feature extraction for series {series_id}: {e}")
            continue

        feature_list.append(total_feats)
        labels.append(ts_obj.has_break)
        series_ids.append(ts_obj.series_id)

    if not feature_list:
        raise RuntimeError("No features extracted; check data and extraction functions.")

    feature_df = pd.DataFrame(feature_list, index=series_ids)
    label_series = pd.Series(labels, index=series_ids, name='has_break')

    # Log feature count for monitoring
    logger.info(f"Extracted {len(feature_df.columns)} features from {len(feature_df)} series")

    return feature_df, label_series


def train_test(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Optuna objective: define hyperparameter space, perform 10-fold CV on X_train/y_train,
    record mean and std of AUC, and return mean AUC.
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring='roc_auc',
        return_train_score=True,
        error_score=np.nan,
        n_jobs=-1
    )

    # Validation scores
    val_scores = cv_results['test_score']
    val_mean = float(np.nanmean(val_scores))
    val_std = float(np.nanstd(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0

    # Train scores
    train_scores = cv_results['train_score']
    train_mean = float(np.nanmean(train_scores))
    train_std = float(np.nanstd(train_scores, ddof=1)) if len(train_scores) > 1 else 0.0

    return val_mean, val_std, train_mean, train_std


def get_feature_importance(X: pd.DataFrame, y: pd.Series, top_n: int = 20):
    """
    Train a final model on all data and extract feature importances.
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
    )

    # Train on full dataset
    model.fit(X, y)

    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return feature_importance.head(top_n)


def t_features(random_state: int = 42):
    """
    Full pipeline: load data, extract features, split train/test, run Optuna search on train set,
    then evaluate best model on test set. Returns results dict.
    """
    logger.info("Loading data and extracting features...")
    X, y = load_and_extract_features()

    logger.info(f"Train shape: {X.shape}")

    cv_val_mean, cv_val_std, cv_train_mean, cv_train_std = train_test(X, y)

    logger.info(f"CV Train ROC AUC - Mean: {cv_train_mean:.4f}, Std: {cv_train_std:.4f}")
    logger.info(f"CV Val ROC AUC - Mean: {cv_val_mean:.4f}, Std: {cv_val_std:.4f}")

    # Print both for easy comparison
    print(f"\n=== Model Performance Summary ===")
    print(f"CV Train ROC AUC:  {cv_train_mean:.4f} ± {cv_train_std:.4f}")
    print(f"CV Val ROC AUC:    {cv_val_mean:.4f} ± {cv_val_std:.4f}")
    print(f"Overfitting gap:   {cv_train_mean - cv_val_mean:.4f}")

    # Get and print top 20 features
    print(f"\n=== Top 20 Most Important Features ===")
    top_features = get_feature_importance(X, y, top_n=20)

    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<50} {row['importance']:.6f}")

    print(f"\n=== Feature Importance Summary ===")
    print(f"Total features: {len(X.columns)}")
    print(f"Top 20 features account for {top_features['importance'].sum():.2%} of total importance")


if __name__ == "__main__":
    print("Running test evaluation...")
    t_features()
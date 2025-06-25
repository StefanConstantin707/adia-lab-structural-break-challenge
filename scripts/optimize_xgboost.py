# scripts/optimize_xgboost.py
"""Script to evaluate model performance"""
import sys

import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from statsmodels.sandbox.tools.cross_val import KFold
from xgboost import XGBClassifier

from src.features.extract_autocorrelation_features import extract_autocorrelation_features
from src.features.extract_changepoint_features import extract_changepoint_features
from src.features.extract_cusum_breakpoint_features import extract_cusum_breakpoint_features
from src.features.extract_distribution_features import extract_distribution_features, extract_distribution_distance_features, \
    extract_residual_distribution_features
from src.features.extract_information_features import extract_information_features
from src.features.extract_regression_breakpoint_features import extract_regression_breakpoint_features
from src.features.extract_rolling_features import extract_rolling_features
from src.features.extract_spectral_features import extract_spectral_features
from src.features.extract_volatility_features import extract_volatility_features

sys.path.append('.')

import numpy as np
import pandas as pd

import logging

from src.data.dataLoader import StructuralBreakDataLoader

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

    for series_id, ts_obj in train_data_dict.items():
        try:
            dist_feats = extract_distribution_features(ts_obj)
            ac_feats = extract_autocorrelation_features(ts_obj)
            distribution_distance_features = extract_distribution_distance_features(ts_obj)

            # print("Got initial features")

            lag = int(max(ac_feats["p0_num_sig_acf_lags"], ac_feats["p1_num_sig_acf_lags"]))

            residual_distribution_features = extract_residual_distribution_features(ts_obj, lags=lag)
            regression_breakpoint_features = extract_regression_breakpoint_features(ts_obj, lags=lag)
            cumsum_breakpoint_features = extract_cusum_breakpoint_features(ts_obj, lags=lag)

            # print("got lag features")

            # Tier 1 new features (highest impact)
            spectral_features = extract_spectral_features(ts_obj)
            information_features = extract_information_features(ts_obj)
            volatility_features = extract_volatility_features(ts_obj)

            # print("got t1 features")

            # Tier 2 new features (good additions)
            rolling_features = extract_rolling_features(ts_obj)
            changepoint_features = extract_changepoint_features(ts_obj)

            # print("got all features")

            # Combine all features
            total_feats = (dist_feats | ac_feats |
                           regression_breakpoint_features |
                           cumsum_breakpoint_features |
                           distribution_distance_features |
                           residual_distribution_features |
                           spectral_features |
                           information_features |
                           volatility_features |
                           rolling_features |
                           changepoint_features
                           )
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

    return feature_df, label_series


def objective(trial, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Optuna objective: define hyperparameter space, perform 10-fold CV on X_train/y_train,
    record mean and std of AUC, and return mean AUC.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,
    }
    model = XGBClassifier(**param)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cross_val_score returns array of scores for each fold
    try:
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='roc_auc',
            error_score=np.nan,
            n_jobs=-1
        )
    except Exception as e:
        # If CV fails, return a poor score
        logger.warning(f"CV failed in trial {trial.number}: {e}")
        trial.set_user_attr("train_auc_mean", np.nan)
        trial.set_user_attr("train_auc_std", np.nan)
        return 0.0

    mean_score = float(np.nanmean(scores))
    std_score = float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0

    # Record as user attributes
    trial.set_user_attr("train_auc_mean", mean_score)
    trial.set_user_attr("train_auc_std", std_score)

    return mean_score


def optimize_model(n_trials: int = 20, test_size: float = 0.2, random_state: int = 42):
    """
    Full pipeline: load data, extract features, split train/test, run Optuna search on train set,
    then evaluate best model on test set. Returns results dict.
    """
    logger.info("Loading data and extracting features...")
    X, y = load_and_extract_features()

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    logger.info(f"Starting Optuna study with {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_trial = study.best_trial
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best trial params: {best_trial.params}")
    train_auc_mean = best_trial.user_attrs.get("train_auc_mean", None)
    train_auc_std = best_trial.user_attrs.get("train_auc_std", None)
    if train_auc_mean is not None:
        logger.info(f"Best CV Train AUC mean: {train_auc_mean:.4f}, std: {train_auc_std:.4f}")

    # Retrain on full train set with best params
    best_params = best_trial.params.copy()
    best_params.update({
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': random_state,
        'verbosity': 0,
    })
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    class_counts = y_test.value_counts().min()
    if class_counts < 2:
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        test_auc_mean = roc_auc_score(y_test, y_pred_proba)
        test_auc_std = 0.0
        logger.info("Not enough samples per class in test set for CV; reporting single test AUC.")
    else:
        n_splits = min(10, class_counts)
        cv_test = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        test_scores = cross_val_score(
            best_model, X_test, y_test,
            cv=cv_test,
            scoring='roc_auc',
            error_score=np.nan,
            n_jobs=-1
        )
        test_auc_mean = float(np.nanmean(test_scores))
        test_auc_std = float(np.nanstd(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0
    logger.info(f"Test AUC mean: {test_auc_mean:.4f}, std: {test_auc_std:.4f}")

    results = {
        'best_params': best_trial.params,
        'train_cv_auc_mean': train_auc_mean,
        'train_cv_auc_std': train_auc_std,
        'test_auc_mean': test_auc_mean,
        'test_auc_std': test_auc_std,
    }
    return results


def quick_test_optimize_model():
    """Quick test of optimize_model with few trials"""
    logger.info("Running quick test of optimize_model with n_trials=2")
    try:
        res = optimize_model(n_trials=2)
        logger.info(f"Quick test results: {res}")
    except Exception as e:
        raise RuntimeError(f"Quick test failed: {e}")


def full_optimize_model():
    """Full run of optimize_model with n_trials=20"""
    logger.info("Running full optimization with n_trials=20")
    try:
        res = optimize_model(n_trials=20)
        logger.info(f"Full optimization results: {res}")
        return res
    except Exception as e:
        raise RuntimeError(f"Full optimization failed: {e}")

if __name__ == "__main__":
    # print("Running quick test...")
    # try:
    #     quick_test_optimize_model()
    #     print("Quick test passed!\n")
    # except Exception as e:
    #     print(f"Quick test failed: {e}\n")

    print("Running full evaluation...")
    try:
        res = full_optimize_model()
        print("Full optimization results:")
        print(f"Best params: {res['best_params']}")
        print(f"Train CV AUC mean: {res['train_cv_auc_mean']:.4f}, std: {res['train_cv_auc_std']:.4f}")
        print(f"Test AUC mean: {res['test_auc_mean']:.4f}, std: {res['test_auc_std']:.4f}")
    except Exception as e:
        print(f"Full evaluation failed: {e}")

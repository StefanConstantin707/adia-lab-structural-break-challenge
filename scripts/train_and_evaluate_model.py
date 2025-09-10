"""Script to evaluate model performance with ensemble methods, class weighting, and calibration"""
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    SelectFromModel,
    VarianceThreshold
)
from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from src.features.extract_autocorrelation_features import AutocorrelationFeatureExtractor
from src.features.extract_changepoint_features import ChangepointFeatureExtractor
from src.features.extract_cusum_breakpoint_features import CusumBreakpointFeatureExtractor

import numpy as np
import pandas as pd
import logging

from src.data.dataLoader import StructuralBreakDataLoader
from src.features.extract_garch_features import AdvancedVolatilityBreakFeatureExtractor
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
    # train_data_dict = data_handler.get_first_n_train_series(n=1000)
    feature_list = []
    labels = []
    series_ids = []

    print(f"Processing {len(train_data_dict)} series...")

    volatilityExtractor = VolatilityFeatureExtractor()
    spectralExtractor = SpectralFeatureExtractor()
    rollingExtractor = RollingFeatureExtractor()
    regressionBreakpointExtractor = RegressionBreakpointFeatureExtractor()
    cusumBreakpointExtractor = CusumBreakpointFeatureExtractor(
        adaptive_lags=True,
        max_lags=30,
        pacf_alpha=0.05,
        test_assumptions=True,
    )
    changepointExtractor = ChangepointFeatureExtractor()
    distributionCombinedExtractor = DistributionCombinedFeatureExtractor(lags=5)
    advancedVolatilityBreakFeatureExtractor = AdvancedVolatilityBreakFeatureExtractor(check_same=False)

    for series_id, ts_obj in tqdm(train_data_dict.items()):
        try:
            volatility_features = volatilityExtractor.extract_features(ts_obj)
            spectral_features = spectralExtractor.extract_features(ts_obj)
            rolling_features = rollingExtractor.extract_features(ts_obj)
            regression_breakpoint_features = regressionBreakpointExtractor.extract_features(ts_obj)
            cusum_breakpoint_features = cusumBreakpointExtractor.extract_features(ts_obj)
            changepoint_features = changepointExtractor.extract_features(ts_obj)
            distribution_combined_features = distributionCombinedExtractor.extract_features(ts_obj)
            advancedVolatilityBreak_features = advancedVolatilityBreakFeatureExtractor.extract_features(ts_obj)

            total_feats = (advancedVolatilityBreak_features | cusum_breakpoint_features |
                          distribution_combined_features | spectral_features | rolling_features |
                          regression_breakpoint_features | changepoint_features | volatility_features)

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

    # Handle missing values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median())

    logger.info(f"Extracted {len(feature_df.columns)} features from {len(feature_df)} series")

    return feature_df, label_series


def apply_feature_selection(X_train, y_train, method='tree', n_features=None):
    """
    Apply feature selection to reduce dimensionality.
    """

    if method == 'tree':
        tree_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )
        selector = SelectFromModel(tree_model, prefit=False, threshold='median')

    elif method == 'l1':
        lasso = LGBMClassifier(
            n_estimators=50,
            reg_alpha=0.1,
            reg_lambda=0,
            random_state=42,
            verbosity=-1
        )
        selector = SelectFromModel(lasso, prefit=False)

    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=min(n_features or 50, X_train.shape[1]))

    elif method == 'none':
        return None, X_train.columns.tolist()

    else:
        raise ValueError(f"Unknown method: {method}")

    # Fit selector
    selector.fit(X_train, y_train)

    # Get selected features
    if hasattr(selector, 'get_support'):
        selected_features = X_train.columns[selector.get_support()].tolist()
    else:
        selected_features = X_train.columns.tolist()

    return selector, selected_features


def get_class_weights(y):
    """Calculate class weights for imbalanced dataset"""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def create_base_models(class_weight_dict=None, use_class_weight=True):
    """Create a dictionary of base models with proper parameters"""

    models = {}

    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = class_weight_dict[True] / class_weight_dict[False] if use_class_weight and class_weight_dict else 1.0

    # XGBoost with class weighting
    models['xgb'] = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
        random_state=42,
        eval_metric='logloss',
    )

    # LightGBM with class weighting
    models['lgb'] = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        class_weight='balanced' if use_class_weight else None,
        random_state=42,
        verbosity=-1
    )

    # CatBoost with class weighting
    models['cat'] = CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.05,
        class_weights=class_weight_dict if use_class_weight else None,
        random_state=42,
        verbose=False
    )

    # Random Forest with class weighting
    models['rf'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced' if use_class_weight else None,
        random_state=42,
        n_jobs=-1
    )

    # Extra Trees with class weighting
    models['et'] = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced' if use_class_weight else None,
        random_state=42,
        n_jobs=-1
    )

    return models


def create_ensemble_models(class_weight_dict=None, use_class_weight=True):
    """Create various ensemble models"""

    base_models = create_base_models(class_weight_dict, use_class_weight)
    ensembles = {}

    # Voting Classifier - Soft voting (average probabilities)
    ensembles['voting_all'] = VotingClassifier(
        estimators=list(base_models.items()),
        voting='soft',
        n_jobs=-1
    )

    # Voting with just tree-based models
    ensembles['voting_trees'] = VotingClassifier(
        estimators=[
            ('xgb', base_models['xgb']),
            ('lgb', base_models['lgb']),
            ('cat', base_models['cat'])
        ],
        voting='soft',
        n_jobs=-1
    )

    # Stacking Classifier with Logistic Regression meta-learner
    ensembles['stacking_lr'] = StackingClassifier(
        estimators=[
            ('xgb', base_models['xgb']),
            ('lgb', base_models['lgb']),
            ('rf', base_models['rf'])
        ],
        final_estimator=LogisticRegression(
            class_weight='balanced' if use_class_weight else None,
            random_state=42
        ),
        cv=5,  # Use 5-fold CV to train meta-learner
        n_jobs=-1
    )

    # Stacking with XGBoost meta-learner
    scale_pos_weight = class_weight_dict[True] / class_weight_dict[False] if use_class_weight and class_weight_dict else 1.0
    ensembles['stacking_xgb'] = StackingClassifier(
        estimators=[
            ('lgb', base_models['lgb']),
            ('cat', base_models['cat']),
            ('rf', base_models['rf'])
        ],
        final_estimator=XGBClassifier(
            n_estimators=50,
            max_depth=3,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            random_state=42,
            eval_metric='logloss'
        ),
        cv=5,
        n_jobs=-1
    )

    return ensembles


def evaluate_model_with_cv(model, X, y, cv_folds=10, model_name="Model"):
    """Evaluate a model using cross-validation"""

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Get both train and validation scores
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scorer,
        return_train_score=True,
        n_jobs=-1,
        error_score='raise'
    )

    val_scores = cv_results['test_score']
    train_scores = cv_results['train_score']

    results = {
        'model': model_name,
        'val_mean': np.mean(val_scores),
        'val_std': np.std(val_scores, ddof=1),
        'train_mean': np.mean(train_scores),
        'train_std': np.std(train_scores, ddof=1),
        'overfit_gap': np.mean(train_scores) - np.mean(val_scores)
    }

    return results


def evaluate_calibrated_model(base_model, X, y, cv_folds=10, model_name="Model"):
    """Evaluate a model with probability calibration"""

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    val_scores = []
    train_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create calibrated classifier
        calibrated = CalibratedClassifierCV(
            base_model.__class__(**base_model.get_params()),
            method='isotonic',
            cv=3
        )

        # Fit and predict
        calibrated.fit(X_train, y_train)

        # Score
        train_proba = calibrated.predict_proba(X_train)[:, 1]
        val_proba = calibrated.predict_proba(X_val)[:, 1]

        train_scores.append(roc_auc_score(y_train, train_proba))
        val_scores.append(roc_auc_score(y_val, val_proba))

    results = {
        'model': f"{model_name}_calibrated",
        'val_mean': np.mean(val_scores),
        'val_std': np.std(val_scores, ddof=1),
        'train_mean': np.mean(train_scores),
        'train_std': np.std(train_scores, ddof=1),
        'overfit_gap': np.mean(train_scores) - np.mean(val_scores)
    }

    return results


def test_all_approaches(X, y, feature_selection='tree'):
    """Test all different approaches: base models, ensembles, calibration, class weighting"""

    results = []

    # Apply feature selection if specified
    if feature_selection != 'none':
        logger.info(f"Applying {feature_selection} feature selection...")
        selector, selected_features = apply_feature_selection(X, y, method=feature_selection)
        if selector:
            X_selected = pd.DataFrame(
                selector.transform(X),
                columns=selected_features,
                index=X.index
            )
            logger.info(f"Selected {len(selected_features)} features")
        else:
            X_selected = X
    else:
        X_selected = X

    # Get class weights
    class_weights = get_class_weights(y)
    logger.info(f"Class weights: {class_weights}")

    # Test configurations
    configs = [
        ('No class weight', False),
        ('With class weight', True)
    ]

    for config_name, use_weights in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print('='*60)

        # Create models
        base_models = create_base_models(class_weights, use_weights)
        ensemble_models = create_ensemble_models(class_weights, use_weights)

        # Test individual base models
        print("\nBase Models:")
        for name, model in base_models.items():
            try:
                result = evaluate_model_with_cv(model, X_selected, y, model_name=f"{name}_{config_name}")
                results.append(result)
                print(f"  {name:15} Val: {result['val_mean']:.4f} ± {result['val_std']:.4f}, "
                      f"Overfit: {result['overfit_gap']:.4f}")
            except Exception as e:
                print(f"  {name:15} Failed: {str(e)[:50]}")

        # Test ensemble models
        print("\nEnsemble Models:")
        for name, model in ensemble_models.items():
            try:
                result = evaluate_model_with_cv(model, X_selected, y, model_name=f"{name}_{config_name}")
                results.append(result)
                print(f"  {name:15} Val: {result['val_mean']:.4f} ± {result['val_std']:.4f}, "
                      f"Overfit: {result['overfit_gap']:.4f}")
            except Exception as e:
                print(f"  {name:15} Failed: {str(e)[:50]}")

        # Test calibrated versions of best individual models
        print("\nCalibrated Models:")
        for name in ['xgb', 'lgb']:
            try:
                model = base_models[name]
                result = evaluate_calibrated_model(model, X_selected, y,
                                                  model_name=f"{name}_{config_name}")
                results.append(result)
                print(f"  {name}_calibrated  Val: {result['val_mean']:.4f} ± {result['val_std']:.4f}, "
                      f"Overfit: {result['overfit_gap']:.4f}")
            except Exception as e:
                print(f"  {name}_calibrated  Failed: {str(e)[:50]}")

    return pd.DataFrame(results)


def get_feature_importance(X: pd.DataFrame, y: pd.Series, top_n: int = 20):
    """Train a final model on all data and extract feature importances"""

    class_weights = get_class_weights(y)
    scale_pos_weight = class_weights[True] / class_weights[False]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
    )

    model.fit(X, y)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return feature_importance.head(top_n)


def main():
    """Main pipeline with ensemble methods, class weighting, and calibration"""

    logger.info("Loading data and extracting features...")
    X, y = load_and_extract_features()

    logger.info(f"Original shape: {X.shape}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    logger.info(f"Class balance: {y.mean():.2%} positive")

    # Test all approaches
    print("\n" + "="*70)
    print("TESTING ALL APPROACHES: BASE MODELS, ENSEMBLES, CALIBRATION")
    print("="*70)

    results_df = test_all_approaches(X, y, feature_selection='tree')

    # Sort by validation score
    results_df = results_df.sort_values('val_mean', ascending=False)

    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY (sorted by validation ROC AUC)")
    print("="*70)
    print(results_df.to_string())

    # Best model analysis
    best_model = results_df.iloc[0]
    print(f"\n{'='*70}")
    print("BEST MODEL")
    print('='*70)
    print(f"Model: {best_model['model']}")
    print(f"Validation ROC AUC: {best_model['val_mean']:.4f} ± {best_model['val_std']:.4f}")
    print(f"Training ROC AUC: {best_model['train_mean']:.4f} ± {best_model['train_std']:.4f}")
    print(f"Overfitting Gap: {best_model['overfit_gap']:.4f}")

    # Compare to baseline
    baseline_models = results_df[results_df['model'].str.contains('xgb_No class weight')]
    if not baseline_models.empty:
        baseline = baseline_models.iloc[0]
        improvement = best_model['val_mean'] - baseline['val_mean']
        overfit_reduction = baseline['overfit_gap'] - best_model['overfit_gap']

        print(f"\n{'='*70}")
        print("IMPROVEMENT OVER BASELINE (XGB without class weight)")
        print('='*70)
        print(f"Validation Score Improvement: {improvement:+.4f}")
        print(f"Overfitting Reduction: {overfit_reduction:+.4f}")

    # Top features analysis
    print(f"\n{'='*70}")
    print("TOP 20 FEATURES (using best configuration)")
    print('='*70)

    # Apply feature selection for feature importance
    selector, selected_features = apply_feature_selection(X, y, method='tree')
    if selector:
        X_selected = pd.DataFrame(
            selector.transform(X),
            columns=selected_features,
            index=X.index
        )
    else:
        X_selected = X

    top_features = get_feature_importance(X_selected, y, top_n=20)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<50} {row['importance']:.6f}")


if __name__ == "__main__":
    print("Running comprehensive model evaluation...")
    main()
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import logging
from tqdm import tqdm

from config import DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data.dataLoader import StructuralBreakDataLoader, TimeSeriesData


class TimeSeriesModelTrainer:
    """
    A comprehensive training class for time series break detection models.

    This class handles feature extraction, model training, cross-validation,
    and diagnostic plotting for time series classification tasks.
    """

    def __init__(self,
                 feature_extraction_func: Callable[[TimeSeriesData], pd.DataFrame],
                 model_params: Optional[Dict[str, Any]] = None,
                 cv_folds: int = 10,
                 random_state: int = 42,
                 data_dir: Optional[Union[str, Path]] = DATA_DIR,
                 use_crunch: bool = False):
        """
        Initialize the trainer.

        Args:
            feature_extraction_func: Function that takes TimeSeriesData and returns DataFrame of features
            model_params: XGBoost parameters (optional)
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            data_dir: Directory containing data files (for auto-loading)
            use_crunch: Whether to use crunch module for data loading (for auto-loading)
        """
        self.feature_extraction_func = feature_extraction_func
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.data_dir = data_dir
        self.use_crunch = use_crunch

        # Default XGBoost parameters
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': random_state,
            'eval_metric': 'logloss',
        }

        # Results storage
        self.results = {}
        self.model = None
        self.feature_df = None
        self.labels = None
        self.data_loader = None

    def load_data_automatically(self) -> Dict[int, TimeSeriesData]:
        """
        Automatically load data using StructuralBreakDataLoader.

        Returns:
            Dictionary with series_id as key and TimeSeriesData as value
        """
        if StructuralBreakDataLoader is None:
            raise ImportError("StructuralBreakDataLoader not available. Please provide data manually.")

        logger.info("Loading data automatically using StructuralBreakDataLoader...")
        self.data_loader = StructuralBreakDataLoader(data_dir=self.data_dir)
        self.data_loader.load_data(use_crunch=self.use_crunch)

        train_data_dict = self.data_loader.get_all_train_series()
        logger.info(f"Loaded {len(train_data_dict)} training time series automatically")

        return train_data_dict

    def extract_features_from_data(self, timeseries_data: Dict[int, TimeSeriesData]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features from dictionary of TimeSeriesData objects using the provided function.

        Args:
            timeseries_data: Dictionary with series_id as key and TimeSeriesData as value

        Returns:
            Tuple of (feature_df, labels)
        """
        feature_list = []
        labels = []
        series_ids = []

        logger.info(f"Extracting features from {len(timeseries_data)} time series...")

        for series_id, ts_data in tqdm(timeseries_data.items(), desc="Extracting features"):
            try:
                # Extract features using provided function
                features = self.feature_extraction_func(ts_data)

                # Handle different return types
                if isinstance(features, pd.DataFrame):
                    if len(features) == 1:
                        features = features.iloc[0].to_dict()
                    else:
                        features = features.to_dict('records')[0]
                elif isinstance(features, pd.Series):
                    features = features.to_dict()

                feature_list.append(features)
                labels.append(ts_data.has_break)
                series_ids.append(series_id)

            except Exception as e:
                logger.warning(f"Failed feature extraction for series {series_id}: {e}")
                continue

        if not feature_list:
            raise RuntimeError("No features extracted; check data and extraction functions.")

        feature_df = pd.DataFrame(feature_list, index=series_ids)
        label_series = pd.Series(labels, index=series_ids, name='has_break')

        logger.info(f"Successfully extracted {len(feature_df.columns)} features from {len(feature_df)} series")

        return feature_df, label_series

    def train_and_evaluate(self, timeseries_data: Optional[Dict[int, TimeSeriesData]] = None) -> Dict[str, Any]:
        """
        Main training pipeline: extract features, train model, and evaluate performance.

        Args:
            timeseries_data: Optional dictionary with series_id as key and TimeSeriesData as value.
                           If None, will attempt to load data automatically using StructuralBreakDataLoader.

        Returns:
            Dictionary containing all evaluation results
        """
        # Load data if not provided
        if timeseries_data is None:
            timeseries_data = self.load_data_automatically()

        # Extract features
        self.feature_df, self.labels = self.extract_features_from_data(timeseries_data)

        logger.info(f"Dataset shape: {self.feature_df.shape}")
        logger.info(f"Class distribution: {self.labels.value_counts().to_dict()}")

        # Check for missing values
        if self.feature_df.isnull().sum().sum() > 0:
            logger.warning("Found missing values in features. Filling with 0.")
            self.feature_df = self.feature_df.fillna(0)

        # Initialize model
        self.model = XGBClassifier(**self.model_params)

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Perform cross-validation
        logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
        cv_results = cross_validate(
            self.model, self.feature_df, self.labels,
            cv=cv,
            scoring='roc_auc',
            return_train_score=True,
            error_score=np.nan,
            n_jobs=-1
        )

        # Calculate metrics
        val_scores = cv_results['test_score']
        train_scores = cv_results['train_score']

        # Get cross-validated predictions for ROC curve
        y_pred_proba = cross_val_predict(
            XGBClassifier(**self.model_params),
            self.feature_df, self.labels,
            cv=cv, method='predict_proba'
        )[:, 1]

        # Train final model on all data for feature importance
        logger.info("Training final model on full dataset...")
        self.model.fit(self.feature_df, self.labels)

        # Store results
        results = {
            'cv_val_mean': float(np.nanmean(val_scores)),
            'cv_val_std': float(np.nanstd(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0,
            'cv_train_mean': float(np.nanmean(train_scores)),
            'cv_train_std': float(np.nanstd(train_scores, ddof=1)) if len(train_scores) > 1 else 0.0,
            'val_scores': val_scores,
            'train_scores': train_scores,
            'y_true': self.labels.values,
            'y_pred_proba': y_pred_proba,
            'feature_count': len(self.feature_df.columns),
            'sample_count': len(self.feature_df),
            'class_distribution': self.labels.value_counts().to_dict()
        }

        self.results = results
        logger.info("Training completed successfully!")

        return results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_and_evaluate first.")

        feature_importance = pd.DataFrame({
            'feature': self.feature_df.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance.head(top_n)

    def plot_diagnostics(self, figsize: Tuple[int, int] = (16, 12), save_path: Optional[str] = None):
        """
        Plot comprehensive diagnostic plots.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the plot
        """
        if not self.results:
            raise ValueError("No results available. Run train_and_evaluate first.")

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Model Training Diagnostics', fontsize=16, fontweight='bold')

        # 1. ROC Curve
        self._plot_roc_curve(axes[0, 0])

        # 2. Feature Importance
        self._plot_feature_importance(axes[0, 1])

        # 3. CV Score Distribution
        self._plot_cv_scores(axes[0, 2])

        # 4. Confusion Matrix
        self._plot_confusion_matrix(axes[1, 0])

        # 5. Prediction Distribution
        self._plot_prediction_distribution(axes[1, 1])

        # 6. Performance Summary (text)
        self._plot_performance_summary(axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diagnostic plots saved to {save_path}")

        plt.show()

    def _plot_roc_curve(self, ax):
        """Plot ROC curve with AUC score"""
        fpr, tpr, _ = roc_curve(self.results['y_true'], self.results['y_pred_proba'])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self, ax, top_n: int = 15):
        """Plot top feature importances"""
        importance_df = self.get_feature_importance(top_n)

        # Truncate long feature names
        feature_names = [f[:25] + '...' if len(f) > 25 else f for f in importance_df['feature']]

        y_pos = np.arange(len(importance_df))
        bars = ax.barh(y_pos, importance_df['importance'], alpha=0.8, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', ha='left', va='center', fontsize=7)

    def _plot_cv_scores(self, ax):
        """Plot cross-validation scores across folds"""
        train_scores = self.results['train_scores']
        val_scores = self.results['val_scores']

        folds = np.arange(1, len(train_scores) + 1)

        ax.plot(folds, train_scores, 'o-', label='Train', linewidth=2, markersize=6)
        ax.plot(folds, val_scores, 's-', label='Validation', linewidth=2, markersize=6)

        # Add mean lines
        ax.axhline(y=np.mean(train_scores), color='blue', linestyle='--', alpha=0.7)
        ax.axhline(y=np.mean(val_scores), color='orange', linestyle='--', alpha=0.7)

        ax.set_xlabel('CV Fold')
        ax.set_ylabel('ROC AUC Score')
        ax.set_title('Cross-Validation Scores by Fold')
        ax.set_xticks(folds)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add text annotations for means
        ax.text(0.02, 0.98, f'Train Mean: {np.mean(train_scores):.3f}',
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.92, f'Val Mean: {np.mean(val_scores):.3f}',
                transform=ax.transAxes, verticalalignment='top')

    def _plot_confusion_matrix(self, ax):
        """Plot confusion matrix"""
        # Convert probabilities to binary predictions using 0.5 threshold
        y_pred = (self.results['y_pred_proba'] > 0.5).astype(int)
        cm = confusion_matrix(self.results['y_true'], y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Break', 'Break'],
                    yticklabels=['No Break', 'Break'])
        ax.set_title('Confusion Matrix\n(Threshold = 0.5)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    def _plot_prediction_distribution(self, ax):
        """Plot distribution of predicted probabilities by class"""
        y_true = self.results['y_true']
        y_pred_proba = self.results['y_pred_proba']

        # Separate predictions by true class
        no_break_probs = y_pred_proba[y_true == 0]
        break_probs = y_pred_proba[y_true == 1]

        ax.hist(no_break_probs, bins=30, alpha=0.7, label='No Break (True)', color='blue', density=True)
        ax.hist(break_probs, bins=30, alpha=0.7, label='Break (True)', color='red', density=True)

        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Decision Boundary')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Probability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_summary(self, ax):
        """Display performance summary as text"""
        ax.axis('off')

        # Calculate additional metrics
        overfitting_gap = self.results['cv_train_mean'] - self.results['cv_val_mean']
        top_features = self.get_feature_importance(20)
        top_20_importance = top_features['importance'].sum()

        summary_text = f"""
MODEL PERFORMANCE SUMMARY
{"=" * 40}

Cross-Validation Results:
• Train ROC AUC: {self.results['cv_train_mean']:.4f} ± {self.results['cv_train_std']:.4f}
• Val ROC AUC: {self.results['cv_val_mean']:.4f} ± {self.results['cv_val_std']:.4f}
• Overfitting gap: {overfitting_gap:.4f}

Dataset Information:
• Total samples: {self.results['sample_count']:,}
• Total features: {self.results['feature_count']:,}
• Class distribution: {self.results['class_distribution']}

Feature Importance:
• Top 20 features account for: {top_20_importance:.1%}
• Most important feature: {top_features.iloc[0]['feature'][:30]}...

Model Configuration:
• CV Folds: {self.cv_folds}
• Random State: {self.random_state}
• XGBoost n_estimators: {self.model_params['n_estimators']}
• XGBoost max_depth: {self.model_params['max_depth']}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    def print_detailed_results(self):
        """Print comprehensive results to console"""
        if not self.results:
            raise ValueError("No results available. Run train_and_evaluate first.")

        print("\n" + "=" * 70)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("=" * 70)

        # Performance metrics
        print(f"\nCROSS-VALIDATION PERFORMANCE:")
        print(f"{'Metric':<20} {'Train':<15} {'Validation':<15} {'Gap':<10}")
        print("-" * 60)
        train_mean = self.results['cv_train_mean']
        val_mean = self.results['cv_val_mean']
        gap = train_mean - val_mean
        print(f"{'ROC AUC':<20} {train_mean:.4f} ± {self.results['cv_train_std']:<6.4f} "
              f"{val_mean:.4f} ± {self.results['cv_val_std']:<6.4f} {gap:.4f}")

        # Dataset info
        print(f"\nDATASET INFORMATION:")
        print(f"• Total samples: {self.results['sample_count']:,}")
        print(f"• Total features: {self.results['feature_count']:,}")
        print(f"• Class distribution: {self.results['class_distribution']}")

        # Top features
        print(f"\nTOP 20 MOST IMPORTANT FEATURES:")
        print("-" * 70)
        top_features = self.get_feature_importance(20)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<50} {row['importance']:.6f}")

        print(f"\nFEATURE IMPORTANCE SUMMARY:")
        print(f"• Top 20 features account for {top_features['importance'].sum():.1%} of total importance")
        print(f"• Most important feature: {top_features.iloc[0]['feature']}")
        print(f"• Least important in top 20: {top_features.iloc[-1]['feature']}")

        # Model configuration
        print(f"\nMODEL CONFIGURATION:")
        for key, value in self.model_params.items():
            print(f"• {key}: {value}")

        print("=" * 70)

    def get_train_val_split(self, val_ratio: float = 0.2, data_ratio: float = 1.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create train/validation split using the data loader.

        Args:
            val_ratio: Fraction to use for validation
            data_ratio: Fraction of total data to use (useful for testing on smaller datasets)

        Returns:
            Tuple of (train_results, val_results) after training on each split
        """
        if self.data_loader is None:
            raise ValueError("Data loader not available. Load data first using load_data_automatically() or train_and_evaluate().")

        train_dict, val_dict = self.data_loader.create_train_val_split(
            val_ratio=val_ratio,
            data_ratio=data_ratio,
            random_state=self.random_state
        )

        logger.info(f"Created train/val split: {len(train_dict)} train, {len(val_dict)} validation")

        # Train on training split
        logger.info("Training on training split...")
        train_results = self.train_and_evaluate(train_dict)

        # Evaluate on validation split
        logger.info("Evaluating on validation split...")
        val_features, val_labels = self.extract_features_from_data(val_dict)

        if val_features.isnull().sum().sum() > 0:
            val_features = val_features.fillna(0)

        # Align features (in case validation set has different features)
        common_features = self.feature_df.columns.intersection(val_features.columns)
        if len(common_features) < len(self.feature_df.columns):
            logger.warning(f"Validation set missing {len(self.feature_df.columns) - len(common_features)} features")

        val_features_aligned = val_features[common_features]
        val_pred_proba = self.model.predict_proba(val_features_aligned)[:, 1]
        val_auc = roc_auc_score(val_labels, val_pred_proba)

        val_results = {
            'val_auc': val_auc,
            'val_features': val_features_aligned,
            'val_labels': val_labels,
            'val_pred_proba': val_pred_proba
        }

        logger.info(f"Validation AUC: {val_auc:.4f}")

        return train_results, val_results

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if self.data_loader is None:
            raise ValueError("Data loader not available. Load data first.")

        return self.data_loader.get_statistics()

    def get_prediction_results(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Get detailed prediction results with custom threshold.

        Args:
            threshold: Classification threshold (default 0.5)

        Returns:
            Dictionary with detailed prediction metrics
        """
        if not self.results:
            raise ValueError("No results available. Run train_and_evaluate first.")

        y_true = self.results['y_true']
        y_pred_proba = self.results['y_pred_proba']
        y_pred = (y_pred_proba > threshold).astype(int)

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        accuracy = accuracy_score(y_true, y_pred)

        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision_no_break': precision[0],
            'precision_break': precision[1],
            'recall_no_break': recall[0],
            'recall_break': recall[1],
            'f1_no_break': f1[0],
            'f1_break': f1[1],
            'support_no_break': support[0],
            'support_break': support[1],
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }


# Example usage and integration:
def example_feature_extraction_function(ts_data: TimeSeriesData) -> pd.DataFrame:
    """
    Example feature extraction function.
    Replace this with your actual feature extraction logic.
    """
    features = {
        'mean_value': np.mean(ts_data.values),
        'std_value': np.std(ts_data.values),
        'min_value': np.min(ts_data.values),
        'max_value': np.max(ts_data.values),
        'length': ts_data.length,
        'period_0_mean': np.mean(ts_data.period_0_values),
        'period_1_mean': np.mean(ts_data.period_1_values),
        'period_diff': np.mean(ts_data.period_1_values) - np.mean(ts_data.period_0_values),
    }

    return pd.DataFrame([features])


def run_complete_example():
    """Example showing how to use the trainer with automatic data loading"""

    # Method 1: Automatic data loading
    trainer = TimeSeriesModelTrainer(
        feature_extraction_func=example_feature_extraction_function,
        cv_folds=10,
        random_state=42,
        use_crunch=False  # Set to True if you have crunch available
    )

    # Train and evaluate (will automatically load data)
    results = trainer.train_and_evaluate()

    # Print comprehensive results
    trainer.print_detailed_results()

    # Plot diagnostics
    trainer.plot_diagnostics()

    # Get dataset statistics
    stats = trainer.get_data_statistics()
    print(f"Dataset statistics: {stats}")

    return trainer


def run_manual_data_example():
    """Example showing how to use the trainer with manually loaded data"""

    # Load data manually
    data_loader = StructuralBreakDataLoader()
    data_loader.load_data(use_crunch=False)
    train_data_dict = data_loader.get_all_train_series()

    # Create trainer
    trainer = TimeSeriesModelTrainer(
        feature_extraction_func=example_feature_extraction_function,
        cv_folds=10,
        random_state=42
    )

    # Train and evaluate with provided data
    results = trainer.train_and_evaluate(train_data_dict)

    # Show results
    trainer.print_detailed_results()
    trainer.plot_diagnostics()

    return trainer


def run_train_val_split_example():
    """Example showing train/validation split functionality"""

    trainer = TimeSeriesModelTrainer(
        feature_extraction_func=example_feature_extraction_function,
        cv_folds=5,
        random_state=42,
        use_crunch=False
    )

    # Get train/val split and evaluate
    train_results, val_results = trainer.get_train_val_split(
        val_ratio=0.2,
        data_ratio=0.5  # Use only 50% of data for faster testing
    )

    print(f"Train CV AUC: {train_results['cv_val_mean']:.4f}")
    print(f"Validation AUC: {val_results['val_auc']:.4f}")

    return trainer


if __name__ == "__main__":
    print("TimeSeriesModelTrainer class is ready for use!")
    print("\nUsage examples:")
    print("""
    # Method 1: Automatic data loading
    trainer = TimeSeriesModelTrainer(
        feature_extraction_func=your_feature_extraction_function,
        cv_folds=10,
        random_state=42,
        use_crunch=False  # Set based on your environment
    )
    results = trainer.train_and_evaluate()  # Auto-loads data

    # Method 2: Manual data loading
    data_loader = StructuralBreakDataLoader()
    data_loader.load_data(use_crunch=False)
    train_data_dict = data_loader.get_all_train_series()

    trainer = TimeSeriesModelTrainer(feature_extraction_func=your_function)
    results = trainer.train_and_evaluate(train_data_dict)

    # Method 3: Train/validation split
    train_results, val_results = trainer.get_train_val_split(val_ratio=0.2)

    # Show results and diagnostics
    trainer.print_detailed_results()
    trainer.plot_diagnostics()
    """)
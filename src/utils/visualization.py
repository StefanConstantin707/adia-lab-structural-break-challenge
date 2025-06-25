# src/utils/visualization.py
"""Visualization utilities for structural break analysis"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from sklearn.metrics import roc_auc_score


def plot_time_series_with_break(
        series_data: pd.DataFrame,
        series_id: int,
        has_break: Optional[bool] = None,
        prediction: Optional[float] = None,
        save_path: Optional[str] = None
):
    """
    Plot a time series with highlighted periods and break point.

    Args:
        series_data: DataFrame with 'value' and 'period' columns
        series_id: ID of the series
        has_break: True label if available
        prediction: Model prediction if available
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Get periods
    period_0_mask = series_data['period'] == 0
    period_1_mask = series_data['period'] == 1

    # Main time series plot
    time_points = range(len(series_data))
    values = series_data['value'].values

    # Plot period 0 in blue
    ax1.plot(
        np.where(period_0_mask)[0],
        values[period_0_mask],
        'b-',
        label='Period 0',
        alpha=0.7
    )

    # Plot period 1 in red
    ax1.plot(
        np.where(period_1_mask)[0],
        values[period_1_mask],
        'r-',
        label='Period 1',
        alpha=0.7
    )

    # Mark boundary point
    boundary_idx = np.where(period_1_mask)[0][0] if any(period_1_mask) else len(values)
    ax1.axvline(x=boundary_idx, color='green', linestyle='--', label='Boundary', alpha=0.5)

    # Add title with prediction info
    title = f"Series {series_id}"
    if has_break is not None:
        title += f" | True Label: {'Break' if has_break else 'No Break'}"
    if prediction is not None:
        title += f" | Prediction: {prediction:.3f}"

    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Period indicator
    periods = series_data['period'].values
    ax2.fill_between(time_points, 0, periods, alpha=0.5, step='post')
    ax2.set_ylabel('Period')
    ax2.set_xlabel('Time')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_feature_distributions(
        features_df: pd.DataFrame,
        labels: pd.Series,
        feature_names: List[str],
        save_path: Optional[str] = None
):
    """
    Plot distributions of features for break vs no-break cases.

    Args:
        features_df: DataFrame with features
        labels: Boolean series of break labels
        feature_names: List of features to plot
        save_path: Path to save the plot
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx]

        # Plot distributions
        feature_values = features_df[feature_name]

        # No break cases
        no_break_values = feature_values[~labels]
        ax.hist(no_break_values, bins=30, alpha=0.5, label='No Break', density=True, color='blue')

        # Break cases
        break_values = feature_values[labels]
        ax.hist(break_values, bins=30, alpha=0.5, label='Break', density=True, color='red')

        ax.set_xlabel(feature_name)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'Distribution of {feature_name}')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(feature_names), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_model_performance(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
):
    """
    Plot ROC curve and prediction distribution.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Prediction Distribution
    ax2.hist(y_pred[y_true == 0], bins=30, alpha=0.5, label='No Break', density=True, color='blue')
    ax2.hist(y_pred[y_true == 1], bins=30, alpha=0.5, label='Break', density=True, color='red')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Prediction Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

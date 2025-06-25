import os
import tempfile
import matplotlib
# Use non-interactive backend for tests
test_backend = matplotlib.get_backend()
matplotlib.use('Agg')
import pytest
import numpy as np
import pandas as pd
from src.utils.visualization import (
    plot_time_series_with_break,
    plot_feature_distributions,
    plot_model_performance
)

class TestVisualizations:

    def test_plot_time_series_with_break_no_break(self):
        # Create synthetic series with no break: all period 0
        np.random.seed(0)
        values_nb = np.random.normal(size=100)
        periods_nb = np.zeros(100)
        df_nb = pd.DataFrame({'value': values_nb, 'period': periods_nb})
        plot_time_series_with_break(df_nb, series_id=1, has_break=False, prediction=0.2)


    def test_plot_time_series_with_break_with_break(self):
        # Create synthetic series with break at midpoint
        np.random.seed(1)
        values_b = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(3, 1, 50)])
        periods_b = np.concatenate([np.zeros(50), np.ones(50)])
        df_b = pd.DataFrame({'value': values_b, 'period': periods_b})
        plot_time_series_with_break(df_b, series_id=2, has_break=True, prediction=0.8)


    def test_plot_feature_distributions(self):
        # Create synthetic features DataFrame with two features
        np.random.seed(2)
        n = 100
        feat1 = np.random.normal(0, 1, n)
        feat2 = np.random.normal(1, 2, n)
        labels_fd = pd.Series(np.array([0] * 50 + [1] * 50, dtype=bool))
        features_df = pd.DataFrame({'feat1': feat1, 'feat2': feat2})
        plot_feature_distributions(features_df, labels_fd, ['feat1', 'feat2'])

    def test_plot_model_performance(self):
        np.random.seed(3)
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.concatenate([np.random.uniform(0, 0.4, 50), np.random.uniform(0.6, 1.0, 50)])
        plot_model_performance(y_true, y_pred)

# Restore backend after tests
matplotlib.use(test_backend)

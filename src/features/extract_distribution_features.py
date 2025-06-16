# src/features/extract_distribution_features.py
from typing import Dict

import pandas as pd
import numpy as np
import scipy.stats as st

def extract_distribution_features(series_data: pd.DataFrame) -> Dict[str, float]:
    """
    Extract features from a time series for structural break detection.

    Args:
        series_data: DataFrame with 'value' and 'period' columns

    Returns:
        Dictionary of features
    """
    # Get period 0 and period 1 data
    period_0 = series_data[series_data['period'] == 0]['value'].values
    period_1 = series_data[series_data['period'] == 1]['value'].values

    features = {}

    # Basic statistics for each period
    if len(period_0) > 0:
        features['p0_mean'] = np.mean(period_0)
        features['p0_std'] = np.std(period_0)
        features['p0_skew'] = float(pd.Series(period_0).skew())
        features['p0_kurt'] = float(pd.Series(period_0).kurt())
    else:
        features['p0_mean'] = 0.0
        features['p0_std'] = 0.0
        features['p0_skew'] = 0.0
        features['p0_kurt'] = 0.0

    if len(period_1) > 0:
        features['p1_mean'] = np.mean(period_1)
        features['p1_std'] = np.std(period_1)
        features['p1_skew'] = float(pd.Series(period_1).skew())
        features['p1_kurt'] = float(pd.Series(period_1).kurt())
    else:
        features['p1_mean'] = 0.0
        features['p1_std'] = 0.0
        features['p1_skew'] = 0.0
        features['p1_kurt'] = 0.0

    # Differences between periods
    features['mean_diff'] = features['p1_mean'] - features['p0_mean']
    features['std_diff'] = features['p1_std'] - features['p0_std']
    features['mean_ratio'] = features['p1_mean'] / (features['p0_mean'] + 1e-8)
    features['std_ratio'] = features['p1_std'] / (features['p0_std'] + 1e-8)

    # Statistical tests (example - add more sophisticated tests)
    if len(period_0) > 1 and len(period_1) > 1:
        # T-test for mean difference
        t_stat, p_value = st.ttest_ind(period_0, period_1)
        features['ttest_stat'] = t_stat
        features['ttest_pvalue'] = p_value

        # Levene test for variance difference
        lev_stat, lev_pvalue = st.levene(period_0, period_1)
        features['levene_stat'] = lev_stat
        features['levene_pvalue'] = lev_pvalue

        # Mann-Whitney U test
        u_stat, u_pvalue = st.mannwhitneyu(period_0, period_1)
        features['mannwhitney_pvalue'] = u_pvalue

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = st.ks_2samp(period_0, period_1)
        features['ks_pvalue'] = ks_pvalue
    else:
        features['ttest_stat'] = 0.0
        features['ttest_pvalue'] = 1.0
        features['levene_stat'] = 0.0
        features['levene_pvalue'] = 1.0
        features['mannwhitney_pvalue'] = 1.0
        features['ks_pvalue'] = 1.0

    # Length features
    features['p0_length'] = len(period_0)
    features['p1_length'] = len(period_1)
    features['length_ratio'] = len(period_1) / (len(period_0) + 1)

    return features
# tests/test_features.py
"""Unit tests for feature extraction"""
import pytest
import numpy as np
import pandas as pd



class TestFeatureExtraction:

    def test_basic_features(self):
        """Test basic statistical feature extraction"""
        # Create synthetic time series
        time_index = pd.MultiIndex.from_product(
            [[0], range(100)],
            names=['id', 'time']
        )

        # No break case
        values = np.random.normal(0, 1, 100)
        periods = np.concatenate([np.zeros(50), np.ones(50)])

        df = pd.DataFrame({
            'value': values,
            'period': periods
        }, index=time_index)

        features = extract_statistical_features(df.loc[0])

        assert 'p0_mean' in features
        assert 'p1_mean' in features
        assert 'mean_diff' in features

    def test_with_structural_break(self):
        """Test features with obvious structural break"""
        time_index = pd.MultiIndex.from_product(
            [[0], range(100)],
            names=['id', 'time']
        )

        # Create break in mean
        values = np.concatenate([
            np.random.normal(0, 1, 50),  # Period 0: mean=0
            np.random.normal(3, 1, 50)  # Period 1: mean=3
        ])
        periods = np.concatenate([np.zeros(50), np.ones(50)])

        df = pd.DataFrame({
            'value': values,
            'period': periods
        }, index=time_index)

        features = extract_statistical_features(df.loc[0])

        # Mean difference should be approximately 3
        assert abs(features['mean_diff'] - 3) < 0.5

# src/features/extract_autocorrelation_features.py
from typing import Dict

import pandas as pd
import numpy as np
import scipy.stats as st
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf


def extract_autocorrelation_features(series_data: pd.DataFrame, max_lag=20) -> Dict[str, float]:
    """
    Extract autocorrelation-related features from a time series split into two periods.

    Args:
        series_data: DataFrame with columns 'value' and 'period' (0 or 1).
        max_lag: maximum lag to consider for ACF/PACF.

    Returns:
        Dictionary of features, including separate features for period 0 (p0_...) and period 1 (p1_...),
        plus differences (diff_...) and distances.
    """
    features: Dict[str, float] = {}

    # Split values
    vals0 = series_data.loc[series_data['period'] == 0, 'value'].dropna().values
    vals1 = series_data.loc[series_data['period'] == 1, 'value'].dropna().values

    def compute_segment_feats(vals: np.ndarray, prefix: str) -> Dict[str, float]:
        seg_feats: Dict[str, float] = {}
        n = len(vals)
        if n < 3:
            # Too short to compute meaningful ACF/PACF up to lag 1
            for lag in [1, 2, 5, 10]:
                seg_feats[f"{prefix}_acf_lag{lag}"] = 0.0
                seg_feats[f"{prefix}_pacf_lag{lag}"] = 0.0
            seg_feats[f"{prefix}_acf_sum_abs"] = 0.0
            seg_feats[f"{prefix}_acf_mean"] = 0.0
            seg_feats[f"{prefix}_acf_std"] = 0.0
            seg_feats[f"{prefix}_acf_first_zero"] = float(max_lag)
            seg_feats[f"{prefix}_num_sig_acf_lags"] = 0.0
            seg_feats[f"{prefix}_lb_stat_lag10"] = np.nan
            seg_feats[f"{prefix}_lb_pvalue_lag10"] = np.nan
            return seg_feats

        # Determine nlags based on length
        nlags = min(max_lag, n - 1)
        # Compute ACF and PACF
        # Use fft for acf if series is long; statsmodels handles shorter automatically
        try:
            acf_vals = acf(vals, nlags=nlags, fft=True, missing='drop')
        except Exception:
            # fallback without fft
            acf_vals = acf(vals, nlags=nlags, fft=False, missing='drop')
        # pacf: use default method
        try:
            pacf_vals = pacf(vals, nlags=nlags, method='ywunbiased')
        except Exception:
            pacf_vals = pacf(vals, nlags=nlags, method='ld')  # another fallback

        # Basic lag features for certain lags if available
        for lag in [1, 2, 5, 10]:
            if lag <= nlags:
                seg_feats[f"{prefix}_acf_lag{lag}"] = float(acf_vals[lag])
                seg_feats[f"{prefix}_pacf_lag{lag}"] = float(pacf_vals[lag])
            else:
                seg_feats[f"{prefix}_acf_lag{lag}"] = 0.0
                seg_feats[f"{prefix}_pacf_lag{lag}"] = 0.0

        # Summary stats over ACF lags 1..nlags
        acf_slice = acf_vals[1 : nlags+1]
        seg_feats[f"{prefix}_acf_sum_abs"] = float(np.sum(np.abs(acf_slice)))
        seg_feats[f"{prefix}_acf_mean"] = float(np.mean(acf_slice))
        seg_feats[f"{prefix}_acf_std"] = float(np.std(acf_slice))

        # First zero-crossing lag (where sign changes) in ACF beyond lag 0
        zero_crossings = np.where(np.diff(np.sign(acf_slice)) != 0)[0]
        if len(zero_crossings) > 0:
            # +1 because acf_slice[0] is lag1, so index 0 → lag1
            seg_feats[f"{prefix}_acf_first_zero"] = float(zero_crossings[0] + 1)
        else:
            seg_feats[f"{prefix}_acf_first_zero"] = float(nlags)

        # Count of “significant” autocorrelations: |acf| > approx 1.96/sqrt(N)
        conf_bound = 1.96 / np.sqrt(n)
        num_sig = np.sum(np.abs(acf_slice) > conf_bound)
        seg_feats[f"{prefix}_num_sig_acf_lags"] = float(num_sig)

        # Ljung-Box test at lag 10 if n > 10
        lb_lag = 10
        if n > lb_lag:
            try:
                # return_df=True gives DataFrame with columns lb_stat, lb_pvalue
                lb_res = acorr_ljungbox(vals, lags=[lb_lag], return_df=True)
                seg_feats[f"{prefix}_lb_stat_lag10"] = float(lb_res['lb_stat'].iloc[0])
                seg_feats[f"{prefix}_lb_pvalue_lag10"] = float(lb_res['lb_pvalue'].iloc[0])
            except Exception:
                seg_feats[f"{prefix}_lb_stat_lag10"] = np.nan
                seg_feats[f"{prefix}_lb_pvalue_lag10"] = np.nan
        else:
            seg_feats[f"{prefix}_lb_stat_lag10"] = np.nan
            seg_feats[f"{prefix}_lb_pvalue_lag10"] = np.nan

        # Optionally, include full-length ACF/PACF vectors for distance computations later
        # We'll return them under special keys:
        seg_feats[f"{prefix}_acf_full"] = acf_vals  # numpy array
        seg_feats[f"{prefix}_pacf_full"] = pacf_vals  # numpy array

        return seg_feats

    # Compute features for each period
    feats0 = compute_segment_feats(vals0, prefix="p0")
    feats1 = compute_segment_feats(vals1, prefix="p1")

    # Transfer basic scalar features
    for k, v in feats0.items():
        if isinstance(v, np.ndarray):
            continue
        features[k] = v
    for k, v in feats1.items():
        if isinstance(v, np.ndarray):
            continue
        features[k] = v

    # Distance / difference features between ACF/PACF vectors
    acf0 = feats0.get("p0_acf_full", None)
    acf1 = feats1.get("p1_acf_full", None)
    if (acf0 is not None) and (acf1 is not None):
        # align lengths for distance: pad shorter with zeros to match longer
        len0, len1 = len(acf0), len(acf1)
        L = max(len0, len1)
        a0 = np.pad(acf0, (0, L - len0), 'constant')
        a1 = np.pad(acf1, (0, L - len1), 'constant')
        # Euclidean distance
        features['diff_acf_euclid'] = float(np.linalg.norm(a1 - a0))
        # L1 distance
        features['diff_acf_l1'] = float(np.sum(np.abs(a1 - a0)))
        # Difference in sum_abs
        # Already captured via p1_acf_sum_abs - p0_acf_sum_abs if desired:
        features['diff_acf_sum_abs'] = features.get('p1_acf_sum_abs', 0.0) - features.get('p0_acf_sum_abs', 0.0)
    else:
        features['diff_acf_euclid'] = 0.0
        features['diff_acf_l1'] = 0.0
        features['diff_acf_sum_abs'] = 0.0

    pacf0 = feats0.get("p0_pacf_full", None)
    pacf1 = feats1.get("p1_pacf_full", None)
    if (pacf0 is not None) and (pacf1 is not None):
        len0, len1 = len(pacf0), len(pacf1)
        L = max(len0, len1)
        p0 = np.pad(pacf0, (0, L - len0), 'constant')
        p1 = np.pad(pacf1, (0, L - len1), 'constant')
        features['diff_pacf_euclid'] = float(np.linalg.norm(p1 - p0))
        features['diff_pacf_l1'] = float(np.sum(np.abs(p1 - p0)))
        features['diff_pacf_sum_abs'] = float(np.sum(np.abs(p1[1:]) - np.abs(p0[1:]))) if L > 1 else 0.0
    else:
        features['diff_pacf_euclid'] = 0.0
        features['diff_pacf_l1'] = 0.0
        features['diff_pacf_sum_abs'] = 0.0

    # Difference in first zero-crossing lag
    fz0 = features.get('p0_acf_first_zero', max_lag)
    fz1 = features.get('p1_acf_first_zero', max_lag)
    features['diff_acf_first_zero'] = float(fz1 - fz0)

    # Difference in number of significant lags
    ns0 = features.get('p0_num_sig_acf_lags', 0.0)
    ns1 = features.get('p1_num_sig_acf_lags', 0.0)
    features['diff_num_sig_acf_lags'] = float(ns1 - ns0)

    # Difference in Ljung-Box stats/pvalues
    lb0_stat = features.get('p0_lb_stat_lag10', np.nan)
    lb1_stat = features.get('p1_lb_stat_lag10', np.nan)
    lb0_p = features.get('p0_lb_pvalue_lag10', np.nan)
    lb1_p = features.get('p1_lb_pvalue_lag10', np.nan)
    # If both are numeric:
    try:
        features['diff_lb_stat_lag10'] = float(lb1_stat - lb0_stat)
    except Exception:
        features['diff_lb_stat_lag10'] = np.nan
    try:
        features['diff_lb_pvalue_lag10'] = float(lb1_p - lb0_p)
    except Exception:
        features['diff_lb_pvalue_lag10'] = np.nan

    # Clean up full arrays from final output (we keep only scalar features)
    # (They were only used above for distance computations)
    # No action needed since we never copied array-valued keys into `features`.

    return features

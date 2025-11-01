# utils/bias_metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# --------------------------------------------------------------------
# 1️⃣ Utility Helpers
# --------------------------------------------------------------------
def safe_unique_nonan(series):
    """Return unique non-null values safely."""
    return pd.Series(series).dropna().unique()


def safe_mean(series):
    """Compute mean safely with NaN handling."""
    try:
        return float(np.nanmean(series))
    except Exception:
        return np.nan


# --------------------------------------------------------------------
# 2️⃣ Data-Level Bias Metrics
# --------------------------------------------------------------------
def calc_spd_binary(df, feature, target):
    """
    Statistical Parity Difference (SPD)
    Difference in positive rates between two groups.
    Ideal ≈ 0
    """
    try:
        vals = safe_unique_nonan(df[feature])
        if target is None or len(vals) != 2:
            return np.nan

        a, b = vals[0], vals[1]
        p_a = safe_mean(df.loc[df[feature] == a, target])
        p_b = safe_mean(df.loc[df[feature] == b, target])

        return round(float(p_b - p_a), 4)
    except Exception as e:
        logging.warning(f"SPD calc failed for {feature}: {e}")
        return np.nan


def calc_di_binary(df, feature, target):
    """
    Disparate Impact (DI)
    Ratio of positive rates across two groups.
    Ideal between 0.8 – 1.25
    """
    try:
        vals = safe_unique_nonan(df[feature])
        if target is None or len(vals) != 2:
            return np.nan

        a, b = vals[0], vals[1]
        p_a = safe_mean(df.loc[df[feature] == a, target])
        p_b = safe_mean(df.loc[df[feature] == b, target])

        if p_a == 0 or np.isnan(p_a):
            return np.nan

        return round(float(p_b / p_a), 4)
    except Exception as e:
        logging.warning(f"DI calc failed for {feature}: {e}")
        return np.nan


def calc_mutual_info(df, feature, target):
    """
    Mutual Information (MI)
    Measures dependency between feature and target.
    Higher → stronger relationship → potential bias.
    """
    try:
        return round(
            float(
                mutual_info_score(
                    df[feature].astype(str).fillna("Unknown"),
                    df[target].astype(str).fillna("Unknown")
                )
            ),
            4,
        )
    except Exception as e:
        logging.warning(f"MI calc failed for {feature}: {e}")
        return np.nan


# --------------------------------------------------------------------
# 3️⃣ Model-Level Fairness Metrics
# --------------------------------------------------------------------
def demographic_parity(y_pred, sens):
    """
    Difference in positive prediction rates between sensitive groups.
    """
    try:
        groups = np.unique(sens)
        rates = [
            safe_mean(y_pred[sens == g])
            for g in groups if np.sum(sens == g) > 0
        ]
        return round(float(np.max(rates) - np.min(rates)), 4)
    except Exception as e:
        logging.error(f"Demographic Parity failed: {e}")
        return np.nan


def equal_opportunity(y_true, y_pred, sens, pos_label=1):
    """
    Difference in True Positive Rates (TPR) between groups.
    """
    try:
        groups = np.unique(sens)
        tprs = []
        for g in groups:
            mask = (sens == g) & (y_true == pos_label)
            if mask.sum() > 0:
                tprs.append(safe_mean(y_pred[mask]))
        return round(float(np.max(tprs) - np.min(tprs)), 4) if len(tprs) > 1 else np.nan
    except Exception as e:
        logging.error(f"Equal Opportunity failed: {e}")
        return np.nan


def predictive_parity(y_true, y_pred, sens, pos_label=1):
    """
    Difference in precision across sensitive groups.
    """
    try:
        groups = np.unique(sens)
        precs = []
        for g in groups:
            mask = (sens == g) & (y_pred == pos_label)
            if mask.sum() > 0:
                precs.append(safe_mean(y_true[mask]))
        return round(float(np.max(precs) - np.min(precs)), 4) if len(precs) > 1 else np.nan
    except Exception as e:
        logging.error(f"Predictive Parity failed: {e}")
        return np.nan

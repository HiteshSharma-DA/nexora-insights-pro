"""Descriptive statistics and careful inference."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include="number").columns:
        v = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(v):
            continue
        rows.append({
            "column": col, "count": len(v), "mean": np.mean(v),
            "median": np.median(v), "std": np.std(v, ddof=1) if len(v) > 1 else np.nan,
            "min": np.min(v), "p25": np.percentile(v, 25), "p75": np.percentile(v, 75),
            "max": np.max(v), "skew": stats.skew(v) if len(v) > 2 and np.ptp(v) else np.nan,
        })
    return pd.DataFrame(rows)


def welch_test(df: pd.DataFrame, group_col: str, num_col: str, first: str, second: str) -> dict:
    s = df[group_col].astype(str)
    a = pd.to_numeric(df.loc[s == str(first), num_col], errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(df.loc[s == str(second), num_col], errors="coerce").dropna().to_numpy(dtype=float)
    if min(len(a), len(b)) < 2:
        raise ValueError("Both groups require at least two numeric observations.")
    test = stats.ttest_ind(a, b, equal_var=False)
    mean_diff = float(a.mean() - b.mean())
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    standard_error = float(np.sqrt(var_a / len(a) + var_b / len(b)))
    dof = (var_a / len(a) + var_b / len(b)) ** 2 / (
        (var_a / len(a)) ** 2 / (len(a) - 1) + (var_b / len(b)) ** 2 / (len(b) - 1)
    ) if standard_error else np.nan
    critical = stats.t.ppf(.975, dof) if np.isfinite(dof) else np.nan
    pooled = np.sqrt(((len(a)-1)*var_a + (len(b)-1)*var_b) / (len(a)+len(b)-2))
    return {
        "n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_difference": mean_diff, "t_statistic": float(test.statistic), "p_value": float(test.pvalue),
        "cohens_d": mean_diff / pooled if pooled else np.nan,
        "ci_low": mean_diff - critical * standard_error,
        "ci_high": mean_diff + critical * standard_error,
    }


def chi_square_test(df: pd.DataFrame, first: str, second: str) -> tuple[dict, pd.DataFrame]:
    table = pd.crosstab(df[first], df[second])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both categorical variables need at least two nonempty levels.")
    result = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(result.statistic / (n * min(table.shape[0]-1, table.shape[1]-1))) if n else np.nan
    return {"chi2": float(result.statistic), "p_value": float(result.pvalue), "dof": int(result.dof), "cramers_v": float(cramers_v), "low_expected": int((result.expected_freq < 5).sum())}, table


def outlier_profile(df: pd.DataFrame, num_col: str) -> tuple[pd.DataFrame, float, float]:
    values = pd.to_numeric(df[num_col], errors="coerce")
    q1, q3 = values.quantile([.25, .75])
    iqr = q3 - q1
    lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    result = df[(values < lower) | (values > upper)]
    return result, lower, upper

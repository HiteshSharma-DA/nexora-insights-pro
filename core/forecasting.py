"""Coverage-aware time series, transparent baseline selection and honest backtests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def aggregate_series(df: pd.DataFrame, date: str, measure: str,
                     freq: str = "W", complete_only: bool = False) -> pd.DataFrame:
    data = df[[date, measure]].dropna().copy()
    if data.empty:
        return pd.DataFrame(columns=["period", "value"])
    data[date] = pd.to_datetime(data[date], errors="coerce")
    data = data.dropna(subset=[date])
    if data.empty:
        return pd.DataFrame(columns=["period", "value"])
    values = data.set_index(date)[measure].resample(freq).sum(min_count=1).dropna()
    if complete_only and len(values):
        observed_start = data[date].min().normalize()
        observed_end = data[date].max().normalize()
        # Coverage is assessed at the time-bucket boundaries, not by the number
        # of transactions. This conservatively drops partial first/last buckets.
        if freq in ("MS", "M", "ME"):
            first_start = values.index[0].to_period("M").start_time.normalize()
            last_end = values.index[-1].to_period("M").end_time.normalize()
        else:
            first_start = (values.index[0] - pd.Timedelta(days=6)).normalize()
            last_end = values.index[-1].normalize()
        if observed_start > first_start:
            values = values.iloc[1:]
        if len(values) and observed_end < last_end:
            values = values.iloc[:-1]
    return values.rename_axis("period").reset_index(name="value")


def forecast_linear_trend(series: pd.DataFrame, horizon: int = 4) -> tuple[pd.DataFrame, dict]:
    vals = series["value"].to_numpy(dtype=float)
    if len(vals) < 6 or not np.isfinite(vals).all():
        raise ValueError("At least 6 complete periods with finite values are needed for a trend forecast.")
    x = np.arange(len(vals), dtype=float)
    result = stats.linregress(x, vals)
    future_x = np.arange(len(vals), len(vals) + horizon, dtype=float)
    predicted = result.intercept + result.slope * future_x
    residual = vals - (result.intercept + result.slope * x)
    stderr = float(np.sqrt(np.sum(residual ** 2) / max(1, len(vals) - 2)))
    sxx = np.sum((x - x.mean()) ** 2)
    tval = float(stats.t.ppf(0.975, len(vals) - 2))
    band = tval * stderr * np.sqrt(1 + 1 / len(vals) + (future_x - x.mean()) ** 2 / sxx)
    output = pd.DataFrame({"step": range(1, horizon+1), "prediction": predicted,
                           "lower": predicted-band, "upper": predicted+band})
    return output, {"slope_per_period": float(result.slope), "r_squared": float(result.rvalue**2),
                    "periods": len(vals), "method": "Linear trend; approximate 95% prediction interval; no seasonality"}


def _one_step(y: np.ndarray, method: str, season: int) -> float:
    if method == "Last-period baseline":
        return float(y[-1])
    if method == "Seasonal naive":
        return float(y[-season])
    t = np.arange(len(y), dtype=float)
    fit = stats.linregress(t, y)
    return float(fit.intercept + fit.slope * len(y))


def forecast_with_backtest(series: pd.DataFrame, horizon: int = 4,
                           freq: str = "MS") -> tuple[pd.DataFrame, dict]:
    """Rolling one-step backtest of two/three interpretable methods; no trained ML required."""
    y = series["value"].to_numpy(dtype=float)
    n = len(y)
    if n < 7 or not np.isfinite(y).all():
        raise ValueError("Need at least 7 complete periods to backtest and forecast.")
    season = 12 if freq in ("MS", "M", "ME") else 52
    methods = ["Last-period baseline", "Linear trend"]
    if n >= 2 * season + 4:
        methods.append("Seasonal naive")
    # Up to 5 held-out one-step periods, with >=6 observations for fitting.
    holdout = min(5, n - 6)
    actual, preds = y[-holdout:], {}
    for method in methods:
        arr = []
        for t in range(n-holdout, n):
            if method == "Seasonal naive" and t < season:
                break
            arr.append(_one_step(y[:t], method, season))
        if len(arr) == holdout:
            preds[method] = np.asarray(arr, dtype=float)
    if not preds:
        raise ValueError("Not enough complete periods for a meaningful backtest.")
    errors = {name: actual - prediction for name, prediction in preds.items()}
    mae = {name: float(np.mean(np.abs(residual))) for name, residual in errors.items()}
    selected = min(mae, key=lambda name: (mae[name], methods.index(name)))
    future = []
    for k in range(1, horizon + 1):
        if selected == "Linear trend":
            future.append(float(_one_step(y, "Linear trend", season) +
                                (k-1) * stats.linregress(np.arange(n), y).slope))
        elif selected == "Seasonal naive":
            future.append(float(y[n-season + ((k-1) % season)]))
        else:
            future.append(float(y[-1]))
    future = np.array(future)
    # Uncalibrated empirical uncertainty: don't claim formal 95% coverage.
    residuals = errors[selected]
    scale = float(np.sqrt(np.mean(residuals**2)))
    interval = 2 * scale * np.sqrt(np.arange(1, horizon+1))
    if (y >= 0).all():
        future = np.maximum(future, 0)
    lower = future - interval
    if (y >= 0).all():
        lower = np.maximum(lower, 0)
    result = pd.DataFrame({"step": np.arange(1, horizon+1), "prediction": future,
                           "lower": lower, "upper": future+interval})
    denom = (np.abs(actual) + np.abs(preds[selected])) / 2
    smape = float(np.mean(np.divide(np.abs(residuals), denom, out=np.zeros_like(denom), where=denom != 0)) * 100)
    meta = {"method": selected, "periods": n, "backtest_periods": holdout,
            "mae": mae[selected], "rmse": scale, "smape_pct": smape,
            "candidate_mae": mae, "interval_note": "Indicative uncertainty band (not a calibrated 95% confidence interval)."}
    return result, meta


def flag_anomalies(series: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    if series.empty:
        return series.assign(robust_z=[], anomaly=[])
    result = series.copy()
    values = result["value"].to_numpy(dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values-median)))
    if mad < 1e-12:
        result["robust_z"] = 0.0
        result["anomaly"] = False
    else:
        result["robust_z"] = 0.6745 * (values-median)/mad
        result["anomaly"] = result["robust_z"].abs() > threshold
    return result

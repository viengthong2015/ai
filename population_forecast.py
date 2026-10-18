"""Population forecasting functions shared by the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


HISTORICAL_YEARS = list(range(2016, 2026))
HISTORICAL_COLUMNS = [str(year) for year in HISTORICAL_YEARS]
MODEL_NAMES = ("Linear", "Polynomial", "Exponential")


@dataclass(frozen=True)
class ForecastResult:
    actual: np.ndarray
    all_years: np.ndarray
    predictions: dict[str, np.ndarray]
    fitted: dict[str, np.ndarray]
    future: dict[str, np.ndarray]
    metrics: pd.DataFrame


def annual_growth_rates(values: np.ndarray) -> np.ndarray:
    """Return year-over-year percentage changes, excluding the first value."""
    population = np.asarray(values, dtype=float)
    if population.ndim != 1:
        raise ValueError("Population values must be a one-dimensional series.")
    if population.size < 2:
        return np.asarray([], dtype=float)

    previous = population[:-1]
    changes = population[1:] - previous
    return np.divide(
        changes,
        previous,
        out=np.full(changes.shape, np.nan, dtype=float),
        where=previous != 0,
    ) * 100


def validate_population_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the ten historical population columns."""
    missing = [column for column in HISTORICAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required year columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("The data file contains no rows.")

    clean = df.copy()
    for column in HISTORICAL_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    invalid = clean[HISTORICAL_COLUMNS].isna().sum().sum()
    if invalid:
        raise ValueError(
            f"The historical year columns contain {invalid:,} blank or non-numeric values."
        )
    if not np.isfinite(clean[HISTORICAL_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("The historical year columns contain infinite values.")
    return clean


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE (%)": mape,
        "R²": float(r2_score(y_true, y_pred)),
        "Accuracy (%)": 100 - mape,
    }


def forecast_totals(df: pd.DataFrame, future_end: int = 2030) -> ForecastResult:
    """Fit the notebook's three models to population totals."""
    if future_end < 2026:
        raise ValueError("The forecast end year must be 2026 or later.")

    historical_x = np.asarray(HISTORICAL_YEARS).reshape(-1, 1)
    all_years = np.arange(HISTORICAL_YEARS[0], future_end + 1)
    all_x = all_years.reshape(-1, 1)
    y = df[HISTORICAL_COLUMNS].sum(axis=0).to_numpy(dtype=float)

    linear = LinearRegression().fit(historical_x, y)

    polynomial_features = PolynomialFeatures(degree=2)
    historical_poly = polynomial_features.fit_transform(historical_x)
    polynomial = LinearRegression().fit(historical_poly, y)

    # This matches the notebook, while preventing log(0) for an empty subgroup.
    exponential = LinearRegression().fit(historical_x, np.log(np.where(y <= 0, 0.1, y)))

    predictions = {
        "Linear": linear.predict(all_x),
        "Polynomial": polynomial.predict(polynomial_features.transform(all_x)),
        "Exponential": np.exp(exponential.predict(all_x)),
    }
    train_size = len(HISTORICAL_YEARS)
    fitted = {name: values[:train_size] for name, values in predictions.items()}
    future = {name: values[train_size:] for name, values in predictions.items()}
    metrics = pd.DataFrame(
        [_metrics(y, fitted[name]) for name in MODEL_NAMES], index=MODEL_NAMES
    )
    metrics.index.name = "Model"

    return ForecastResult(y, all_years, predictions, fitted, future, metrics)


def forecast_each_row(df: pd.DataFrame, future_end: int = 2030) -> pd.DataFrame:
    """Create row-level predictions for all models using vectorized multi-output fits."""
    future_years = np.arange(2026, future_end + 1)
    historical_x = np.asarray(HISTORICAL_YEARS).reshape(-1, 1)
    future_x = future_years.reshape(-1, 1)
    values = df[HISTORICAL_COLUMNS].to_numpy(dtype=float).T

    linear = LinearRegression().fit(historical_x, values)
    linear_predictions = linear.predict(future_x).T

    polynomial_features = PolynomialFeatures(degree=2)
    historical_poly = polynomial_features.fit_transform(historical_x)
    polynomial = LinearRegression().fit(historical_poly, values)
    polynomial_predictions = polynomial.predict(
        polynomial_features.transform(future_x)
    ).T

    safe_values = np.where(values <= 0, 0.1, values)
    exponential = LinearRegression().fit(historical_x, np.log(safe_values))
    exponential_predictions = np.exp(exponential.predict(future_x)).T

    result = df.reset_index(drop=True).copy()
    model_predictions = {
        "Linear": linear_predictions,
        "Poly": polynomial_predictions,
        "Exp": exponential_predictions,
    }
    for suffix, predictions in model_predictions.items():
        rounded = np.rint(predictions).clip(min=0).astype(np.int64)
        for position, year in enumerate(future_years):
            result[f"{year}_{suffix}"] = rounded[:, position]
    return result


def polynomial_html_table(predicted: pd.DataFrame, future_end: int = 2030) -> pd.DataFrame:
    """Reproduce the notebook's HTML table (Polynomial forecast plus totals)."""
    excluded = [
        column
        for column in predicted.columns
        if column.endswith("_Linear") or column.endswith("_Exp")
    ]
    table = predicted.drop(columns=excluded).copy()
    total_columns = HISTORICAL_COLUMNS + [
        f"{year}_Poly" for year in range(2026, future_end + 1)
    ]
    table["Total"] = table[total_columns].sum(axis=1)

    grand_total = {column: "" for column in table.columns}
    grand_total[table.columns[0]] = "Grand Total"
    for column in total_columns + ["Total"]:
        grand_total[column] = table[column].sum()
    return pd.concat([table, pd.DataFrame([grand_total])], ignore_index=True)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_html_bytes(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_html(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

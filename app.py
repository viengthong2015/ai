"""Interactive Streamlit implementation of estimatePop.ipynb."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
import streamlit as st

from outlier_detection import OutlierResults, detect_population_outliers
from population_forecast import (
    HISTORICAL_COLUMNS,
    HISTORICAL_YEARS,
    MODEL_NAMES,
    dataframe_to_csv_bytes,
    dataframe_to_html_bytes,
    forecast_each_row,
    forecast_totals,
    polynomial_html_table,
    validate_population_data,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "data" / "dataDistrict.csv"
MODEL_STYLES = {
    "Linear": {"color": "#e63946", "linestyle": "--"},
    "Polynomial": {"color": "#2a9d5b", "linestyle": "-."},
    "Exponential": {"color": "#277da1", "linestyle": ":"},
}
MODEL_LABELS = {
    "Linear": "Linear Regression",
    "Polynomial": "Polynomial Regression",
    "Exponential": "Exponential Growth",
}
MODEL_EXPORT_SUFFIXES = {
    "Linear": "_Linear",
    "Polynomial": "_Poly",
    "Exponential": "_Exp",
}
ALL_MODELS_OPTION = "All models"

st.set_page_config(
    page_title="Laos Population Forecast",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_default_data(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time  # Included only to invalidate the cache when the file changes.
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_data(content: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(content))


@st.cache_data(show_spinner="Calculating row-level forecasts…")
def cached_row_forecast(df: pd.DataFrame, future_end: int) -> pd.DataFrame:
    return forecast_each_row(df, future_end)


@st.cache_data(show_spinner="Running nationwide outlier checks…")
def cached_outlier_checks(df: pd.DataFrame, threshold: float) -> OutlierResults:
    return detect_population_outliers(df, HISTORICAL_COLUMNS, threshold)


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df
    filter_specs = (
        ("ouName", "Districts"),
        ("sexName", "Sex"),
        ("cocName", "Age groups"),
    )
    for column, label in filter_specs:
        if column not in df.columns:
            continue
        options = sorted(df[column].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(
            label,
            options,
            placeholder=f"All {label.lower()}",
        )
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]
    return filtered


def forecast_figure(result, selected_models: list[str]) -> plt.Figure:
    fig, (forecast_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]}
    )
    forecast_axis.plot(
        HISTORICAL_YEARS,
        result.actual,
        "ko-",
        label="Actual population",
        linewidth=2,
        markersize=6,
    )
    for name in selected_models:
        forecast_axis.plot(
            result.all_years,
            result.predictions[name],
            label=MODEL_LABELS[name],
            linewidth=2,
            **MODEL_STYLES[name],
        )
    forecast_axis.axvline(2025, color="gray", linestyle="--", alpha=0.7)
    forecast_axis.axvspan(2025, result.all_years[-1], color="#e9ecef", alpha=0.45)
    forecast_axis.set_title("Population forecast")
    forecast_axis.set_ylabel("Total population")
    forecast_axis.ticklabel_format(axis="y", style="plain")
    forecast_axis.legend(ncol=2)
    forecast_axis.grid(alpha=0.25)

    for model_position, name in enumerate(selected_models):
        residuals = result.actual - result.fitted[name]
        residual_axis.plot(
            HISTORICAL_YEARS,
            residuals,
            marker="o",
            label=MODEL_LABELS[name],
            **MODEL_STYLES[name],
        )
        series_offset = (model_position - (len(selected_models) - 1) / 2) * 9
        for year, residual in zip(HISTORICAL_YEARS, residuals):
            rounded_residual = int(np.rint(residual))
            value_label = (
                "0"
                if rounded_residual == 0
                else f"{rounded_residual:+,}".replace("-", "−")
            )
            point_offset = (8 if residual >= 0 else -12) + series_offset
            residual_axis.annotate(
                value_label,
                xy=(year, residual),
                xytext=(0, point_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if point_offset >= 0 else "top",
                fontsize=8 if len(selected_models) == 1 else 7,
                color=MODEL_STYLES[name]["color"],
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )
    residual_axis.axhline(0, color="black", linewidth=1)
    residual_axis.set_title("Yearly residuals (actual − predicted)")
    residual_axis.set_xlabel("Year")
    residual_axis.set_ylabel("People")
    residual_axis.legend(ncol=3)
    residual_axis.grid(alpha=0.25)
    residual_axis.margins(y=0.22)
    fig.tight_layout()
    return fig


def actual_vs_predicted_figure(result, selected_models: list[str]) -> plt.Figure:
    """Compare historical observations with each selected model's fitted values."""
    panel_count = len(selected_models)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(13, 4.8 if panel_count == 1 else 3.8 * panel_count),
        squeeze=False,
        sharex=True,
    )
    x_positions = np.arange(len(HISTORICAL_YEARS))
    column_width = 0.38
    actual_values = np.asarray(result.actual, dtype=float)

    for axis, name in zip(axes.flat, selected_models):
        predicted_values = np.asarray(result.fitted[name], dtype=float)
        actual_bars = axis.bar(
            x_positions - column_width / 2,
            actual_values,
            width=column_width,
            color="#495057",
            edgecolor="white",
            linewidth=0.8,
            label="Actual",
            zorder=3,
        )
        predicted_bars = axis.bar(
            x_positions + column_width / 2,
            predicted_values,
            width=column_width,
            color=MODEL_STYLES[name]["color"],
            edgecolor="white",
            linewidth=0.8,
            label="Predicted",
            zorder=3,
        )

        percentage_differences = np.divide(
            predicted_values - actual_values,
            actual_values,
            out=np.full_like(predicted_values, np.nan),
            where=actual_values != 0,
        ) * 100
        for bar, difference in zip(predicted_bars, percentage_differences):
            if np.isnan(difference):
                label = "N/A"
                label_color = "#6c757d"
            else:
                rounded_difference = round(float(difference), 4)
                if rounded_difference == 0:
                    label = "0.0000%"
                    label_color = "#6c757d"
                else:
                    label = f"{rounded_difference:+.4f}%".replace("-", "−")
                    label_color = "#198754" if rounded_difference > 0 else "#dc3545"
            axis.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="semibold",
                color=label_color,
                rotation=45 if panel_count > 1 else 0,
            )

        axis.set_title(MODEL_LABELS[name], loc="left", fontsize=12, fontweight="bold")
        axis.set_ylabel("Population")
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.set_facecolor("#f8f9fa")
        axis.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", labelbottom=True)
        axis.margins(y=0.18)
        axis.legend(
            handles=[actual_bars, predicted_bars],
            labels=["Actual", "Predicted"],
            frameon=False,
            ncol=2,
            loc="upper left",
        )

    axes.flat[-1].set_xlabel("Year")
    for axis in axes.flat:
        axis.set_xticks(x_positions, HISTORICAL_YEARS)
    fig.suptitle(
        "Actual vs predicted population (historical fit)",
        fontsize=15,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def filter_outlier_results(results: pd.DataFrame, visible_data: pd.DataFrame) -> pd.DataFrame:
    """Apply display filters without changing nationwide reference thresholds."""
    visible = results
    mappings = (
        ("District", "ouName"),
        ("Age group", "cocName"),
    )
    for result_column, source_column in mappings:
        if result_column in visible and source_column in visible_data:
            allowed = set(visible_data[source_column].dropna().astype(str))
            visible = visible[visible[result_column].astype(str).isin(allowed)]
    if "Sex" in visible and "sexName" in visible_data:
        allowed_sexes = set(visible_data["sexName"].dropna().astype(str))
        visible = visible[
            visible["Sex"].astype(str).isin(allowed_sexes)
            | visible["Sex"].eq("Male/Female")
        ]
    return visible.reset_index(drop=True)


def show_outlier_check(
    title: str,
    description: str,
    results: pd.DataFrame,
    visible_data: pd.DataFrame,
    years: list[int],
) -> None:
    st.markdown(f"#### {title}")
    st.caption(description)
    visible = filter_outlier_results(results, visible_data)
    visible = visible[visible["Year"].isin(years)].reset_index(drop=True)
    st.metric("Flagged observations", f"{len(visible):,}")
    if visible.empty:
        st.success("No observations exceed the selected robust-score threshold.")
        return
    st.bar_chart(visible.groupby("Year").size().rename("Flags"), height=220)
    st.dataframe(
        visible.style.format(
            {
                "Population": "{:,.0f}",
                "Observed": "{:,.3f}",
                "Expected": "{:,.3f}",
                "Robust score": "{:+.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )
    st.download_button(
        f"Download {title.lower()} flags",
        visible.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{title.lower().replace(' ', '_')}_flags.csv",
        mime="text/csv",
        key=f"download-{title}",
    )


st.title("Laos Population Forecast")
st.caption(
    "Interactive implementation of `estimatePop.ipynb`: compare three models, "
    "inspect errors, and export district-level forecasts."
)

st.sidebar.header("Data and forecast")
uploaded_file = st.sidebar.file_uploader("Upload population CSV", type=["csv"])
future_end = st.sidebar.slider("Forecast through", 2026, 2040, 2030)
selected_model_option = st.sidebar.selectbox(
    "Forecast model",
    [ALL_MODELS_OPTION, *MODEL_NAMES],
    format_func=lambda name: MODEL_LABELS.get(name, name),
    help="Choose one model to inspect, or All models to compare their results and accuracy.",
)
selected_models = (
    list(MODEL_NAMES)
    if selected_model_option == ALL_MODELS_OPTION
    else [selected_model_option]
)

try:
    if uploaded_file is None:
        if not DEFAULT_DATA.exists():
            st.error(f"Default data file not found: {DEFAULT_DATA}")
            st.stop()
        raw_data = load_default_data(str(DEFAULT_DATA), DEFAULT_DATA.stat().st_mtime)
        source_name = DEFAULT_DATA.name
    else:
        raw_data = load_uploaded_data(uploaded_file.getvalue())
        source_name = uploaded_file.name
    data = validate_population_data(raw_data)
except Exception as error:
    st.error(f"Could not load the population data: {error}")
    st.stop()

filtered_data = filter_data(data)
if filtered_data.empty:
    st.warning("The selected filters contain no records. Change or clear the filters.")
    st.stop()

result = forecast_totals(filtered_data, future_end)
future_years = list(range(2026, future_end + 1))

summary_column_count = 3 if selected_model_option == ALL_MODELS_OPTION else 4
summary_columns = st.columns(summary_column_count)
summary_columns[0].metric("Rows", f"{len(filtered_data):,}")
summary_columns[1].metric("Population in 2016", f"{result.actual[0]:,.0f}")
summary_columns[2].metric("Population in 2025", f"{result.actual[-1]:,.0f}")
if selected_model_option == ALL_MODELS_OPTION:
    best_model = result.metrics.loc[selected_models, "MAPE (%)"].idxmin()
    st.subheader("All-model accuracy summary")
    all_model_summary = result.metrics.loc[list(MODEL_NAMES)].reset_index()
    all_model_summary["Model"] = all_model_summary["Model"].map(MODEL_LABELS)
    st.dataframe(
        all_model_summary.style.format(
            {
                "MAE": "{:,.2f}",
                "RMSE": "{:,.2f}",
                "MAPE (%)": "{:.4f}",
                "R²": "{:.4f}",
                "Accuracy (%)": "{:.4f}",
            }
        ).highlight_min(subset=["MAPE (%)"], color="#d1e7dd"),
        hide_index=True,
        width="stretch",
        height=143,
    )
    st.caption(f"Best fit by MAPE: {MODEL_LABELS[best_model]}")
else:
    selected_mape = result.metrics.loc[selected_model_option, "MAPE (%)"]
    summary_columns[3].metric("Selected model MAPE", f"{selected_mape:.4f}%")
st.caption(f"Source: {source_name} · {len(data):,} total source rows")

overview_tab, data_tab, export_tab = st.tabs(
    ["Forecast overview", "Source data & outliers", "Row-level exports"]
)

with overview_tab:
    forecast_chart = forecast_figure(result, selected_models)
    st.pyplot(forecast_chart, width="stretch")
    plt.close(forecast_chart)

    performance_title = (
        "Model accuracy comparison on 2016–2025 data"
        if selected_model_option == ALL_MODELS_OPTION
        else f"{MODEL_LABELS[selected_model_option]} accuracy on 2016–2025 data"
    )
    st.subheader(performance_title)
    metrics = result.metrics.loc[selected_models].reset_index()
    metrics["Model"] = metrics["Model"].map(MODEL_LABELS)
    st.dataframe(
        metrics.style.format(
            {
                "MAE": "{:,.2f}",
                "RMSE": "{:,.2f}",
                "MAPE (%)": "{:.4f}",
                "R²": "{:.4f}",
                "Accuracy (%)": "{:.4f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Accuracy is shown as 100 − MAPE, matching the notebook. It is an "
        "in-sample descriptive score, not a validation-set accuracy measure."
    )

    st.subheader("Actual and predicted values")
    actual_predicted_chart = actual_vs_predicted_figure(result, selected_models)
    st.pyplot(actual_predicted_chart, width="stretch")
    plt.close(actual_predicted_chart)

    st.subheader("Predicted totals")
    future_table = pd.DataFrame(
        {
            "Year": future_years,
            **{
                MODEL_LABELS[name]: np.rint(result.future[name])
                .clip(min=0)
                .astype(np.int64)
                for name in selected_models
            },
        }
    )
    st.dataframe(
        future_table.style.format(
            {MODEL_LABELS[name]: "{:,}" for name in selected_models}
        ),
        hide_index=True,
        width="stretch",
    )

with data_tab:
    st.subheader("Filtered source data")
    st.dataframe(filtered_data, hide_index=True, width="stretch", height=420)

    st.subheader("Nationwide outlier review")
    st.caption(
        "Checks always use the complete nationwide dataset as their reference. "
        "Sidebar selections and the year selection below only filter the displayed flags. "
        "A flag indicates a record to review; it is not automatically an error."
    )
    control_column, year_column = st.columns([1, 3])
    threshold = control_column.slider(
        "Robust-score threshold",
        min_value=2.5,
        max_value=6.0,
        value=3.5,
        step=0.25,
        help="Higher values produce fewer, more exceptional flags.",
    )
    selected_outlier_years = year_column.multiselect(
        "Years to display",
        HISTORICAL_YEARS,
        default=HISTORICAL_YEARS,
    )
    if not selected_outlier_years:
        st.info("Select at least one year to display outlier results.")
    else:
        outlier_results = cached_outlier_checks(data, threshold)
        nationwide_check, historical_check, demographic_check = st.tabs(
            [
                "Nationwide district comparison",
                "Historical change",
                "Demographic consistency",
            ]
        )
        with nationwide_check:
            show_outlier_check(
                "Nationwide district comparison",
                "For each age, sex, and year, compares that group's share of its "
                "district population across all districts using median-based robust scores.",
                outlier_results.nationwide,
                filtered_data,
                selected_outlier_years,
            )
        with historical_check:
            show_outlier_check(
                "Historical change",
                "Compares each age 1–79 population with the same birth cohort in the "
                "previous year, then evaluates that change against all districts.",
                outlier_results.historical,
                filtered_data,
                selected_outlier_years,
            )
        with demographic_check:
            show_outlier_check(
                "Demographic consistency",
                "Combines unusual male-share checks with sharp local breaks between "
                "adjacent ages. The open-ended 80+ group is excluded from age-neighbor checks.",
                outlier_results.demographic,
                filtered_data,
                selected_outlier_years,
            )

with export_tab:
    st.write(
        "Generate predictions for every currently filtered row. The CSV includes "
        "all three models; the HTML export follows the notebook and keeps the "
        "Polynomial forecast with row and grand totals."
    )
    predicted = cached_row_forecast(filtered_data, future_end)
    html_table = polynomial_html_table(predicted, future_end)
    prediction_suffixes = tuple(MODEL_EXPORT_SUFFIXES[name] for name in MODEL_NAMES)
    selected_suffixes = tuple(MODEL_EXPORT_SUFFIXES[name] for name in selected_models)
    preview_columns = [
        column
        for column in predicted.columns
        if not column.endswith(prediction_suffixes)
        or column.endswith(selected_suffixes)
    ]
    st.caption(
        "The preview follows the selected model option. The all-model CSV continues to "
        "include Linear Regression, Polynomial Regression, and Exponential Growth."
    )
    st.dataframe(
        predicted[preview_columns], hide_index=True, width="stretch", height=480
    )

    csv_column, html_column = st.columns(2)
    csv_column.download_button(
        "Download all-model CSV",
        dataframe_to_csv_bytes(predicted),
        file_name="dataDistrict_Predicted_All_Models.csv",
        mime="text/csv",
        width="stretch",
    )
    html_column.download_button(
        "Download Polynomial HTML",
        dataframe_to_html_bytes(html_table),
        file_name="dataDistrict_Predicted_Polynomial.html",
        mime="text/html",
        width="stretch",
    )

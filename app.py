"""Interactive Streamlit implementation of estimatePop.ipynb."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import mimetypes
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
import streamlit as st

from outlier_detection import (
    OutlierCorrectionResult,
    OutlierResults,
    correct_population_outliers,
)
from population_forecast import (
    HISTORICAL_COLUMNS,
    HISTORICAL_YEARS,
    MODEL_NAMES,
    annual_growth_rates,
    dataframe_to_csv_bytes,
    dataframe_to_html_bytes,
    forecast_each_row,
    forecast_totals,
    polynomial_html_table,
    time_holdout_validation,
    validate_population_data,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "data" / "dataDistrict.csv"
PRESENTATION_FILE = APP_DIR / "index.html"
PLOTLY_JAVASCRIPT = get_plotlyjs()
MODEL_STYLES = {
    "Linear": {"color": "#FF2B2B", "linestyle": "--"},
    "Polynomial": {"color": "#29B09D", "linestyle": "-."},
    "Exponential": {"color": "#FF8700", "linestyle": ":"},
}
MODEL_LABELS = {
    "Linear": "Linear Regression",
    "Polynomial": "Polynomial Regression",
    "Exponential": "Exponential Growth",
}
PRESENTATION_MODEL_LABELS = {
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
CORRECTED_INPUT_OPTION = "Corrected data (recommended)"
ORIGINAL_INPUT_OPTION = "Original data"
DEFAULT_CORRECTION_THRESHOLD = 3.5
STREAMLIT_CHART_BLUE = "#0068C9"
STREAMLIT_CHART_TEXT = "#31333F"
STREAMLIT_CHART_MUTED = "#6B7280"
STREAMLIT_CHART_GRID = "#E6E9EF"
STREAMLIT_CHART_FORECAST_BAND = "#EAF4FB"

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


@st.cache_data(show_spinner=False)
def load_embedded_presentation(
    html_modified_time: float,
    css_modified_time: float,
    js_modified_time: float,
    image_modified_times: tuple[tuple[str, float], ...],
    live_results_json: str,
) -> str:
    """Build a self-contained presentation for Streamlit's HTML component."""
    del html_modified_time, css_modified_time, js_modified_time, image_modified_times
    html = PRESENTATION_FILE.read_text(encoding="utf-8")
    css = (APP_DIR / "image" / "style.css").read_text(encoding="utf-8")
    javascript = (APP_DIR / "image" / "js.js").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="image/style.css">',
        f"<style>{css}</style>",
    )
    html = html.replace('<script src="image/js.js" defer></script>', "")

    for image_path in (APP_DIR / "image").iterdir():
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if not mime_type or not mime_type.startswith("image/"):
            continue
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        html = html.replace(
            f'src="image/{image_path.name}"',
            f'src="data:{mime_type};base64,{encoded}"',
        )
    encoded_results = base64.b64encode(live_results_json.encode("utf-8")).decode("ascii")
    live_results_script = (
        "<script>window.PRESENTATION_RESULTS = JSON.parse("
        f"decodeURIComponent(escape(atob('{encoded_results}'))));</script>"
    )
    plotly_script = f"<script>{PLOTLY_JAVASCRIPT}</script>"
    html = html.replace(
        "</head>", f"{plotly_script}{live_results_script}</head>"
    )
    return html.replace("</body>", f"<script>{javascript}</script></body>")


@st.cache_data(show_spinner="Calculating row-level forecasts…")
def cached_row_forecast(df: pd.DataFrame, future_end: int) -> pd.DataFrame:
    return forecast_each_row(df, future_end)


@st.cache_data(show_spinner="Preparing conservative outlier corrections…")
def cached_outlier_correction(
    df: pd.DataFrame, threshold: float
) -> OutlierCorrectionResult:
    return correct_population_outliers(
        df,
        HISTORICAL_COLUMNS,
        threshold=threshold,
        max_corrections_per_series=2,
    )


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


def style_streamlit_chart_axis(
    axis: plt.Axes, *, grid_axis: str = "y", show_bottom_spine: bool = True
) -> None:
    """Apply the clean light visual language used by Streamlit's native charts."""
    axis.set_facecolor("#FFFFFF")
    axis.set_axisbelow(True)
    axis.grid(
        axis=grid_axis,
        color=STREAMLIT_CHART_GRID,
        linestyle="-",
        linewidth=0.9,
        alpha=1.0,
        zorder=0,
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_visible(show_bottom_spine)
    axis.spines["bottom"].set_color(STREAMLIT_CHART_GRID)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.tick_params(
        axis="both",
        colors=STREAMLIT_CHART_MUTED,
        labelsize=9,
        length=0,
        pad=7,
    )
    axis.xaxis.label.set_color(STREAMLIT_CHART_TEXT)
    axis.yaxis.label.set_color(STREAMLIT_CHART_TEXT)
    axis.title.set_color(STREAMLIT_CHART_TEXT)


def forecast_figure(result, selected_models: list[str]) -> plt.Figure:
    fig, (forecast_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 11),
        facecolor="#FFFFFF",
        gridspec_kw={"height_ratios": [2, 1.5], "hspace": 0.3},
    )
    forecast_axis.plot(
        HISTORICAL_YEARS,
        result.actual,
        color=STREAMLIT_CHART_BLUE,
        marker="o",
        markerfacecolor="#FFFFFF",
        markeredgecolor=STREAMLIT_CHART_BLUE,
        markeredgewidth=1.8,
        label="Actual population",
        linewidth=2.6,
        markersize=6.5,
        zorder=5,
    )
    for name in selected_models:
        forecast_axis.plot(
            result.all_years,
            result.predictions[name],
            label=MODEL_LABELS[name],
            linewidth=2.3,
            zorder=4,
            **MODEL_STYLES[name],
        )
    forecast_axis.axvline(
        2025, color="#9CA3AF", linestyle="--", linewidth=1.1, alpha=0.9
    )
    forecast_axis.axvspan(
        2025,
        result.all_years[-1],
        color=STREAMLIT_CHART_FORECAST_BAND,
        alpha=0.8,
        zorder=0,
    )
    forecast_axis.set_title(
        "ສົມທຽບຄວາມຊັດເຈນຂອງ ຜົນການຄາດຄະເນປະຊາກອນ",
        loc="left",
        fontsize=14,
        fontweight="semibold",
        pad=14,
    )
    forecast_axis.set_ylabel("ປະຊາກອນ (ຄົນ)")
    forecast_axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    forecast_axis.legend(
        ncol=2,
        frameon=False,
        loc="upper left",
        labelcolor=STREAMLIT_CHART_TEXT,
    )
    style_streamlit_chart_axis(forecast_axis)

    year_spacing = 1.35
    year_positions = np.arange(len(HISTORICAL_YEARS)) * year_spacing
    group_width = 0.78
    bar_width = group_width / len(selected_models)
    for model_position, name in enumerate(selected_models):
        residuals = np.asarray(result.actual - result.fitted[name], dtype=float)
        bar_positions = (
            year_positions
            - group_width / 2
            + bar_width / 2
            + model_position * bar_width
        )
        bars = residual_axis.bar(
            bar_positions,
            residuals,
            width=bar_width,
            color=MODEL_STYLES[name]["color"],
            edgecolor="#FFFFFF",
            linewidth=1.0,
            label=MODEL_LABELS[name],
            zorder=3,
        )
        for bar, residual in zip(bars, residuals):
            rounded_residual = int(np.rint(residual))
            value_label = (
                "0"
                if rounded_residual == 0
                else f"{rounded_residual:+,}".replace("-", "−")
            )
            residual_axis.annotate(
                value_label,
                xy=(bar.get_x() + bar.get_width() / 2, residual),
                xytext=(0, 4 if residual >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if residual >= 0 else "top",
                fontsize=8 if len(selected_models) == 1 else 7,
                color=MODEL_STYLES[name]["color"],
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#FFFFFF",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )
    for divider_position in (year_positions[:-1] + year_positions[1:]) / 2:
        residual_axis.axvline(
            divider_position,
            color="#F0F2F6",
            linewidth=0.8,
            alpha=1.0,
            zorder=1,
        )
    residual_axis.axhline(
        0, color=STREAMLIT_CHART_TEXT, linewidth=1.1, zorder=4
    )
    
    residual_axis.set_title(
        "ຄ່າຜິດດ່ຽງຂອງການຄາດຄະເນ (ຕົວຈິງ − ຜົນການຄາດຄະເນ)",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=14,
    )
    residual_axis.set_xlabel("ປີ")
    residual_axis.set_ylabel("ປະຊາກອນ (ຄົນ)")
    residual_axis.set_xticks(year_positions, HISTORICAL_YEARS)
    residual_axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    residual_axis.legend(
        ncol=3,
        frameon=False,
        loc="upper left",
        labelcolor=STREAMLIT_CHART_TEXT,
    )
    style_streamlit_chart_axis(residual_axis)
    residual_axis.margins(y=0.22)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.06, hspace=0.34)
    return fig


def interactive_forecast_figure(
    result, selected_models: list[str], *, lao_presentation: bool = False
) -> go.Figure:
    """Build the interactive Streamlit forecast and residual chart."""
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.16,
        subplot_titles=(
            "ສົມທຽບຄວາມຊັດເຈນຂອງ ຜົນການຄາດຄະເນປະຊາກອນ",
            "ຄ່າຜິດດ່ຽງຂອງການຄາດຄະເນ (ຕົວຈິງ − ຜົນການຄາດຄະເນ)",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=HISTORICAL_YEARS,
            y=result.actual,
            name=(
                "ປະຊາກອນຕົວຈິງ (Actual population)"
                if lao_presentation
                else "Actual population"
            ),
            mode="lines+markers",
            line={"color": STREAMLIT_CHART_BLUE, "width": 3},
            marker={
                "color": "#FFFFFF",
                "line": {"color": STREAMLIT_CHART_BLUE, "width": 2},
                "size": 8,
            },
            hovertemplate=(
                (
                    "<b>ປະຊາກອນຈິງ (Actual population)</b><br>ປີ (Year): %{x}<br>"
                    "ປະຊາກອນ (Population): %{y:,.0f}<extra></extra>"
                )
                if lao_presentation
                else (
                    "<b>Actual population</b><br>Year: %{x}<br>"
                    "Population: %{y:,.0f}<extra></extra>"
                )
            ),
        ),
        row=1,
        col=1,
    )
    dash_styles = {"Linear": "dash", "Polynomial": "dashdot", "Exponential": "dot"}
    for name in selected_models:
        label = (
            PRESENTATION_MODEL_LABELS[name]
            if lao_presentation
            else MODEL_LABELS[name]
        )
        figure.add_trace(
            go.Scatter(
                x=result.all_years,
                y=result.predictions[name],
                name=label,
                mode="lines",
                line={
                    "color": MODEL_STYLES[name]["color"],
                    "width": 2.5,
                    "dash": dash_styles[name],
                },
                hovertemplate=(
                    f"<b>{label}</b><br>ປີ (Year): %{{x}}<br>"
                    "ປະຊາກອນ (Population): %{y:,.0f}<extra></extra>"
                    if lao_presentation
                    else f"<b>{label}</b><br>Year: %{{x}}<br>Population: %{{y:,.0f}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        residuals = np.asarray(result.actual - result.fitted[name], dtype=float)
        residual_labels = [
            "0" if int(np.rint(value)) == 0 else f"{int(np.rint(value)):+,}".replace("-", "−")
            for value in residuals
        ]
        figure.add_trace(
            go.Bar(
                x=HISTORICAL_YEARS,
                y=residuals,
                name=(
                    f"ຄ່າຄົງເຫຼືອຂອງ {label} (residual)"
                    if lao_presentation
                    else f"{label} residual"
                ),
                marker={
                    "color": MODEL_STYLES[name]["color"],
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                text=residual_labels,
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>ປີ (Year): %{{x}}<br>"
                    "ຄ່າຄົງເຫຼືອ (Residual: Actual − Predicted): %{y:+,.0f}<extra></extra>"
                    if lao_presentation
                    else f"<b>{label}</b><br>Year: %{{x}}<br>Residual (Actual − Predicted): %{{y:+,.0f}}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    figure.add_vrect(
        x0=2025,
        x1=float(result.all_years[-1]),
        fillcolor=STREAMLIT_CHART_FORECAST_BAND,
        opacity=0.8,
        line_width=0,
        layer="below",
        row=1,
        col=1,
    )
    figure.add_vline(
        x=2025,
        line={"color": "#9CA3AF", "width": 1.2, "dash": "dash"},
        row=1,
        col=1,
    )
    figure.add_hline(
        y=0,
        line={"color": STREAMLIT_CHART_TEXT, "width": 1.2},
        row=2,
        col=1,
    )
    figure.update_xaxes(
        title_text="ປີ",
        tickmode="array",
        tickvals=HISTORICAL_YEARS,
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="ປະຊາກອນ (ຄົນ)", tickformat=",.0f")
    figure.update_layout(
        template="plotly_white",
        height=900,
        barmode="group",
        hovermode="closest",
        margin={"l": 55, "r": 25, "t": 70, "b": 55},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        hoverlabel={"bgcolor": "#FFFFFF", "font_color": STREAMLIT_CHART_TEXT},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    figure.update_xaxes(gridcolor=STREAMLIT_CHART_GRID, zeroline=False)
    figure.update_yaxes(gridcolor=STREAMLIT_CHART_GRID, zeroline=False)
    figure.update_annotations(font={"color": STREAMLIT_CHART_TEXT, "size": 16})
    return figure


def population_growth_rate_summary(
    result, selected_models: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build a compact annual-growth table and series averages."""
    years = np.asarray(result.all_years[1:])
    actual_rates = annual_growth_rates(result.actual)
    actual_column = np.full(years.shape, np.nan, dtype=float)
    actual_column[: actual_rates.size] = actual_rates

    table_data = {"Year": years, "Actual": actual_column}
    averages = {"Actual": float(np.nanmean(actual_rates))}
    for name in selected_models:
        label = MODEL_LABELS[name]
        rates = annual_growth_rates(result.predictions[name])
        table_data[label] = rates
        averages[label] = float(np.nanmean(rates))

    return pd.DataFrame(table_data), averages


def actual_vs_predicted_figure(result, selected_models: list[str]) -> plt.Figure:
    """Compare historical observations with each selected model's fitted values."""
    panel_count = len(selected_models)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(13, 4.8 if panel_count == 1 else 3.8 * panel_count),
        squeeze=False,
        sharex=True,
        facecolor="#FFFFFF",
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
            color=STREAMLIT_CHART_BLUE,
            edgecolor="#FFFFFF",
            linewidth=1.0,
            label="Actual",
            zorder=3,
        )
        predicted_bars = axis.bar(
            x_positions + column_width / 2,
            predicted_values,
            width=column_width,
            color=MODEL_STYLES[name]["color"],
            edgecolor="#FFFFFF",
            linewidth=1.0,
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
                label_color = STREAMLIT_CHART_MUTED
            else:
                rounded_difference = round(float(difference), 4)
                if rounded_difference == 0:
                    label = "0.0000%"
                    label_color = STREAMLIT_CHART_MUTED
                else:
                    label = f"{rounded_difference:+.4f}%".replace("-", "−")
                    label_color = "#29B09D" if rounded_difference > 0 else "#FF2B2B"
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

        axis.set_title(
            MODEL_LABELS[name],
            loc="left",
            fontsize=12,
            fontweight="semibold",
            pad=12,
        )
        axis.set_ylabel("Population")
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        style_streamlit_chart_axis(axis)
        axis.tick_params(axis="x", labelbottom=True)
        axis.margins(y=0.18)
        axis.legend(
            handles=[actual_bars, predicted_bars],
            labels=["Actual", "Predicted"],
            frameon=False,
            ncol=2,
            loc="upper left",
            labelcolor=STREAMLIT_CHART_TEXT,
        )

    axes.flat[-1].set_xlabel("Year")
    for axis in axes.flat:
        axis.set_xticks(x_positions, HISTORICAL_YEARS)
    fig.suptitle(
        "Actual vs predicted population (historical fit)",
        fontsize=15,
        fontweight="semibold",
        color=STREAMLIT_CHART_TEXT,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def interactive_actual_vs_predicted_figure(
    result, selected_models: list[str], *, lao_presentation: bool = False
) -> go.Figure:
    """Build interactive historical actual-versus-predicted panels."""
    panel_count = len(selected_models)
    figure = make_subplots(
        rows=panel_count,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1 if panel_count > 1 else 0.08,
        subplot_titles=tuple(
            (
                PRESENTATION_MODEL_LABELS[name]
                if lao_presentation
                else MODEL_LABELS[name]
            )
            for name in selected_models
        ),
    )
    actual_values = np.asarray(result.actual, dtype=float)
    for row_number, name in enumerate(selected_models, start=1):
        label = (
            PRESENTATION_MODEL_LABELS[name]
            if lao_presentation
            else MODEL_LABELS[name]
        )
        predicted_values = np.asarray(result.fitted[name], dtype=float)
        percentage_differences = np.divide(
            predicted_values - actual_values,
            actual_values,
            out=np.full_like(predicted_values, np.nan),
            where=actual_values != 0,
        ) * 100
        difference_labels = []
        label_colors = []
        hover_differences = []
        for difference in percentage_differences:
            if np.isnan(difference):
                difference_labels.append("N/A")
                hover_differences.append("N/A")
                label_colors.append(STREAMLIT_CHART_MUTED)
                continue
            rounded = round(float(difference), 4)
            difference_labels.append(
                "0.0000%" if rounded == 0 else f"{rounded:+.4f}%".replace("-", "−")
            )
            hover_differences.append(f"{float(difference):+.4f}%")
            label_colors.append(
                STREAMLIT_CHART_MUTED
                if rounded == 0
                else "#29B09D" if rounded > 0 else "#FF2B2B"
            )

        figure.add_trace(
            go.Bar(
                x=HISTORICAL_YEARS,
                y=actual_values,
                name="ຄ່າຈິງ (Actual)" if lao_presentation else "Actual",
                legendgroup="actual",
                showlegend=row_number == 1,
                marker={
                    "color": STREAMLIT_CHART_BLUE,
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                hovertemplate=(
                    "<b>ຄ່າຈິງ (Actual)</b><br>ປີ (Year): %{x}<br>"
                    "ປະຊາກອນ (Population): %{y:,.0f}<extra></extra>"
                    if lao_presentation
                    else "<b>Actual</b><br>Year: %{x}<br>Population: %{y:,.0f}<extra></extra>"
                ),
            ),
            row=row_number,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=HISTORICAL_YEARS,
                y=predicted_values,
                name="ຄ່າຄາດຄະເນ (Predicted)" if lao_presentation else "Predicted",
                legendgroup="predicted",
                showlegend=row_number == 1,
                marker={
                    "color": MODEL_STYLES[name]["color"],
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                text=difference_labels,
                textfont={"color": label_colors, "size": 11},
                textposition="outside",
                cliponaxis=False,
                customdata=np.asarray(hover_differences).reshape(-1, 1),
                hovertemplate=(
                    f"<b>ຜົນຄາດຄະເນ {label} (prediction)</b><br>ປີ (Year): %{{x}}<br>"
                    "ປະຊາກອນ (Population): %{y:,.0f}<br>ຜົນຕ່າງ (Difference): %{customdata[0]}"
                    "<extra></extra>"
                    if lao_presentation
                    else f"<b>{label} prediction</b><br>Year: %{{x}}<br>Population: %{{y:,.0f}}<br>Difference: %{{customdata[0]}}<extra></extra>"
                ),
            ),
            row=row_number,
            col=1,
        )
        figure.update_xaxes(
            tickmode="array",
            tickvals=HISTORICAL_YEARS,
            gridcolor=STREAMLIT_CHART_GRID,
            row=row_number,
            col=1,
        )
        figure.update_yaxes(
            title_text=(
                "ປະຊາກອນ (Population)"
                if lao_presentation
                else "Population"
            ),
            tickformat=",.0f",
            gridcolor=STREAMLIT_CHART_GRID,
            rangemode="tozero",
            row=row_number,
            col=1,
        )

    figure.update_xaxes(
        title_text="ປີ (Year)" if lao_presentation else "Year",
        row=panel_count,
        col=1,
    )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                "ປະຊາກອນຈິງ ທຽບກັບ ຄາດຄະເນ (Actual vs predicted population — historical fit)"
                if lao_presentation
                else "Actual vs predicted population (historical fit)"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        height=420 if panel_count == 1 else 350 * panel_count,
        barmode="group",
        bargap=0.2,
        bargroupgap=0.04,
        hovermode="closest",
        margin={"l": 55, "r": 25, "t": 95, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        hoverlabel={"bgcolor": "#FFFFFF", "font_color": STREAMLIT_CHART_TEXT},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    figure.update_annotations(font={"color": STREAMLIT_CHART_TEXT, "size": 16})
    return figure


def interactive_validation_figure(
    validation, selected_models: list[str], *, lao_presentation: bool = False
) -> go.Figure:
    """Compare holdout predictions with unseen historical observations."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=validation.test_years,
            y=validation.actual,
            name=(
                "ປະຊາກອນຈິງ (Actual population)"
                if lao_presentation
                else "Actual population"
            ),
            mode="lines+markers",
            line={"color": STREAMLIT_CHART_BLUE, "width": 4},
            marker={"size": 10, "color": "#FFFFFF", "line": {"color": STREAMLIT_CHART_BLUE, "width": 2.5}},
            hovertemplate=(
                "ຄ່າຈິງ (Actual)<br>ປີ (Year): %{x}<br>ປະຊາກອນ (Population): %{y:,.0f}<extra></extra>"
                if lao_presentation
                else "Actual<br>Year: %{x}<br>Population: %{y:,.0f}<extra></extra>"
            ),
        )
    )
    dash_styles = {"Linear": "dash", "Polynomial": "dashdot", "Exponential": "dot"}
    for name in selected_models:
        label = (
            PRESENTATION_MODEL_LABELS[name]
            if lao_presentation
            else MODEL_LABELS[name]
        )
        figure.add_trace(
            go.Scatter(
                x=validation.test_years,
                y=validation.predictions[name],
                name=label,
                mode="lines+markers",
                line={
                    "color": MODEL_STYLES[name]["color"],
                    "width": 3,
                    "dash": dash_styles[name],
                },
                marker={"size": 8},
                customdata=(
                    (validation.predictions[name] - validation.actual)
                    / np.where(validation.actual == 0, np.nan, validation.actual)
                    * 100
                ),
                hovertemplate=(
                    f"{label}<br>ປີ (Year): %{{x}}<br>ຄ່າຄາດຄະເນ (Prediction): %{{y:,.0f}}"
                    "<br>ຄວາມຜິດພາດ (Error): %{customdata:+.4f}%<extra></extra>"
                    if lao_presentation
                    else f"{label}<br>Year: %{{x}}<br>Prediction: %{{y:,.0f}}<br>Error: %{{customdata:+.4f}}%<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=440,
        hovermode="x unified",
        margin={"l": 70, "r": 28, "t": 28, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    figure.update_xaxes(
        title_text=(
            "ປີຂອງຊຸດທົດສອບ (Holdout year)"
            if lao_presentation
            else "Holdout year"
        ),
        tickmode="array",
        tickvals=validation.test_years,
        gridcolor=STREAMLIT_CHART_GRID,
    )
    figure.update_yaxes(
        title_text=(
            "ປະຊາກອນ (Population)"
            if lao_presentation
            else "Population"
        ),
        tickformat=",.0f",
        gridcolor=STREAMLIT_CHART_GRID,
    )
    return figure


def population_pyramid_figure(
    df: pd.DataFrame,
    district: str | None,
    year: int,
    year_label: str | None = None,
    age_display: str = "By age group",
) -> tuple[plt.Figure, int, int]:
    """Build a population pyramid for the requested district and age bands."""
    required_columns = {"ouName", "cocCode", "sexCode", str(year)}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "The population pyramid requires these columns: "
            + ", ".join(missing_columns)
        )

    district_data = (
        df.copy()
        if district is None
        else df[df["ouName"].astype(str).eq(district)].copy()
    )
    if district_data.empty:
        raise ValueError(f"No population records were found for {district}.")

    numeric_ages = pd.to_numeric(
        district_data["cocCode"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    if numeric_ages.isna().any():
        raise ValueError("The cocCode column contains an unrecognized age value.")
    if age_display == "By age group":
        profile_column = "Age group"
        profile_labels = ["0–4", "5–18", "19–25", "26–40", "41–60", "61–80+"]
        district_data[profile_column] = pd.cut(
            numeric_ages,
            bins=[-1, 4, 18, 25, 40, 60, np.inf],
            labels=profile_labels,
        )
        y_axis_label = "Age group"
        figure_height = 7.0
    elif age_display == "By single age":
        profile_column = "Age"
        district_data[profile_column] = district_data["cocCode"].astype(str).str.strip()
        profile_labels = sorted(
            district_data[profile_column].unique(),
            key=lambda age: (
                int("".join(character for character in age if character.isdigit()) or -1),
                "+" in age,
            ),
        )
        y_axis_label = "Age"
        figure_height = max(12.0, len(profile_labels) * 0.22)
    else:
        raise ValueError(f"Unknown age display option: {age_display}.")
    district_data["Population"] = pd.to_numeric(
        district_data[str(year)], errors="coerce"
    )
    if district_data["Population"].isna().any():
        raise ValueError(f"The {year} population column contains non-numeric values.")

    profile = district_data.pivot_table(
        index=profile_column,
        columns="sexCode",
        values="Population",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    ).reindex(profile_labels, fill_value=0)

    sex_codes = {str(code).strip().upper(): code for code in profile.columns}
    if "M" not in sex_codes or "F" not in sex_codes:
        raise ValueError("The population pyramid requires M and F values in sexCode.")

    male_values = profile[sex_codes["M"]].to_numpy(dtype=float)
    female_values = profile[sex_codes["F"]].to_numpy(dtype=float)
    positions = np.arange(len(profile.index))

    figure, axis = plt.subplots(figsize=(12, figure_height))
    axis.barh(
        positions,
        -male_values,
        height=0.82,
        color="#277da1",
        edgecolor="white",
        linewidth=0.35,
        label="Male",
    )
    axis.barh(
        positions,
        female_values,
        height=0.82,
        color="#e76f91",
        edgecolor="white",
        linewidth=0.35,
        label="Female",
    )
    axis.axvline(0, color="#495057", linewidth=0.9)
    axis.set_yticks(positions, profile.index)
    axis.tick_params(axis="y", labelsize=7)
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{abs(value):,.0f}")
    )
    population_limit = max(float(male_values.max()), float(female_values.max()), 1.0)
    axis.set_xlim(-population_limit * 1.12, population_limit * 1.12)
    axis.set_xlabel("Population (people)")
    axis.set_ylabel(y_axis_label)
    location_label = district if district is not None else "All districts"
    axis.set_title(
        f"Population pyramid — {location_label} ({year_label or year})",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    axis.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False, ncol=2)
    figure.tight_layout()
    return figure, int(np.rint(male_values.sum())), int(np.rint(female_values.sum()))


def figure_to_data_uri(figure: plt.Figure) -> str:
    """Serialize a live Matplotlib result for the embedded presentation."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
) -> pd.DataFrame:
    st.markdown(f"#### {title}")
    st.caption(description)
    visible = filter_outlier_results(results, visible_data)
    visible = visible[visible["Year"].isin(years)].reset_index(drop=True)
    st.metric("Flagged observations", f"{len(visible):,}")
    if visible.empty:
        st.success("No observations exceed the selected robust-score threshold.")
        return visible
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
    return visible


st.markdown(
    """
    <style>
    @keyframes global-loading-pulse {
        0%, 100% {
            box-shadow: 0 10px 30px rgba(21, 101, 192, 0.22);
            transform: translate(-50%, -50%) scale(1);
        }
        50% {
            box-shadow: 0 12px 34px rgba(2, 136, 209, 0.38);
            transform: translate(-50%, -50%) scale(1.025);
        }
    }

    .stApp[data-test-script-state="running" i]::before {
        content: "⏳ ກຳລັງດຳເນີນການ, ກະລຸນາລໍຖ້າ…";
        position: fixed;
        left: 50%;
        top: 50%;
        z-index: 999999;
        padding: 1.15rem 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.38);
        border-radius: 999px;
        background: linear-gradient(135deg, #1565c0 0%, #0288d1 100%);
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        line-height: 1.25;
        pointer-events: none;
        animation: global-loading-pulse 1.15s ease-in-out infinite;
    }

    @media (max-width: 640px) {
        .stApp[data-test-script-state="running" i]::before {
            max-width: calc(100vw - 2rem);
            padding: 0.95rem 1.25rem;
            font-size: 1.1rem;
            text-align: center;
            white-space: nowrap;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp[data-test-script-state="running" i]::before {
            animation: none;
            transform: translate(-50%, -50%);
        }
    }

    .baseline-caption {
        display: inline-flex;
        align-items: center;
        margin-top: 0.15rem;
        padding: 0.45rem 0.85rem;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #0288d1;
        border-radius: 0.55rem;
        background: linear-gradient(135deg, #eff8ff 0%, #e8f4ff 100%);
        color: #0f4c81;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.1);
    }

    div[data-testid="stButtonGroup"] {
        margin: 0.35rem 0 1.25rem;
    }

    div[data-testid="stButtonGroup"] [role="radiogroup"] {
        gap: 0.55rem !important;
        padding: 0.42rem !important;
        border: 1px solid #dbe5f0;
        border-radius: 1rem;
        background: linear-gradient(135deg, #f8fbff 0%, #f1f5f9 100%);
        box-shadow: 0 8px 24px rgba(15, 76, 129, 0.08);
    }

    div[data-testid="stButtonGroup"] button {
        min-height: 3.15rem;
        position: relative;
        border: 1px solid #dbe5f0 !important;
        border-radius: 0.75rem !important;
        background: #ffffff !important;
        color: #475569 !important;
        font-weight: 650 !important;
        letter-spacing: 0.01em;
        transition:
            transform 160ms ease,
            box-shadow 160ms ease,
            background 160ms ease,
            color 160ms ease !important;
    }

    div[data-testid="stButtonGroup"] button:hover {
        border-color: #bfdbfe !important;
        background: #ffffff !important;
        color: #1565c0 !important;
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.12);
        transform: translateY(-1px);
    }

    div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {
        border-color: transparent !important;
        background: linear-gradient(135deg, #1565c0 0%, #0288d1 100%) !important;
        color: #ffffff !important;
        box-shadow:
            0 8px 20px rgba(21, 101, 192, 0.32),
            inset 0 1px 0 rgba(255, 255, 255, 0.28);
        transform: translateY(-1px);
    }

    div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] *,
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] * {
        color: #ffffff !important;
    }

    div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"]::after,
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"]::after {
        content: "✓";
        display: inline-grid;
        place-items: center;
        width: 1.35rem;
        height: 1.35rem;
        margin-left: 0.5rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 800;
    }

    div[data-testid="stButtonGroup"] button:focus-visible {
        outline: 3px solid rgba(2, 136, 209, 0.28) !important;
        outline-offset: 2px;
    }

    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        gap: 0.55rem;
        padding: 0.42rem;
        overflow-x: auto;
        border: 1px solid #dbe5f0;
        border-radius: 1rem;
        background: linear-gradient(135deg, #f8fbff 0%, #f1f5f9 100%);
        box-shadow: 0 8px 24px rgba(15, 76, 129, 0.08);
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        flex: 1 0 auto;
        min-height: 3rem;
        padding: 0.65rem 1rem;
        border: 1px solid #dbe5f0;
        border-radius: 0.75rem;
        background: #ffffff;
        color: #475569;
        font-weight: 650;
        letter-spacing: 0.01em;
        transition:
            transform 160ms ease,
            box-shadow 160ms ease,
            background 160ms ease,
            color 160ms ease;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        border-color: #bfdbfe;
        color: #1565c0;
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.12);
        transform: translateY(-1px);
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        border-color: transparent;
        background: linear-gradient(135deg, #1565c0 0%, #0288d1 100%);
        color: #ffffff;
        box-shadow:
            0 8px 20px rgba(21, 101, 192, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.28);
        transform: translateY(-1px);
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] * {
        color: #ffffff !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"]:focus-visible {
        outline: 3px solid rgba(2, 136, 209, 0.28);
        outline-offset: 2px;
    }

    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

app_view = st.segmented_control(
    "Page",
    ["Presentation", "Demo"],
    default="Demo",
    format_func=lambda view: (
        "🎞️  Presentation" if view == "Presentation" else "📊  Demo"
    ),
    label_visibility="collapsed",
    width="stretch",
)

if app_view == "Presentation":
    if not PRESENTATION_FILE.exists():
        st.error(f"Presentation file not found: {PRESENTATION_FILE}")
        st.stop()
    st.markdown(
        """
        <style>
        .stAppHeader { display: none; }
        [data-testid="stMainBlockContainer"] {
            max-width: none;
            padding: 0.5rem;
        }
        iframe[title="st.iframe"] {
            height: calc(100vh - 4.5rem) !important;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    presentation_html = load_embedded_presentation(
        PRESENTATION_FILE.stat().st_mtime,
        (APP_DIR / "image" / "style.css").stat().st_mtime,
        (APP_DIR / "image" / "js.js").stat().st_mtime,
        tuple(
            (path.name, path.stat().st_mtime)
            for path in (APP_DIR / "image").iterdir()
            if path.is_file()
        ),
        json.dumps(
            st.session_state.get("presentation_results", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    st.iframe(presentation_html, height=900, width="stretch")
    st.stop()

st.title("ການຄາດຄະເນປະຊາກອນລາວ (ແຍກເປັນ: ເມືອງ, ອາຍຸ ແລະ ເພດ)")
st.caption(
    '<span class="baseline-caption">ຂໍ້ມູນພື້ນຖານ (baseline)</span>',
    unsafe_allow_html=True,
)

st.sidebar.header("Data and forecast")
uploaded_file = st.sidebar.file_uploader("Upload population CSV", type=["csv"])
future_end = st.sidebar.slider("Forecast through", 2026, 2040, 2030)
correction_threshold = st.sidebar.slider(
    "Outlier correction threshold",
    min_value=2.5,
    max_value=6.0,
    value=DEFAULT_CORRECTION_THRESHOLD,
    step=0.25,
    help=(
        "Lower values flag and potentially correct more observations. Higher values "
        "require more exceptional robust scores. Every correction still requires an "
        "independent demographic confirmation."
    ),
)
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

forecast_input_option = st.segmented_control(
    "Baseline data type",
    [ORIGINAL_INPUT_OPTION, CORRECTED_INPUT_OPTION],
    default=CORRECTED_INPUT_OPTION,
    format_func=lambda option: (
        "Original data"
        if option == ORIGINAL_INPUT_OPTION
        else f"Outlier-corrected data (threshold {correction_threshold:g})"
    ),
    help=(
        "Choose the baseline used by the forecast. Outlier-corrected data changes only "
        "temporal anomalies independently confirmed by a demographic check; the "
        "original upload is never overwritten."
    ),
    width="stretch",
)
forecast_input_option = forecast_input_option or CORRECTED_INPUT_OPTION
forecast_input_label = (
    "Original data"
    if forecast_input_option == ORIGINAL_INPUT_OPTION
    else f"Outlier-corrected data (threshold {correction_threshold:g})"
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
    correction_result = cached_outlier_correction(data, correction_threshold)
    corrected_data = correction_result.data
except Exception as error:
    st.error(f"Could not load the population data: {error}")
    st.stop()

filtered_original_data = filter_data(data)
if filtered_original_data.empty:
    st.warning("The selected filters contain no records. Change or clear the filters.")
    st.stop()

with st.sidebar.expander("5–7 minute demo path"):
    st.markdown(
        """
1. **Forecast Overview** — show holdout validation, selected model and national forecast.
2. **Source data & outliers** — explain the nationwide correction summary and audit trail.
3. **Population pyramid** — switch between nationwide and one district.
4. **Row-level exports** — show reusable CSV/HTML outputs.

The methodology tab is supporting material; it does not need to be opened during the timed demo.
        """
    )

filtered_corrected_data = corrected_data.loc[filtered_original_data.index].copy()
filtered_data = (
    filtered_corrected_data
    if forecast_input_option == CORRECTED_INPUT_OPTION
    else filtered_original_data
)

result = forecast_totals(filtered_data, future_end)
validation_result = time_holdout_validation(filtered_data, train_end=2022)
original_forecast = forecast_totals(filtered_original_data, future_end)
corrected_forecast = forecast_totals(filtered_corrected_data, future_end)
nationwide_original_forecast = forecast_totals(data, future_end)
nationwide_corrected_forecast = forecast_totals(corrected_data, future_end)
future_years = list(range(2026, future_end + 1))
row_predictions = cached_row_forecast(filtered_data, future_end)
polynomial_year_columns = [f"{year}_Poly" for year in future_years]
polynomial_identity_columns = ["ouName", "deName", "cocName", "sexName"]
polynomial_table = row_predictions.reindex(
    columns=[*polynomial_identity_columns, *polynomial_year_columns]
)

polynomial_table_rows = []
for row in polynomial_table.itertuples(index=False, name=None):
    identity_values = [
        "" if pd.isna(value) else str(value)
        for value in row[: len(polynomial_identity_columns)]
    ]
    forecast_values = [
        "" if pd.isna(value) else f"{int(value):,}"
        for value in row[len(polynomial_identity_columns) :]
    ]
    polynomial_table_rows.append([*identity_values, *forecast_values])

presentation_corrections = correction_result.corrections.copy()
presentation_corrected_series = (
    presentation_corrections[["District", "Age group", "Sex"]]
    .drop_duplicates()
    .shape[0]
    if not presentation_corrections.empty
    else 0
)
presentation_cell_count = len(data) * len(HISTORICAL_COLUMNS)
presentation_method_counts = (
    presentation_corrections["Correction method"].value_counts().to_dict()
    if not presentation_corrections.empty
    else {}
)
presentation_audit_rows = []
presentation_correction_example = (
    "ບໍ່ມີການແກ້ໄຂລະດັບຊາດທີ່ຢືນຢັນແລ້ວ ຢູ່ເກນທີ່ເຮັດວຽກ (active threshold) ປັດຈຸບັນ."
)
presentation_outlier_evidence = (
    "ບໍ່ມີຫຼັກຖານຄ່າຜິດປົກກະຕິລະດັບຊາດທີ່ຢືນຢັນແລ້ວ ຢູ່ເກນທີ່ເຮັດວຽກ (active threshold) ປັດຈຸບັນ."
)
correction_comparison_figure = go.Figure()
outlier_evidence_figure = go.Figure()
nationwide_corrections = correction_result.corrections
if not nationwide_corrections.empty:
    outlier_evidence_figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "ຄ່າຜິດປົກກະຕິຕາມປີ ແລະ ເພດ (Confirmed cases by year and sex)",
            "ເມືອງທີ່ມີຄ່າຜິດປົກກະຕິຫຼາຍທີ່ສຸດ (Districts with the most confirmed cases)",
            "ກຸ່ມອາຍຸທີ່ຄ່າຜິດປົກກະຕິຫຼາຍທີ່ສຸດ (Age groups with the most confirmed cases)",
        ),
        vertical_spacing=0.24,
        horizontal_spacing=0.14,
        row_heights=[0.52, 0.48],
    )
    year_sex_counts = (
        nationwide_corrections.groupby(["Year", "Sex"]).size().unstack(fill_value=0)
        .reindex(HISTORICAL_YEARS, fill_value=0)
    )
    sex_colors = {"Female": "#D946EF", "Male": "#1686A3"}
    for sex_name in year_sex_counts.columns:
        values = year_sex_counts[sex_name].astype(int).tolist()
        outlier_evidence_figure.add_trace(
            go.Bar(
                x=HISTORICAL_YEARS,
                y=values,
                name=(
                    "ຍິງ (Female)"
                    if str(sex_name) == "Female"
                    else "ຊາຍ (Male)"
                    if str(sex_name) == "Male"
                    else str(sex_name)
                ),
                marker_color=sex_colors.get(str(sex_name), "#D97706"),
                text=values,
                textposition="auto",
                hovertemplate=(
                    f"{sex_name}<br>ປີ (Year): %{{x}}<br>ກໍລະນີຢືນຢັນ (Confirmed cases): %{{y:,}}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    top_districts = nationwide_corrections["District"].value_counts().head(10).sort_values()
    outlier_evidence_figure.add_trace(
        go.Bar(
            x=top_districts.astype(int),
            y=top_districts.index.astype(str),
            orientation="h",
            name="ກໍລະນີຕາມເມືອງ (District cases)",
            marker_color="#027235",
            text=top_districts.astype(int),
            textposition="auto",
            hovertemplate="%{y}<br>ກໍລະນີຢືນຢັນ (Confirmed cases): %{x:,}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    top_ages = nationwide_corrections["Age group"].value_counts().head(10).sort_values()
    outlier_evidence_figure.add_trace(
        go.Bar(
            x=top_ages.astype(int),
            y=top_ages.index.astype(str),
            orientation="h",
            name="ກໍລະນີຕາມກຸ່ມອາຍຸ (Age-group cases)",
            marker_color="#D97706",
            text=top_ages.astype(int),
            textposition="auto",
            hovertemplate="%{y}<br>ກໍລະນີຢືນຢັນ (Confirmed cases): %{x:,}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    outlier_evidence_figure.update_xaxes(
        title_text="ປີ (Year)",
        tickmode="array",
        tickvals=HISTORICAL_YEARS,
        gridcolor=STREAMLIT_CHART_GRID,
        row=1,
        col=1,
    )
    outlier_evidence_figure.update_yaxes(
        title_text="ກໍລະນີຢືນຢັນ (Confirmed cases)",
        rangemode="tozero",
        gridcolor=STREAMLIT_CHART_GRID,
        row=1,
        col=1,
    )
    outlier_evidence_figure.update_xaxes(
        title_text="ກໍລະນີຢືນຢັນ (Confirmed cases)", rangemode="tozero", row=2, col=1
    )
    outlier_evidence_figure.update_xaxes(
        title_text="ກໍລະນີຢືນຢັນ (Confirmed cases)", rangemode="tozero", row=2, col=2
    )
    outlier_evidence_figure.update_layout(
        template="plotly_white",
        height=560,
        barmode="stack",
        margin={"l": 90, "r": 36, "t": 58, "b": 52},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0,
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    outlier_evidence_figure.update_annotations(
        font={"color": STREAMLIT_CHART_TEXT, "size": 14}
    )
    nationwide_series_count = nationwide_corrections[
        ["District", "Age group", "Sex"]
    ].drop_duplicates().shape[0]
    presentation_outlier_evidence = (
        f"ພາບລວມທົ່ວປະເທດ (Nationwide overview) · ເກນ (threshold) {correction_threshold:g} · "
        f"{len(nationwide_corrections):,} ຄ່າສັງເກດທີ່ຢືນຢັນ (confirmed observations) · "
        f"{nationwide_corrections['District'].nunique():,} ເມືອງ (districts) · "
        f"{nationwide_series_count:,} ຊຸດຂໍ້ມູນເມືອງ–ອາຍຸ–ເພດ (district–age–sex series)"
    )
else:
    outlier_evidence_figure.add_annotation(
        text="ບໍ່ມີຄ່າຜິດປົກກະຕິລະດັບຊາດທີ່ຢືນຢັນ ຢູ່ເກນທີ່ເຮັດວຽກ (active threshold)",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18, "color": "#5E7077"},
    )
    outlier_evidence_figure.update_layout(
        template="plotly_white",
        height=300,
        margin={"l": 24, "r": 24, "t": 20, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )

if not presentation_corrections.empty:
    audit_preview = presentation_corrections.assign(
        _impact=presentation_corrections["Difference"].abs()
    ).nlargest(6, "_impact")
    for audit_row in audit_preview.to_dict("records"):
        presentation_audit_rows.append(
            {
                "district": str(audit_row["District"]),
                "age_group": str(audit_row["Age group"]),
                "sex": (
                    "ຍິງ (Female)"
                    if str(audit_row["Sex"]) == "Female"
                    else "ຊາຍ (Male)"
                    if str(audit_row["Sex"]) == "Male"
                    else str(audit_row["Sex"])
                ),
                "year": str(int(audit_row["Year"])),
                "original": f"{audit_row['Original value']:,.0f}",
                "corrected": f"{audit_row['Corrected value']:,.0f}",
                "method": {
                    "Two-sided log-linear interpolation": "ການຄາດຄ່າແຊກແບບລັອກ-ເສັ້ນຊື່ສອງດ້ານ (Two-sided log-linear interpolation)",
                    "Theil-Sen endpoint estimate": "ການປະເມີນຈຸດປາຍ Theil–Sen (Theil-Sen endpoint estimate)",
                }.get(
                    str(audit_row["Correction method"]),
                    str(audit_row["Correction method"]),
                ),
            }
        )

    original_series = data[HISTORICAL_COLUMNS].sum(axis=0).to_numpy(dtype=float)
    corrected_series = (
        corrected_data[HISTORICAL_COLUMNS].sum(axis=0).to_numpy(dtype=float)
    )
    corrected_years = sorted(presentation_corrections["Year"].astype(int).unique())
    corrected_positions = [HISTORICAL_YEARS.index(year) for year in corrected_years]
    annual_adjustments = corrected_series - original_series

    correction_comparison_figure.add_trace(
        go.Scatter(
            x=HISTORICAL_YEARS,
            y=original_series,
            name="ຍອດລວມທົ່ວປະເທດຕົ້ນສະບັບ (Original nationwide total)",
            mode="lines+markers",
            line={"color": "#D97706", "width": 3, "dash": "dash"},
            marker={"size": 7, "symbol": "circle-open"},
            hovertemplate=(
                "ຍອດລວມທົ່ວປະເທດຕົ້ນສະບັບ (Original nationwide total)<br>ປີ (Year): %{x}<br>ປະຊາກອນ (Population): %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    correction_comparison_figure.add_trace(
        go.Scatter(
            x=HISTORICAL_YEARS,
            y=corrected_series,
            name="ຍອດລວມທົ່ວປະເທດທີ່ແກ້ໄຂແລ້ວ (Corrected nationwide total)",
            mode="lines+markers",
            line={"color": "#027235", "width": 3},
            marker={"size": 7},
            customdata=annual_adjustments,
            hovertemplate=(
                "ຍອດລວມທົ່ວປະເທດທີ່ແກ້ໄຂແລ້ວ (Corrected nationwide total)<br>ປີ (Year): %{x}<br>ປະຊາກອນ (Population): %{y:,.0f}"
                "<br>ການປັບສຸດທິ (Net adjustment): %{customdata:+,.0f}<extra></extra>"
            ),
        )
    )
    correction_comparison_figure.add_trace(
        go.Scatter(
            x=corrected_years,
            y=original_series[corrected_positions],
            name="ກ່ອນແກ້ໄຂ — ປີທີ່ໄດ້ຮັບຜົນ (Before treatment — affected year)",
            mode="markers",
            marker={"color": "#DC2626", "size": 14, "symbol": "x", "line": {"width": 2}},
            hovertemplate=(
                "ປີທີ່ໄດ້ຮັບຜົນກ່ອນແກ້ໄຂ (Affected year before treatment)<br>ປີ (Year): %{x}"
                "<br>ຍອດລວມທົ່ວປະເທດ (Nationwide total): %{y:,.0f}<extra></extra>"
            ),
        )
    )
    correction_comparison_figure.add_trace(
        go.Scatter(
            x=corrected_years,
            y=corrected_series[corrected_positions],
            name="ຫຼັງແກ້ໄຂ — ປີທີ່ໄດ້ຮັບຜົນ (After treatment — affected year)",
            mode="markers",
            marker={
                "color": "#16A34A",
                "size": 13,
                "symbol": "diamond",
                "line": {"color": "#FFFFFF", "width": 1.5},
            },
            customdata=annual_adjustments[corrected_positions],
            hovertemplate=(
                "ປີທີ່ໄດ້ຮັບຜົນຫຼັງແກ້ໄຂ (Affected year after treatment)<br>ປີ (Year): %{x}"
                "<br>ຍອດລວມທົ່ວປະເທດ (Nationwide total): %{y:,.0f}"
                "<br>ການປັບສຸດທິ (Net adjustment): %{customdata:+,.0f}<extra></extra>"
            ),
        )
    )
    correction_comparison_figure.update_layout(
        template="plotly_white",
        height=390,
        margin={"l": 62, "r": 24, "t": 18, "b": 52},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    correction_comparison_figure.update_xaxes(
        title_text="ປີ (Year)",
        tickmode="array",
        tickvals=HISTORICAL_YEARS,
        gridcolor=STREAMLIT_CHART_GRID,
    )
    correction_comparison_figure.update_yaxes(
        title_text="ປະຊາກອນ (Population)",
        tickformat=",.0f",
        gridcolor=STREAMLIT_CHART_GRID,
        rangemode="tozero",
    )
    example_years = ", ".join(str(year) for year in corrected_years)
    district_count = data["ouName"].astype(str).nunique()
    presentation_correction_example = (
        f"ຍອດລວມທົ່ວປະເທດ (Nationwide total) · {district_count:,} ເມືອງ (districts) · ທຸກອາຍຸ · ທັງສອງເພດ · "
        f"ປີທີ່ມີການແກ້ໄຂທີ່ຢືນຢັນ (Years containing confirmed corrections): {example_years} · "
        f"{len(presentation_corrections):,} ຄ່າສັງເກດທີ່ແກ້ໄຂແລ້ວ (corrected observations) · "
        f"ການປັບສຸດທິ (net adjustment) {presentation_corrections['Difference'].sum():+,.0f}"
    )
else:
    correction_comparison_figure.add_annotation(
        text="ບໍ່ມີການແກ້ໄຂຄ່າຜິດປົກກະຕິທີ່ຢືນຢັນ ສຳລັບຕົວກອງປັດຈຸບັນ (current filters)",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18, "color": "#5E7077"},
    )
    correction_comparison_figure.update_layout(
        template="plotly_white",
        height=300,
        margin={"l": 24, "r": 24, "t": 20, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
presentation_impact_rows = []
for model_name in MODEL_NAMES:
    for impact_year in sorted({2026, future_end}):
        position = impact_year - 2026
        original_value = float(nationwide_original_forecast.future[model_name][position])
        corrected_value = float(nationwide_corrected_forecast.future[model_name][position])
        difference = corrected_value - original_value
        presentation_impact_rows.append(
            {
                "model": PRESENTATION_MODEL_LABELS[model_name],
                "year": str(impact_year),
                "original": f"{original_value:,.0f}",
                "corrected": f"{corrected_value:,.0f}",
                "difference": f"{difference:+,.0f}",
                "change": (
                    f"{difference / original_value * 100:+.4f}%"
                    if original_value != 0
                    else "—"
                ),
            }
        )

live_results = {
    "source_name": source_name,
    "forecast_input": (
        "ຂໍ້ມູນທີ່ແກ້ໄຂແລ້ວ (Corrected data — recommended)"
        if forecast_input_option == CORRECTED_INPUT_OPTION
        else "ຂໍ້ມູນຕົ້ນສະບັບ (Original data)"
    ),
    "forecast_input_type": (
        "corrected"
        if forecast_input_option == CORRECTED_INPUT_OPTION
        else "original"
    ),
    "correction_threshold": f"{correction_threshold:g}",
    "correction_count": f"{len(presentation_corrections):,}",
    "correction_series": f"{presentation_corrected_series:,}",
    "correction_share": (
        f"{len(presentation_corrections) / presentation_cell_count * 100:.3f}%"
        if presentation_cell_count
        else "0.000%"
    ),
    "correction_loglinear": f"{presentation_method_counts.get('Two-sided log-linear interpolation', 0):,}",
    "correction_theilsen": f"{presentation_method_counts.get('Theil-Sen endpoint estimate', 0):,}",
    "correction_net": (
        f"{presentation_corrections['Difference'].sum():+,.0f}"
        if not presentation_corrections.empty
        else "0"
    ),
    "correction_audit_rows": presentation_audit_rows,
    "correction_impact_rows": presentation_impact_rows,
    "correction_example": presentation_correction_example,
    "outlier_evidence": presentation_outlier_evidence,
    "outlier_evidence_plotly": json.loads(outlier_evidence_figure.to_json()),
    "correction_comparison_plotly": json.loads(
        correction_comparison_figure.to_json()
    ),
    "baseline_summaries": {
        "original": {
            "rows": f"{len(data):,}",
            "population_2016": f"{data['2016'].sum():,.0f}",
            "population_2025": f"{data['2025'].sum():,.0f}",
            "source": "DHIS2, ກສທ",
        },
        "corrected": {
            "rows": f"{len(corrected_data):,}",
            "population_2016": f"{corrected_data['2016'].sum():,.0f}",
            "population_2025": f"{corrected_data['2025'].sum():,.0f}",
            "source": "DHIS2, ກສທ",
        },
    },
    "row_count": f"{len(filtered_data):,}",
    "population_2016": f"{result.actual[0]:,.0f}",
    "population_2025": f"{result.actual[-1]:,.0f}",
    "future_end": str(future_end),
    "forecast_period": f"2026–{future_end}",
    "selected_model": PRESENTATION_MODEL_LABELS.get(
        selected_model_option,
        "ທຸກແບບຈຳລອງ (All models)"
        if selected_model_option == ALL_MODELS_OPTION
        else selected_model_option,
    ),
    "validation_train_period": (
        f"{validation_result.train_years[0]}–{validation_result.train_years[-1]}"
    ),
    "validation_test_period": (
        f"{validation_result.test_years[0]}–{validation_result.test_years[-1]}"
    ),
    "validation_plotly": json.loads(
        interactive_validation_figure(
            validation_result,
            list(MODEL_NAMES),
            lao_presentation=True,
        ).to_json()
    ),
    "validation_rows": [
        {
            "model": PRESENTATION_MODEL_LABELS[name],
            "mae": f"{validation_result.metrics.loc[name, 'MAE']:,.2f}",
            "rmse": f"{validation_result.metrics.loc[name, 'RMSE']:,.2f}",
            "mape": f"{validation_result.metrics.loc[name, 'MAPE (%)']:.4f}%",
            "r2": f"{validation_result.metrics.loc[name, 'R²']:.4f}",
        }
        for name in MODEL_NAMES
    ],
    "future_years": future_years,
    "polynomial_table_rows": polynomial_table_rows,
    "future_table_models": [
        {
            "name": PRESENTATION_MODEL_LABELS[name],
            "key": name,
            "values": [
                f"{value:,.0f}" for value in np.rint(result.future[name]).clip(min=0)
            ],
        }
        for name in selected_models
    ],
}
presentation_best_model = validation_result.metrics["MAPE (%)"].idxmin()
live_results["best_model"] = PRESENTATION_MODEL_LABELS[presentation_best_model]
live_results["best_model_mape"] = (
    f"{validation_result.metrics.loc[presentation_best_model, 'MAPE (%)']:.4f}%"
)
live_results["best_model_2026"] = (
    f"{result.future[presentation_best_model][0]:,.0f}"
)
live_results["best_model_end_year"] = str(future_end)
live_results["best_model_end_value"] = (
    f"{result.future[presentation_best_model][-1]:,.0f}"
)
for model_name in MODEL_NAMES:
    model_key = model_name.lower()
    model_metrics = validation_result.metrics.loc[model_name]
    live_results.update(
        {
            f"{model_key}_mae": f"{model_metrics['MAE']:,.2f}",
            f"{model_key}_rmse": f"{model_metrics['RMSE']:,.2f}",
            f"{model_key}_mape": f"{model_metrics['MAPE (%)']:.4f}%",
            f"{model_key}_r2": f"{model_metrics['R²']:.4f}",
            f"{model_key}_accuracy": f"{model_metrics['Accuracy (%)']:.4f}%",
        }
    )
    for year in range(2026, 2031):
        if year <= future_end:
            value = result.future[model_name][year - 2026]
            live_results[f"{model_key}_{year}"] = f"{value:,.0f}"
        else:
            live_results[f"{model_key}_{year}"] = "—"
st.session_state["presentation_results"] = live_results

summary_column_count = 4 if selected_model_option == ALL_MODELS_OPTION else 4
summary_columns = st.columns(summary_column_count)
summary_columns[0].metric("ຈຳນວນຂໍ້ມູນ (Rows)", f"{len(filtered_data):,}")
summary_columns[1].metric("ປະຊາກອນ ປີ 2016", f"{result.actual[0]:,.0f}")
summary_columns[2].metric("ປະຊາກອນ ປີ 2025", f"{result.actual[-1]:,.0f}")

summary_columns[3].metric("ແຫຼ່ງຂໍ້ມູນ", f"DHIS2, ກສທ")

st.markdown(
            '<hr style="border: none; border-top: 2px solid #2196F3;">',
            unsafe_allow_html=True
        )
if selected_model_option == ALL_MODELS_OPTION:
    best_model = validation_result.metrics.loc[selected_models, "MAPE (%)"].idxmin()
    st.subheader("Time-based validation model comparison")
    st.caption(
        "Models are trained on 2016–2022 and evaluated on the unseen 2023–2025 "
        "holdout period. The final 2026+ forecasts are then retrained on all 2016–2025 data."
    )
    all_model_summary = validation_result.metrics.loc[list(MODEL_NAMES)].reset_index()
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
    # st.caption(f"Best fit by MAPE: {MODEL_LABELS[best_model]}")
    st.markdown(
        '<hr style="border: none; border-top: 2px solid #2196F3;">',
        unsafe_allow_html=True
    )
    # st.divider()
else:
    selected_mape = validation_result.metrics.loc[selected_model_option, "MAPE (%)"]
    summary_columns[3].metric("Validation MAPE", f"{selected_mape:.4f}%")
# st.caption(f"Source: {source_name} · {len(data):,} total source rows")


overview_tab, data_tab, pyramid_tab, export_tab, methodology_tab = st.tabs(
    [
        "Forecast Overview",
        "Source data & outliers",
        "Population pyramid",
        "Row-level exports",
        "Methodology & limitations",
    ]
)

with overview_tab:
    st.subheader("Time-based validation: 2016–2022 train → 2023–2025 test")
    st.caption(
        "This chart measures performance on later historical years that each model did "
        "not see during fitting. Model selection uses holdout MAPE, not training fit."
    )
    validation_chart = interactive_validation_figure(validation_result, selected_models)
    st.plotly_chart(
        validation_chart,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    validation_table = validation_result.metrics.loc[selected_models].reset_index()
    validation_table["Model"] = validation_table["Model"].map(MODEL_LABELS)
    st.dataframe(
        validation_table.style.format(
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
    )
    st.caption(
        "Accuracy is displayed as 100 − MAPE for continuity with the original project; "
        "it is not classification accuracy."
    )
    st.markdown(
        '<hr style="border: none; border-top: 2px solid #2196F3;">',
        unsafe_allow_html=True,
    )
    st.subheader("Final fit and forecast: retrained on all 2016–2025 observations")
    interactive_forecast_chart = interactive_forecast_figure(result, selected_models)
    live_results["forecast_plotly"] = json.loads(
        interactive_forecast_figure(
            result,
            selected_models,
            lao_presentation=True,
        ).to_json()
    )
    st.plotly_chart(
        interactive_forecast_chart,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    st.markdown(
            '<hr style="border: none; border-top: 2px solid #2196F3;">',
            unsafe_allow_html=True
        )
    # performance_title = (
    #     "Model accuracy comparison on 2016–2025 data"
    #     if selected_model_option == ALL_MODELS_OPTION
    #     else f"{MODEL_LABELS[selected_model_option]} accuracy on 2016–2025 data"
    # )
    # st.subheader(performance_title)
    # metrics = result.metrics.loc[selected_models].reset_index()
    # metrics["Model"] = metrics["Model"].map(MODEL_LABELS)
    # st.dataframe(
    #     metrics.style.format(
    #         {
    #             "MAE": "{:,.2f}",
    #             "RMSE": "{:,.2f}",
    #             "MAPE (%)": "{:.4f}",
    #             "R²": "{:.4f}",
    #             "Accuracy (%)": "{:.4f}",
    #         }
    #     ),
    #     hide_index=True,
    #     width="stretch",
    # )
    # st.markdown(
    #         '<hr style="border: none; border-top: 2px solid #2196F3;">',
    #         unsafe_allow_html=True
    #     )
    # st.caption(
    #     "Accuracy is shown as 100 − MAPE, matching the notebook. It is an "
    #     "in-sample descriptive score, not a validation-set accuracy measure."
    # )

    st.subheader("ສົມທຽບລະຫວ່າງຂໍ້ມູນຈິງ ແລະ ຜົນການຄາດຄະເນ (2016–2025), ຕາມແຕ່ລະແບບຈຳລອງ")
    interactive_actual_predicted_chart = interactive_actual_vs_predicted_figure(
        result, selected_models
    )
    live_results["actual_predicted_plotly"] = json.loads(
        interactive_actual_vs_predicted_figure(
            result,
            selected_models,
            lao_presentation=True,
        ).to_json()
    )
    st.plotly_chart(
        interactive_actual_predicted_chart,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    st.session_state["presentation_results"] = live_results

    st.markdown(
            '<hr style="border: none; border-top: 2px solid #2196F3;">',
            unsafe_allow_html=True
        )
    st.subheader("ສະຫຼຸບການຄາດຄະເນ ປະຊາກອນໂດຍລວມ ປີ 2026-{future_end}, ຕາມແຕ່ລະແບບຈຳລອງ".format(future_end=future_end))
    future_table = pd.DataFrame(
        {
            "ປີ": future_years,
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

    st.markdown(
        '<hr style="border: none; border-top: 2px solid #2196F3;">',
        unsafe_allow_html=True,
    )
    st.subheader("ອັດຕາການຂະຫຍາຍຕົວ ແລະ ອັດຕາສະເລ່ຍ")
    growth_rate_table, average_growth_rates = population_growth_rate_summary(
        result, selected_models
    )
    average_columns = st.columns(len(average_growth_rates))
    for column, (label, average_rate) in zip(
        average_columns, average_growth_rates.items()
    ):
        column.metric(f"{label} average", f"{average_rate:+.2f}%")

    growth_columns = [column for column in growth_rate_table if column != "Year"]
    st.dataframe(
        growth_rate_table.style.format(
            {column: "{:+.2f}%" for column in growth_columns}, na_rep="—"
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "ແຕ່ລະຄ່າແມ່ນອັດຕາການປ່ຽນແປງ (ເຕີບໂຕ) ທຽບກັບປີຕໍ່ປີ (Year-over-Year). ຕົວເລກຕົວຈິງແມ່ນຮອດ ປີ 2025; ຕົວເລກຄາດຄະເນຂອງແບບຈຳລອງແມ່ນສືບຕໍ່ໄປຈົນຮອດປີທີເລືອກໄວ້."
    )

with pyramid_tab:
    st.subheader("Population pyramid by district")
    st.caption(
        "Compare male and female populations in the age groups 0–4, 5–18, "
        "19–25, 26–40, 41–60, and 61–80+."
    )
    if "ouName" not in data.columns:
        st.error("The uploaded data must contain an ouName district column.")
    else:
        pyramid_districts = sorted(data["ouName"].dropna().astype(str).unique())
        district_column, year_column, age_display_column = st.columns(3)
        selected_pyramid_district = district_column.selectbox(
            "District",
            ["All", *pyramid_districts],
            key="population_pyramid_district",
            help="Choose All for the nationwide pyramid or select one district.",
        )
        selected_pyramid_year = year_column.selectbox(
            "Year",
            [*HISTORICAL_YEARS, *future_years],
            index=len(HISTORICAL_YEARS) - 1,
            key="population_pyramid_year",
        )
        selected_age_display = age_display_column.selectbox(
            "Age display",
            ["By age group", "By single age"],
            key="population_pyramid_age_display",
        )
        pyramid_data = data
        pyramid_year_label = str(selected_pyramid_year)
        if selected_pyramid_year > HISTORICAL_YEARS[-1]:
            pyramid_model = (
                presentation_best_model
                if selected_model_option == ALL_MODELS_OPTION
                else selected_model_option
            )
            pyramid_predictions = cached_row_forecast(data, future_end)
            prediction_column = (
                f"{selected_pyramid_year}{MODEL_EXPORT_SUFFIXES[pyramid_model]}"
            )
            pyramid_data = pyramid_predictions.copy()
            pyramid_data[str(selected_pyramid_year)] = pyramid_data[prediction_column]
            pyramid_year_label = (
                f"{selected_pyramid_year} forecast · {MODEL_LABELS[pyramid_model]}"
            )
            if selected_model_option == ALL_MODELS_OPTION:
                st.info(
                    "All models is selected, so this future pyramid uses the current "
                    f"lowest-MAPE model: {MODEL_LABELS[pyramid_model]}."
                )
            else:
                st.caption(
                    f"The {selected_pyramid_year} pyramid uses the selected "
                    f"{MODEL_LABELS[pyramid_model]} forecast."
                )
        try:
            pyramid_chart, pyramid_male_total, pyramid_female_total = (
                population_pyramid_figure(
                    pyramid_data,
                    None
                    if selected_pyramid_district == "All"
                    else selected_pyramid_district,
                    selected_pyramid_year,
                    pyramid_year_label,
                    selected_age_display,
                )
            )
        except ValueError as error:
            st.error(str(error))
        else:
            male_total_column, female_total_column, combined_total_column = st.columns(3)
            male_total_column.metric("Male population", f"{pyramid_male_total:,}")
            female_total_column.metric("Female population", f"{pyramid_female_total:,}")
            combined_total_column.metric(
                "Total population", f"{pyramid_male_total + pyramid_female_total:,}"
            )
            st.pyplot(pyramid_chart, width="stretch")
            plt.close(pyramid_chart)

with data_tab:
    st.subheader("Forecast source data")
    st.caption(
        f"Forecasts currently use **{forecast_input_label}**. The original source "
        "is preserved and remains available for comparison."
    )
    original_source_tab, corrected_source_tab = st.tabs(
        ["Original filtered data", "Corrected filtered data"]
    )
    with original_source_tab:
        st.dataframe(
            filtered_original_data, hide_index=True, width="stretch", height=420
        )
    with corrected_source_tab:
        st.dataframe(
            filtered_corrected_data, hide_index=True, width="stretch", height=420
        )

    st.subheader("Conservative outlier correction")
    st.caption(
        f"Automatic correction uses the selected robust-score threshold of {correction_threshold:g}. A value is "
        "changed only when a cohort-aware historical anomaly is independently confirmed "
        "by a district-share, sex-ratio, or adjacent-age check."
    )
    with st.expander("ຂັ້ນຕອນການແກ້ໄຂຄ່າຜິດປົກກະຕິ", expanded=True):
        st.markdown(
            """
1. Run the nationwide median/MAD outlier checks on the complete, unfiltered dataset.
2. Require two independent signals: a historical cohort-change flag and at least one
   demographic consistency or nationwide district-share flag.
3. For an internal year, replace the confirmed value with **two-sided log-linear
   interpolation** from the unchanged previous and following years:
   `exp((log(1 + previous) + log(1 + following)) / 2) - 1`.
4. For 2016 or 2025, use a **Theil–Sen endpoint estimate** calculated from the remaining
   years' median pairwise log-growth slopes.
5. Correct no more than two non-adjacent years in one district–age–sex series, clamp at
   zero, and round to a whole person. The original values are retained in the audit log.
            """
        )

    visible_corrections = filter_outlier_results(
        correction_result.corrections, filtered_original_data
    )
    corrected_series = (
        visible_corrections[["District", "Age group", "Sex"]]
        .drop_duplicates()
        .shape[0]
        if not visible_corrections.empty
        else 0
    )
    visible_cell_count = len(filtered_original_data) * len(HISTORICAL_COLUMNS)
    corrected_percent = (
        len(visible_corrections) / visible_cell_count * 100
        if visible_cell_count
        else 0.0
    )
    correction_metrics = st.columns(4)
    correction_metrics[0].metric("Corrected observations", f"{len(visible_corrections):,}")
    correction_metrics[1].metric("Affected series", f"{corrected_series:,}")
    correction_metrics[2].metric("Share of historical cells", f"{corrected_percent:.3f}%")
    correction_metrics[3].metric(
        "Net historical-cell adjustment",
        f"{visible_corrections['Difference'].sum():+,.0f}"
        if not visible_corrections.empty
        else "0",
    )

    if visible_corrections.empty:
        st.success("No observations in the current filters met the conservative correction rule.")
    else:
        st.bar_chart(
            visible_corrections.groupby("Year").size().rename("Corrections"),
            height=220,
        )
        st.dataframe(
            visible_corrections.style.format(
                {
                    "Original value": "{:,.0f}",
                    "Corrected value": "{:,.0f}",
                    "Difference": "{:+,.0f}",
                    "Change (%)": "{:+.3f}%",
                    "Temporal robust score": "{:+.2f}",
                    "Supporting robust score": "{:+.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
            height=420,
        )

    correction_download, corrected_data_download = st.columns(2)
    correction_download.download_button(
        "Download correction audit",
        visible_corrections.to_csv(index=False).encode("utf-8-sig"),
        file_name="population_outlier_correction_audit.csv",
        mime="text/csv",
        width="stretch",
    )
    corrected_data_download.download_button(
        "Download corrected source data",
        dataframe_to_csv_bytes(corrected_data),
        file_name="dataDistrict_Corrected.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("#### Forecast impact: original versus corrected data")
    st.caption(
        "This comparison uses the current district, age, and sex filters. It shows how "
        "the conservative corrections affect each of the existing three models."
    )
    impact_rows = []
    for model_name in MODEL_NAMES:
        for impact_year in sorted({2026, future_end}):
            position = impact_year - 2026
            original_value = float(original_forecast.future[model_name][position])
            corrected_value = float(corrected_forecast.future[model_name][position])
            difference = corrected_value - original_value
            impact_rows.append(
                {
                    "Model": MODEL_LABELS[model_name],
                    "Forecast year": impact_year,
                    "Original forecast": original_value,
                    "Corrected-data forecast": corrected_value,
                    "Difference": difference,
                    "Change (%)": (
                        difference / original_value * 100
                        if original_value != 0
                        else np.nan
                    ),
                }
            )
    forecast_impact = pd.DataFrame(impact_rows)
    st.dataframe(
        forecast_impact.style.format(
            {
                "Original forecast": "{:,.0f}",
                "Corrected-data forecast": "{:,.0f}",
                "Difference": "{:+,.0f}",
                "Change (%)": "{:+.4f}%",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Nationwide outlier review")
    st.caption(
        "Checks always use the complete nationwide dataset as their reference. "
        "Sidebar selections and the year selection below only filter the displayed flags. "
        "A flag indicates a record to review; it is not automatically an error."
    )
    control_column, year_column = st.columns([1, 3])
    control_column.metric(
        "Active robust-score threshold", f"{correction_threshold:g}"
    )
    selected_outlier_years = year_column.multiselect(
        "Years to display",
        HISTORICAL_YEARS,
        default=HISTORICAL_YEARS,
    )
    if not selected_outlier_years:
        st.info("Select at least one year to display outlier results.")
        visible_outlier_results = {
            "nationwide": pd.DataFrame(),
            "historical": pd.DataFrame(),
            "demographic": pd.DataFrame(),
        }
    else:
        outlier_results = correction_result.outliers
        nationwide_check, historical_check, demographic_check = st.tabs(
            [
                "Nationwide district comparison",
                "Historical change",
                "Demographic consistency",
            ]
        )
        with nationwide_check:
            visible_nationwide = show_outlier_check(
                "Nationwide district comparison",
                "For each age, sex, and year, compares that group's share of its "
                "district population across all districts using median-based robust scores.",
                outlier_results.nationwide,
                filtered_data,
                selected_outlier_years,
            )
        with historical_check:
            visible_historical = show_outlier_check(
                "Historical change",
                "Compares each age 1–79 population with the same birth cohort in the "
                "previous year, then evaluates that change against all districts.",
                outlier_results.historical,
                filtered_data,
                selected_outlier_years,
            )
        with demographic_check:
            visible_demographic = show_outlier_check(
                "Demographic consistency",
                "Combines unusual male-share checks with sharp local breaks between "
                "adjacent ages. The open-ended 80+ group is excluded from age-neighbor checks.",
                outlier_results.demographic,
                filtered_data,
                selected_outlier_years,
            )
        visible_outlier_results = {
            "nationwide": visible_nationwide,
            "historical": visible_historical,
            "demographic": visible_demographic,
        }

    live_results.update(
        {
            "outlier_threshold": f"{correction_threshold:g}",
            "outlier_years": (
                f"{min(selected_outlier_years)}–{max(selected_outlier_years)}"
                if selected_outlier_years
                else "None"
            ),
        }
    )
    for check_name, visible_results in visible_outlier_results.items():
        live_results[f"outlier_{check_name}"] = f"{len(visible_results):,}"
        year_counts = (
            visible_results.groupby("Year").size().to_dict()
            if not visible_results.empty
            else {}
        )
        for year in HISTORICAL_YEARS:
            live_results[f"outlier_{check_name}_{year}"] = (
                f"{int(year_counts.get(year, 0)):,}"
                if year in selected_outlier_years
                else "—"
            )
    st.session_state["presentation_results"] = live_results

with export_tab:
    st.write(
        "ຜົນການຄາດຄະເນປະຊາກອນ ສາມາດດາວໂຫຼດໄດ້ໃນແບບ CSV ຫຼື HTML. ປະກອບມີຂໍ້ມູນການຄາດຄະເນທັງໝົດ ຕາມແຕ່ລະແບບ Linear Regression, Polynomial Regression, ແລະ Exponential Growth."
    )
    predicted = row_predictions
    html_table = polynomial_html_table(predicted, future_end)
    prediction_suffixes = tuple(MODEL_EXPORT_SUFFIXES[name] for name in MODEL_NAMES)
    selected_suffixes = tuple(MODEL_EXPORT_SUFFIXES[name] for name in selected_models)
    preview_columns = [
        column
        for column in predicted.columns
        if not column.endswith(prediction_suffixes)
        or column.endswith(selected_suffixes)
    ]
    # st.caption(
    #     "The preview follows the selected model option. The all-model CSV continues to "
    #     "include Linear Regression, Polynomial Regression, and Exponential Growth."
    # )
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

with methodology_tab:
    st.info(
        "Course purpose: help every student understand the end-to-end AI workflow "
        "and apply it to a practical task that addresses a real-world problem."
    )
    st.subheader("CRISP-DM project workflow")
    st.markdown(
        """
1. **Business understanding** — provide clearer population targets for health-service planning.
2. **Data understanding** — review nationwide DHIS2 population data by district, age, sex, and year.
3. **Data preparation** — validate the schema and treat only independently confirmed outliers.
4. **Modeling** — compare Linear, degree-2 Polynomial, and Exponential Growth models.
5. **Evaluation** — train on 2016–2022 and test on unseen 2023–2025 observations.
6. **Deployment** — retrain on the full 2016–2025 history and deliver forecasts through Streamlit and downloadable exports.
        """
    )

    st.subheader("Application architecture")
    st.code(
        "DHIS2 / CSV\n"
        "      ↓\n"
        "Schema validation and nationwide outlier review\n"
        "      ↓\n"
        "Original or conservatively corrected baseline\n"
        "      ↓\n"
        "Time-based validation and model comparison\n"
        "      ↓\n"
        "Full-history retraining and 2026+ forecast\n"
        "      ↓\n"
        "Streamlit charts, audit tables, CSV and HTML exports",
        language="text",
    )

    st.subheader("Limitations and responsible use")
    st.markdown(
        """
- Only ten annual observations are available, so long-horizon forecasts are uncertain.
- The models extrapolate historical trends and do not directly model migration, policy changes, disasters, epidemics, fertility, or mortality.
- Polynomial extrapolation can become unrealistic as the forecast horizon grows.
- Automated outlier treatment can hide a genuine demographic event; original values and the correction audit are therefore retained.
- Prediction intervals are not yet included.
- Results are decision-support estimates and should not replace official projections or demographic expert review.
        """
    )

    st.subheader("Method and software references")
    st.markdown(
        """
- Chapman et al. (2000), *CRISP-DM 1.0: Step-by-step data mining guide*.
- Theil (1950) and Sen (1968), robust median-slope estimation.
- scikit-learn documentation: `LinearRegression`, regression metrics, and model evaluation.
- Plotly and Streamlit documentation for interactive visualization and application delivery.
- Population source: DHIS2, Ministry of Health dataset used by this application.
        """
    )

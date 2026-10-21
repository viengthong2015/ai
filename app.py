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
import streamlit as st

from outlier_detection import OutlierResults, detect_population_outliers
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
    validate_population_data,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "data" / "dataDistrict.csv"
PRESENTATION_FILE = APP_DIR / "index.html"
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
    html = html.replace("</head>", f"{live_results_script}</head>")
    return html.replace("</body>", f"<script>{javascript}</script></body>")


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
        2, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [2, 1.5]}
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
    forecast_axis.set_title("ສົມທຽບຄວາມຊັດເຈນຂອງ ຜົນການຄາດຄະເນປະຊາກອນ")
    forecast_axis.set_ylabel("ປະຊາກອນ (ຄົນ)")
    forecast_axis.ticklabel_format(axis="y", style="plain")
    forecast_axis.legend(ncol=2)
    forecast_axis.grid(alpha=0.25)

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
            edgecolor="white",
            linewidth=0.7,
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
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )
    for divider_position in (year_positions[:-1] + year_positions[1:]) / 2:
        residual_axis.axvline(
            divider_position,
            color="#ced4da",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )
    residual_axis.axhline(0, color="black", linewidth=1, zorder=4)
    
    residual_axis.set_title("ຄ່າຜິດດ່ຽງຂອງການຄາດຄະເນ (ຕົວຈິງ − ຜົນການພຄາດຄະເນ)")
    residual_axis.set_xlabel("ປີ")
    residual_axis.set_ylabel("ປະຊາກອນ (ຄົນ)")
    residual_axis.set_xticks(year_positions, HISTORICAL_YEARS)
    residual_axis.legend(ncol=3)
    residual_axis.grid(axis="y", alpha=0.25, zorder=0)
    residual_axis.set_axisbelow(True)
    residual_axis.margins(y=0.22)
    fig.tight_layout()
    return fig


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
        # border-radius: 10px 10px 0px 0px;
        background: linear-gradient(135deg, #eff8ff 0%, #e8f4ff 100%);
        color: #0f4c81;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.1);
        # width: 100%;
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

live_results = {
    "source_name": source_name,
    "row_count": f"{len(filtered_data):,}",
    "population_2016": f"{result.actual[0]:,.0f}",
    "population_2025": f"{result.actual[-1]:,.0f}",
    "future_end": str(future_end),
    "forecast_period": f"2026–{future_end}",
    "selected_model": MODEL_LABELS.get(selected_model_option, selected_model_option),
    "future_years": future_years,
    "polynomial_table_rows": polynomial_table_rows,
    "future_table_models": [
        {
            "name": MODEL_LABELS[name],
            "key": name,
            "values": [
                f"{value:,.0f}" for value in np.rint(result.future[name]).clip(min=0)
            ],
        }
        for name in selected_models
    ],
}
presentation_best_model = result.metrics["MAPE (%)"].idxmin()
live_results["best_model"] = MODEL_LABELS[presentation_best_model]
live_results["best_model_mape"] = (
    f"{result.metrics.loc[presentation_best_model, 'MAPE (%)']:.4f}%"
)
for model_name in MODEL_NAMES:
    model_key = model_name.lower()
    model_metrics = result.metrics.loc[model_name]
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
    best_model = result.metrics.loc[selected_models, "MAPE (%)"].idxmin()
    st.subheader("ຜົນສະຫຼຸບຂອງແຕ່ລະແບບຈຳລອງ (models)")
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
    # st.caption(f"Best fit by MAPE: {MODEL_LABELS[best_model]}")
    st.markdown(
        '<hr style="border: none; border-top: 2px solid #2196F3;">',
        unsafe_allow_html=True
    )
    # st.divider()
else:
    selected_mape = result.metrics.loc[selected_model_option, "MAPE (%)"]
    summary_columns[3].metric("Selected model MAPE", f"{selected_mape:.4f}%")
# st.caption(f"Source: {source_name} · {len(data):,} total source rows")


overview_tab, pyramid_tab, data_tab, export_tab = st.tabs(
    [
        "Forecast Overview",
        "Population pyramid",
        "Source data & outliers",
        "Row-level exports",
    ]
)

with overview_tab:
    forecast_chart = forecast_figure(result, selected_models)
    live_results["forecast_chart_src"] = figure_to_data_uri(forecast_chart)
    st.pyplot(forecast_chart, width="stretch")
    plt.close(forecast_chart)
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
    actual_predicted_chart = actual_vs_predicted_figure(result, selected_models)
    live_results["actual_predicted_chart_src"] = figure_to_data_uri(
        actual_predicted_chart
    )
    st.pyplot(actual_predicted_chart, width="stretch")
    plt.close(actual_predicted_chart)
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
        visible_outlier_results = {
            "nationwide": pd.DataFrame(),
            "historical": pd.DataFrame(),
            "demographic": pd.DataFrame(),
        }
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
            "outlier_threshold": f"{threshold:g}",
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

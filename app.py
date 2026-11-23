"""Interactive Streamlit implementation of estimatePop.ipynb."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import mimetypes
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from outlier_detection import (
    OutlierCorrectionResult,
    OutlierResults,
    correct_population_outliers,
)
from i18n import DEFAULT_LANGUAGE, LANGUAGES, block, translate
import population_map
from population_map import (
    DENSITY_COLORSCALE,
    district_totals,
    load_boundaries,
    population_map_figure,
    province_totals,
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
BOUNDARY_FILE = APP_DIR / "data" / "orgunits.geojson"
MAP_LEVEL_OPTIONS = {
    "Province (level 2)": "provinces",
    "District (level 3)": "districts",
}
PRESENTATION_FILE = APP_DIR / "index.html"
PLOTLY_JAVASCRIPT = get_plotlyjs()
# UI style: "default" (light atlas design) or "crystal" (dark emerald glass).
# The "Switch UI Style" button flips it; Streamlit reruns the script without a
# page reload, so every selection in st.session_state is kept.
UI_MODE = st.session_state.get("ui_mode", "default")
CRYSTAL_MODE = UI_MODE == "crystal"

# Atlas palette, shared with the presentation (image/style.css). Mekong, Madder
# and Rice were checked for colour-blind separation as a set.
ATLAS_PAPER = "#F6F7F3"
ATLAS_INK = "#14302A"
ATLAS_SLATE = "#5C6470"
ATLAS_RULE = "#D3DED5"
ATLAS_MEKONG_SOFT = "#DCE8F3"
ATLAS_GOLD = "#B8902F"
ATLAS_GOLD_SOFT = "#F4ECD8"
if CRYSTAL_MODE:
    # Data colours brightened for the dark UI; SLIDE_COLOURS maps them back to
    # the original atlas colours for the light presentation slides.
    ATLAS_MEKONG = "#4F9DEB"
    ATLAS_FOREST = "#2BD98A"
    ATLAS_FOREST_SOFT = "rgba(43, 217, 138, 0.28)"  # best-value highlight in tables
    ATLAS_MADDER = "#E0607A"
    ATLAS_RICE = "#9CCB5B"
else:
    ATLAS_MEKONG = "#1D6AAE"
    ATLAS_FOREST = "#1E7B4F"
    ATLAS_FOREST_SOFT = "#DDEFE3"
    ATLAS_MADDER = "#A63D4F"
    ATLAS_RICE = "#5C8A2E"
MODEL_STYLES = {
    "Linear": {"color": ATLAS_MEKONG, "linestyle": "--"},
    "Polynomial": {"color": ATLAS_MADDER, "linestyle": "-."},
    "Exponential": {"color": ATLAS_RICE, "linestyle": ":"},
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
if CRYSTAL_MODE:
    # Dark chart palette. Charts that also go to the (light) presentation slides
    # are converted back with slide_json(), so every colour here must be unique
    # to the charts.
    CHART_TEXT = "#E3F1EA"
    CHART_MUTED = "#8FB5A5"
    CHART_GRID = "rgba(160, 230, 190, 0.10)"
    CHART_SURFACE = "#0B2A21"
    CHART_HOVER = "rgba(8, 36, 28, 0.96)"
    CRYSTAL_GREEN = "#2BD98A"
    CRYSTAL_CYAN = "#19C3D9"
    STREAMLIT_CHART_FORECAST_BAND = "rgba(184, 144, 47, 0.16)"
    FORECAST_LABEL_COLOUR = "#E2B954"
    SEX_RGB = {"Male": "79,157,235", "Female": "224,96,122"}
    SLIDE_COLOURS = {
        CHART_TEXT: ATLAS_INK,
        CHART_MUTED: ATLAS_SLATE,
        CHART_GRID: "#E4E7E0",
        CHART_SURFACE: "#FFFFFF",
        CHART_HOVER: "#FFFFFF",
        STREAMLIT_CHART_FORECAST_BAND: ATLAS_GOLD_SOFT,
        "#4F9DEB": "#1D6AAE",
        "#2BD98A": "#1E7B4F",
        "#E0607A": "#A63D4F",
        "#9CCB5B": "#5C8A2E",
    }
else:
    CHART_TEXT = ATLAS_INK
    CHART_MUTED = ATLAS_SLATE
    CHART_GRID = "#E4E7E0"
    CHART_SURFACE = "#FFFFFF"
    CHART_HOVER = "#FFFFFF"
    CRYSTAL_GREEN = ATLAS_INK  # actual line in the trajectory chart
    CRYSTAL_CYAN = ATLAS_GOLD  # forecast line in the trajectory chart
    STREAMLIT_CHART_FORECAST_BAND = ATLAS_GOLD_SOFT
    FORECAST_LABEL_COLOUR = "#7A5F1C"
    SEX_RGB = {"Male": "29,106,174", "Female": "166,61,79"}
    SLIDE_COLOURS = {}
STREAMLIT_CHART_BLUE = CHART_TEXT  # actual (observed) values
STREAMLIT_CHART_TEXT = CHART_TEXT
STREAMLIT_CHART_MUTED = CHART_MUTED
STREAMLIT_CHART_GRID = CHART_GRID
LAO_FONT_FAMILIES = ["Noto Sans Lao", "Phetsarath OT"]
LAO_FONT_STACK = "'Noto Sans Lao', 'Phetsarath OT', sans-serif"
CRYSTAL_FONT_STACK = "Inter, 'Noto Sans Lao', 'Phetsarath OT', sans-serif"
CRYSTAL_MONO_STACK = "'JetBrains Mono', 'Noto Sans Lao', monospace"

# Plotly charts (Streamlit and the embedded presentation): Lao font everywhere.
pio.templates["lao_font"] = go.layout.Template(
    layout={"font": {"family": LAO_FONT_STACK}}
)
pio.templates["lao_crystal"] = go.layout.Template(
    layout={
        "font": {"color": "#E3F1EA", "family": CRYSTAL_FONT_STACK},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "hoverlabel": {
            "bgcolor": "rgba(8, 36, 28, 0.96)",
            "bordercolor": "rgba(43, 217, 138, 0.45)",
            "font": {"color": "#E3F1EA", "family": CRYSTAL_MONO_STACK},
        },
        "xaxis": {
            "gridcolor": "rgba(160, 230, 190, 0.10)",
            "linecolor": "rgba(160, 230, 190, 0.10)",
            "zerolinecolor": "rgba(160, 230, 190, 0.10)",
            "tickfont": {"color": "#8FB5A5", "family": CRYSTAL_MONO_STACK},
        },
        "yaxis": {
            "gridcolor": "rgba(160, 230, 190, 0.10)",
            "linecolor": "rgba(160, 230, 190, 0.10)",
            "zerolinecolor": "rgba(160, 230, 190, 0.10)",
            "tickfont": {"color": "#8FB5A5", "family": CRYSTAL_MONO_STACK},
        },
        "legend": {"font": {"color": "#E3F1EA"}},
        "title": {"font": {"color": "#F8FAFC"}},
        "annotationdefaults": {"font": {"color": "#A7F3D0"}},
        "coloraxis": {"colorbar": {"tickfont": {"color": "#A7F3D0"}, "title": {"font": {"color": "#E3F1EA"}}}},
    }
)
for _axis in ("xaxis", "yaxis"):
    pio.templates["lao_crystal"].layout[_axis].title.font.color = "#A7F3D0"
# Crystal charts carry their own dark template; Streamlit's chart theme (light
# base theme) would override its text colours, so it is switched off there.
CHART_THEME = None if CRYSTAL_MODE else "streamlit"
PLOTLY_TEMPLATE = (
    "plotly_white+lao_font+lao_crystal" if CRYSTAL_MODE else "plotly_white+lao_font"
)


def slide_json(figure: go.Figure) -> dict:
    """Plotly JSON for the light presentation slides (Crystal colours swapped back)."""
    payload = figure.to_json()
    for dark, light in SLIDE_COLOURS.items():
        payload = payload.replace(f'"{dark}"', f'"{light}"')
    if CRYSTAL_MODE:
        # The slides use the light template and fonts.
        payload = payload.replace(CRYSTAL_MONO_STACK, LAO_FONT_STACK).replace(
            CRYSTAL_FONT_STACK, LAO_FONT_STACK
        )
    return json.loads(payload)

# Matplotlib charts: use whichever Lao fonts are installed, with DejaVu Sans
# as the fallback for Latin characters.
_installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [
    *(name for name in LAO_FONT_FAMILIES if name in _installed_fonts),
    "DejaVu Sans",
]

st.set_page_config(
    page_title="Laos Population Forecast",
    page_icon="📈",
    layout="wide",
)


def current_language() -> str:
    """The demo language chosen in the sidebar ("lo" or "en")."""
    return st.session_state.get("demo_language") or DEFAULT_LANGUAGE


def tr(text: str, **values: object) -> str:
    """Demo text in the selected language; translations live in i18n.py."""
    return translate(text, current_language(), **values)


def tr_table(df: pd.DataFrame, value_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Translate a table's headers, and the fixed text in ``value_columns``."""
    table = df.copy()
    for column in value_columns:
        if column in table:
            table[column] = table[column].map(
                lambda value: tr(value) if isinstance(value, str) else value
            )
    return table.rename(columns=lambda column: tr(column) if isinstance(column, str) else column)


def tr_formats(formats: dict[str, str]) -> dict[str, str]:
    """Translate the column names of a Styler format mapping."""
    return {tr(column): pattern for column, pattern in formats.items()}


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


@st.cache_data(show_spinner="Loading map boundaries…")
def cached_boundaries(path: str, modified_time: float, loader_version: float) -> dict:
    # Both arguments only invalidate the cache: when the GeoJSON changes, or when
    # population_map.py changes (Streamlit does not see edits to that module).
    del modified_time, loader_version
    return load_boundaries(path)


def map_boundaries() -> dict:
    return cached_boundaries(
        str(BOUNDARY_FILE),
        BOUNDARY_FILE.stat().st_mtime,
        Path(population_map.__file__).stat().st_mtime,
    )


@st.cache_data(show_spinner="Calculating row-level forecasts…")
def cached_row_forecast(
    df: pd.DataFrame, future_end: int, round_and_clip: bool = True
) -> pd.DataFrame:
    return forecast_each_row(df, future_end, round_and_clip=round_and_clip)


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


def filter_data(df: pd.DataFrame, container=None) -> pd.DataFrame:
    container = container or st.sidebar
    filtered = df
    filter_specs = (
        ("ouName", "Districts", "All districts"),
        ("sexName", "Sex", "All sexes"),
        ("cocName", "Age groups", "All age groups"),
    )
    for column, label, placeholder in filter_specs:
        if column not in df.columns:
            continue
        options = sorted(df[column].dropna().astype(str).unique().tolist())
        selected = container.multiselect(
            tr(label),
            options,
            placeholder=tr(placeholder),
            key=f"filter_{column}",
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
        2025, color=ATLAS_GOLD, linestyle="-", linewidth=1.6, alpha=1
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
            color="#ECEEE8",
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
            "ສົມທຽບຄວາມຊັດເຈນຂອງ ຜົນການຄາດຄະເນປະຊາກອນ"
            if lao_presentation
            else tr("Accuracy of the population forecasts"),
            "ຄ່າຜິດດ່ຽງຂອງການຄາດຄະເນ (ຕົວຈິງ − ຜົນການຄາດຄະເນ)"
            if lao_presentation
            else tr("Forecast residuals (actual − predicted)"),
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=HISTORICAL_YEARS,
            y=result.actual,
            name=(
                "ປະຊາກອນຕົວຈິງ (Actual population)"
                if lao_presentation
                else tr("Actual population")
            ),
            mode="lines+markers",
            line={"color": STREAMLIT_CHART_BLUE, "width": 3},
            marker={
                "color": CHART_SURFACE,
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
                    f"<b>{tr('Actual population')}</b><br>{tr('Year')}: %{{x}}<br>"
                    f"{tr('Population')}: %{{y:,.0f}}<extra></extra>"
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
                    else f"<b>{label}</b><br>{tr('Year')}: %{{x}}<br>{tr('Population')}: %{{y:,.0f}}<extra></extra>"
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
                    else tr("{model} residual", model=label)
                ),
                marker={
                    "color": MODEL_STYLES[name]["color"],
                    "line": {"color": CHART_SURFACE, "width": 1},
                },
                text=residual_labels,
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>ປີ (Year): %{{x}}<br>"
                    "ຄ່າຄົງເຫຼືອ (Residual: Actual − Predicted): %{y:+,.0f}<extra></extra>"
                    if lao_presentation
                    else f"<b>{label}</b><br>{tr('Year')}: %{{x}}<br>{tr('Residual (actual − predicted)')}: %{{y:+,.0f}}<extra></extra>"
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
        line={"color": ATLAS_GOLD, "width": 1.6},
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
        title_text="ປີ" if lao_presentation else tr("Year"),
        tickmode="array",
        tickvals=HISTORICAL_YEARS,
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="ປະຊາກອນ (ຄົນ)" if lao_presentation else tr("Population (people)"),
        tickformat=",.0f",
    )
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
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
        hoverlabel={"bgcolor": CHART_HOVER, "font_color": STREAMLIT_CHART_TEXT},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
                    label_color = ATLAS_INK
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
                else CHART_TEXT
            )

        figure.add_trace(
            go.Bar(
                x=HISTORICAL_YEARS,
                y=actual_values,
                name="ຄ່າຈິງ (Actual)" if lao_presentation else tr("Actual"),
                legendgroup="actual",
                showlegend=row_number == 1,
                marker={
                    "color": STREAMLIT_CHART_BLUE,
                    "line": {"color": CHART_SURFACE, "width": 1},
                },
                hovertemplate=(
                    "<b>ຄ່າຈິງ (Actual)</b><br>ປີ (Year): %{x}<br>"
                    "ປະຊາກອນ (Population): %{y:,.0f}<extra></extra>"
                    if lao_presentation
                    else f"<b>{tr('Actual')}</b><br>{tr('Year')}: %{{x}}<br>{tr('Population')}: %{{y:,.0f}}<extra></extra>"
                ),
            ),
            row=row_number,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=HISTORICAL_YEARS,
                y=predicted_values,
                name="ຄ່າຄາດຄະເນ (Predicted)" if lao_presentation else tr("Predicted"),
                legendgroup="predicted",
                showlegend=row_number == 1,
                marker={
                    "color": MODEL_STYLES[name]["color"],
                    "line": {"color": CHART_SURFACE, "width": 1},
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
                    else (
                        "<b>" + tr("{model} prediction", model=label) + "</b><br>"
                        f"{tr('Year')}: %{{x}}<br>{tr('Population')}: %{{y:,.0f}}<br>"
                        f"{tr('Difference')}: %{{customdata[0]}}<extra></extra>"
                    )
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
                else tr("Population")
            ),
            tickformat=",.0f",
            gridcolor=STREAMLIT_CHART_GRID,
            rangemode="tozero",
            row=row_number,
            col=1,
        )

    figure.update_xaxes(
        title_text="ປີ (Year)" if lao_presentation else tr("Year"),
        row=panel_count,
        col=1,
    )
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": (
                "ປະຊາກອນຈິງ ທຽບກັບ ຄາດຄະເນ (Actual vs predicted population — historical fit)"
                if lao_presentation
                else tr("Actual vs predicted population (historical fit)")
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
        hoverlabel={"bgcolor": CHART_HOVER, "font_color": STREAMLIT_CHART_TEXT},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
                else tr("Actual population")
            ),
            mode="lines+markers",
            line={"color": STREAMLIT_CHART_BLUE, "width": 4},
            marker={"size": 10, "color": CHART_SURFACE, "line": {"color": STREAMLIT_CHART_BLUE, "width": 2.5}},
            hovertemplate=(
                "ຄ່າຈິງ (Actual)<br>ປີ (Year): %{x}<br>ປະຊາກອນ (Population): %{y:,.0f}<extra></extra>"
                if lao_presentation
                else f"{tr('Actual')}<br>{tr('Year')}: %{{x}}<br>{tr('Population')}: %{{y:,.0f}}<extra></extra>"
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
                    else f"{label}<br>{tr('Year')}: %{{x}}<br>{tr('Prediction')}: %{{y:,.0f}}<br>{tr('Error')}: %{{customdata:+.4f}}%<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(
        title_text=(
            "ປີຂອງຊຸດທົດສອບ (Holdout year)"
            if lao_presentation
            else tr("Holdout year")
        ),
        tickmode="array",
        tickvals=validation.test_years,
        gridcolor=STREAMLIT_CHART_GRID,
    )
    figure.update_yaxes(
        title_text=(
            "ປະຊາກອນ (Population)"
            if lao_presentation
            else tr("Population")
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
) -> tuple[go.Figure, int, int]:
    """Build an interactive population pyramid for the requested district and age bands."""
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
    total = max(float(male_values.sum() + female_values.sum()), 1.0)
    population_limit = max(float(male_values.max()), float(female_values.max()), 1.0)
    figure = go.Figure()
    for values, sign, label, color in (
        (male_values, -1, tr("Male"), ATLAS_MEKONG),
        (female_values, 1, tr("Female"), ATLAS_MADDER),
    ):
        figure.add_trace(
            go.Bar(
                y=list(profile.index),
                x=sign * values,
                customdata=np.column_stack([values, values / total * 100]),
                name=label,
                orientation="h",
                marker={"color": color, "opacity": 0.82, "line": {"color": CHART_SURFACE, "width": 1}},
                hovertemplate=(
                    f"<b>%{{y}}</b> · {label}<br>{tr('Population')}: %{{customdata[0]:,.0f}}"
                    f"<br>{tr('Share')}: %{{customdata[1]:.2f}}%<extra></extra>"
                ),
            )
        )
    tick_step = 10 ** max(int(np.floor(np.log10(population_limit))), 0)
    if population_limit / tick_step < 3:
        tick_step /= 2
    ticks = np.arange(-population_limit, population_limit + tick_step, tick_step)
    ticks = np.round(ticks / tick_step) * tick_step
    location_label = district if district is not None else tr("All districts")
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": tr(
                "Population pyramid — {location} ({year})",
                location=location_label,
                year=year_label or year,
            ),
            "x": 0,
            "xanchor": "left",
        },
        barmode="relative",
        bargap=0.08,
        height=max(460, 22 * len(profile.index) + 160),
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.0, "yanchor": "bottom"},
        hoverlabel={"bgcolor": CHART_HOVER, "font_color": CHART_TEXT},
    )
    figure.update_xaxes(
        title_text=tr("Population (people)"),
        range=[-population_limit * 1.12, population_limit * 1.12],
        tickvals=ticks,
        ticktext=[f"{abs(value):,.0f}" for value in ticks],
        gridcolor="rgba(20,48,42,0.08)",
        zeroline=True,
        zerolinecolor=CHART_TEXT,
    )
    figure.update_yaxes(title_text=tr(y_axis_label), automargin=True)
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
    st.markdown(f"#### {tr(title)}")
    st.caption(tr(description))
    visible = filter_outlier_results(results, visible_data)
    visible = visible[visible["Year"].isin(years)].reset_index(drop=True)
    st.metric(tr("Flagged observations"), f"{len(visible):,}")
    if visible.empty:
        st.success(tr("No observations exceed the selected robust-score threshold."))
        return visible
    st.bar_chart(visible.groupby("Year").size().rename(tr("Flags")), height=220)
    st.dataframe(
        tr_table(visible, ("Check", "Reason", "Unit")).style.format(
            tr_formats(
                {
                    "Population": "{:,.0f}",
                    "Observed": "{:,.3f}",
                    "Expected": "{:,.3f}",
                    "Robust score": "{:+.2f}",
                }
            )
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )
    st.download_button(
        tr("Download flags: {title}", title=tr(title)),
        visible.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{title.lower().replace(' ', '_')}_flags.csv",
        mime="text/csv",
        key=f"download-{title}",
    )
    return visible


st.markdown(
    """
    <style>
    /* Atlas styling. Colours come from .streamlit/config.toml; this block only
       adds what the theme cannot express. */
    :root {
        --atlas-ink: #14302A;
        --atlas-slate: #5C6470;
        --atlas-rule: #D3DED5;
        --atlas-forest: #1E7B4F;
        --atlas-gradient: linear-gradient(120deg, #0B5D3B 0%, #17864F 55%, #3DBA74 100%);
        --atlas-gold: #B8902F;
        --atlas-gold-soft: #F4ECD8;
    }

    /* Headings: Noto Sans Lao at a condensed width, so titles read as an atlas
       heading rather than body text set large. */
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3,
    .atlas-masthead h1 {
        font-stretch: 80%;
        letter-spacing: -0.005em;
    }

    [data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums lining-nums;
    }

    /* Busy state: a flat ink label while the forecast recalculates. */
    .stApp[data-test-script-state="running" i]::before {
        content: "ກຳລັງຄຳນວນ…";
        position: fixed;
        top: 1rem;
        left: 50%;
        z-index: 999999;
        transform: translateX(-50%);
        padding: 0.6rem 1.2rem;
        border-radius: 0.3rem;
        background: var(--atlas-ink);
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 600;
        pointer-events: none;
    }

    /* Tabs: larger labels on Streamlit's own underline tabs. */
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {
        font-size: 1.05rem;
        font-weight: 600;
    }

    /* Masthead */
    .atlas-masthead {
        margin: 0.25rem 0 1.5rem;
    }

    .atlas-masthead h1 {
        margin: 0;
        padding: 0;
        width: fit-content;
        color: var(--atlas-ink);
        background: var(--atlas-gradient);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(2.3rem, 4.6vw, 3.6rem);
        font-weight: 800;
        line-height: 1.1;
    }

    .atlas-masthead p {
        max-width: 68ch;
        margin: 0.5rem 0 0;
        color: var(--atlas-slate);
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* Time ribbon: train → test → forecast, one cell per year. */
    .atlas-ribbon {
        display: grid;
        grid-template-columns: var(--ribbon-columns);
        gap: 0 0.35rem;
        margin: 0 0 1.75rem;
        font-variant-numeric: tabular-nums lining-nums;
    }

    .atlas-ribbon__span {
        color: var(--atlas-slate);
        font-size: 0.95rem;
        font-weight: 600;
    }

    .atlas-ribbon__band {
        display: flex;
        gap: 2px;
        height: 14px;
        margin: 0.45rem 0 0.7rem;
    }

    .atlas-ribbon__band i {
        flex: 1;
        background: var(--atlas-gradient);
    }

    .atlas-ribbon__seg--test .atlas-ribbon__band i {
        background: repeating-linear-gradient(
            135deg,
            var(--atlas-forest) 0 2px,
            transparent 2px 5px
        );
        box-shadow: inset 0 0 0 1.5px var(--atlas-forest);
    }

    /* The gold thread: forecast years are woven, not solid. */
    .atlas-ribbon__seg--forecast .atlas-ribbon__band i {
        background:
            repeating-linear-gradient(90deg, var(--atlas-gold) 0 3px, transparent 3px 6px),
            var(--atlas-gold-soft);
        box-shadow: inset 0 0 0 1.5px var(--atlas-gold);
    }

    .atlas-ribbon__label {
        color: var(--atlas-ink);
        font-size: 1rem;
        font-weight: 700;
    }

    .atlas-ribbon__value {
        margin-top: 0.15rem;
        color: var(--atlas-ink);
        font-size: clamp(1.5rem, 2.4vw, 2.1rem);
        font-weight: 750;
        font-stretch: 85%;
        line-height: 1.15;
    }

    .atlas-ribbon__note {
        color: var(--atlas-slate);
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .atlas-ribbon__seg--forecast .atlas-ribbon__span {
        color: #7A5F1C;
    }

    .atlas-meta {
        margin: -0.75rem 0 1.5rem;
        color: var(--atlas-slate);
        font-size: 0.95rem;
    }

    @media (max-width: 760px) {
        .atlas-ribbon {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    /* ---------- Crystal layer: mint glass, glowing edges, dark-green step bar ---------- */
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(61, 186, 116, 0.22), transparent 36%),
            radial-gradient(circle at 92% 6%, rgba(61, 186, 116, 0.16), transparent 32%),
            radial-gradient(circle at 50% 100%, rgba(30, 123, 79, 0.10), transparent 45%),
            linear-gradient(180deg, #EEF8F1 0%, #E4F3E9 100%);
        background-attachment: fixed;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.7), rgba(220, 241, 228, 0.6));
        backdrop-filter: blur(18px) saturate(140%);
        border-right: 1px solid rgba(61, 186, 116, 0.25);
    }

    /* Cards: mint glass with a faint green edge. No backdrop-filter here: it
       would trap a chart's fullscreen view inside the card. */
    [class*="st-key-glass-"],
    [data-testid="stExpander"] details {
        background: linear-gradient(160deg, rgba(255, 255, 255, 0.9), rgba(226, 244, 233, 0.74)) !important;
        border: 1px solid rgba(61, 186, 116, 0.32) !important;
        border-radius: 16px !important;
        box-shadow:
            0 10px 28px rgba(30, 123, 79, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.95);
    }

    [class*="st-key-glass-"] {
        padding: 1rem 1.1rem 0.6rem !important;
        transition: box-shadow 200ms ease, border-color 200ms ease;
    }

    [class*="st-key-glass-"]:hover {
        border-color: rgba(61, 186, 116, 0.55) !important;
        box-shadow:
            0 14px 34px rgba(30, 123, 79, 0.18),
            0 0 16px rgba(61, 186, 116, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.95);
    }

    [class*="st-key-glass-"] ul li::marker {
        color: #2BAA63;
        font-size: 1.15em;
    }

    /* Tiles: glowing crystal edge. */
    [data-testid="stMetric"] {
        padding: 0.75rem 1rem 0.55rem !important;
        background: linear-gradient(165deg, rgba(255, 255, 255, 0.92), rgba(206, 238, 218, 0.72)) !important;
        backdrop-filter: blur(14px) saturate(150%);
        border: 1.5px solid rgba(61, 186, 116, 0.6) !important;
        border-radius: 14px !important;
        box-shadow:
            0 8px 22px rgba(30, 123, 79, 0.16),
            0 0 14px rgba(61, 186, 116, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.95);
    }

    [data-testid="stMetricValue"] {
        font-weight: 800;
    }

    .st-key-dataset-tiles [data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }

    /* Tiles in a row share one height. */
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
        align-items: stretch;
    }

    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stElementContainer"]:has(> [data-testid="stMetric"]) {
        height: 100%;
    }

    [data-testid="stMetric"] {
        height: 100%;
    }

    [data-testid="stMetricDelta"] {
        background: rgba(61, 186, 116, 0.16) !important;
        border-radius: 999px;
    }

    /* Sub-tabs (inside a step): light glass bar. */
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        padding: 0.35rem;
        border: 1px solid rgba(61, 186, 116, 0.3);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.62);
        backdrop-filter: blur(14px);
        box-shadow: 0 6px 18px rgba(30, 123, 79, 0.08);
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: background 180ms ease, color 180ms ease, box-shadow 180ms ease;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(120deg, #17864F 0%, #2BAA63 60%, #4FD08A 100%);
        box-shadow: 0 4px 14px rgba(43, 170, 99, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.35);
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] span {
        color: #FFFFFF !important;
    }

    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {
        display: none;
    }

    /* The step bar: a stepper on a dark green band. Each step has a round
       number badge and a line joining it to the next step; finished steps
       show a check mark and a green line. */
    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) {
        counter-reset: step;
        gap: 0 !important;
        padding: 0.7rem 1.4rem;
        border: 1px solid rgba(120, 220, 160, 0.35);
        border-radius: 16px;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0)),
            linear-gradient(120deg, #072B1D 0%, #0D4430 50%, #0A3524 100%);
        box-shadow:
            0 12px 30px rgba(7, 43, 29, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.12);
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"] {
        counter-increment: step;
        display: flex;
        flex: 1 1 0;
        align-items: center;
        gap: 0.75rem;
        padding: 0.2rem 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]:last-of-type {
        flex: 0 1 auto;
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"] > div {
        flex: none;
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]::before {
        content: counter(step);
        display: grid;
        flex: none;
        place-items: center;
        width: 2.4rem;
        height: 2.4rem;
        border: 2px solid rgba(160, 230, 190, 0.55);
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.06);
        color: rgba(225, 245, 233, 0.9);
        font-size: 1.05rem;
        font-weight: 800;
        transition: background 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]:not(:last-of-type)::after {
        content: "";
        flex: 1 1 auto;
        min-width: 1.5rem;
        height: 2px;
        margin: 0 1rem 0 0.25rem;
        border-radius: 2px;
        background: rgba(160, 230, 190, 0.3);
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"] p {
        color: rgba(225, 245, 233, 0.8);
        font-size: 1.05rem;
        white-space: nowrap;
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]:hover::before {
        border-color: #FFFFFF;
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]:hover p {
        color: #FFFFFF;
    }

    /* Current step */
    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"][aria-selected="true"]::before {
        border-color: transparent;
        background:
            radial-gradient(circle at 32% 28%, rgba(255, 255, 255, 0.5), transparent 38%),
            linear-gradient(135deg, #2BAA63, #4FD08A);
        color: #FFFFFF;
        box-shadow: 0 0 0 4px rgba(79, 208, 138, 0.25), 0 0 18px rgba(79, 208, 138, 0.6);
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* Finished steps (before the current one) */
    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]:has(~ button[aria-selected="true"])::before {
        content: "✓";
        border-color: transparent;
        background: rgba(43, 170, 99, 0.9);
        color: #FFFFFF;
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]:has(~ button[aria-selected="true"])::after {
        background: linear-gradient(90deg, #2BAA63, #4FD08A);
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]:focus-visible {
        outline: none;
    }

    .st-key-step-tabs div[data-baseweb="tab-list"]:not([role="tabpanel"] div[data-baseweb="tab-list"]) > button[data-baseweb="tab"]:focus-visible::before {
        box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.6);
    }

    /* Step header: a glossy green orb with the step number. */
    .step-hero {
        display: flex;
        gap: 1rem;
        align-items: center;
        margin: 0.9rem 0 1.1rem;
        padding: 0;
    }

    .step-hero__number {
        display: grid;
        flex: none;
        place-items: center;
        width: 3.4rem;
        height: 3.4rem;
        border-radius: 50%;
        background:
            radial-gradient(circle at 32% 28%, rgba(255, 255, 255, 0.55), transparent 32%),
            radial-gradient(circle at 50% 55%, #1E9E5F 0%, #0B5D3B 62%, #06331F 100%);
        color: #FFFFFF;
        font-size: 1.55rem;
        font-weight: 800;
        box-shadow:
            0 10px 22px rgba(7, 43, 29, 0.35),
            0 0 0 4px rgba(61, 186, 116, 0.18);
    }

    .step-hero h2 {
        margin: 0;
        padding: 0;
        color: var(--atlas-ink);
        font-size: 1.55rem;
        font-weight: 800;
        font-stretch: 85%;
    }

    .step-hero p {
        margin: 0.15rem 0 0;
        color: var(--atlas-slate);
        font-size: 0.98rem;
    }

    /* Step 1 map: OpenStreetMap inside a rounded frame. */
    .st-key-dark-map {
        overflow: hidden;
        border: 1px solid rgba(61, 186, 116, 0.45);
        border-radius: 14px;
        box-shadow: 0 8px 22px rgba(30, 123, 79, 0.16);
    }


    /* Switch buttons: every segmented control is a pill-shaped track; the
       selected option is a green thumb. */
    [data-testid="stButtonGroup"] > div {
        gap: 4px !important;
        padding: 4px;
        border: 1px solid rgba(61, 186, 116, 0.35);
        border-radius: 999px;
        background: rgba(20, 48, 42, 0.07);
        box-shadow: inset 0 2px 5px rgba(20, 48, 42, 0.12);
    }

    [data-testid="stButtonGroup"] button {
        border: none !important;
        border-radius: 999px !important;
        background: transparent !important;
        color: var(--atlas-slate) !important;
        font-weight: 600;
        box-shadow: none !important;
        transition: background 220ms ease, color 220ms ease, box-shadow 220ms ease;
    }

    [data-testid="stButtonGroup"] button:hover {
        color: var(--atlas-ink) !important;
        background: rgba(255, 255, 255, 0.55) !important;
    }

    [data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {
        background: linear-gradient(120deg, #17864F 0%, #2BAA63 60%, #4FD08A 100%) !important;
        color: #FFFFFF !important;
        box-shadow:
            0 3px 10px rgba(23, 134, 79, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
    }

    [data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] * {
        color: #FFFFFF !important;
    }

    /* Numbered sidebar sections */
    [data-testid="stSidebarContent"] {
        counter-reset: side-step;
    }

    [class*="st-key-side-step-"] {
        counter-increment: side-step;
    }

    [class*="st-key-side-step-"] summary {
        gap: 0.6rem;
    }

    [class*="st-key-side-step-"] summary::before {
        content: counter(side-step);
        display: grid;
        flex: none;
        place-items: center;
        width: 1.7rem;
        height: 1.7rem;
        border-radius: 50%;
        background: linear-gradient(135deg, #17864F, #4FD08A);
        color: #FFFFFF;
        font-size: 0.9rem;
        font-weight: 800;
        box-shadow: 0 2px 8px rgba(23, 134, 79, 0.35);
    }

    [class*="st-key-side-step-"] summary p {
        margin: 0;
        color: #1E7B4F;
        font-size: 0.86rem;
        font-weight: 500;
        line-height: 1.35;
    }

    [class*="st-key-side-step-"] summary p strong {
        color: var(--atlas-ink);
        font-size: 0.98rem;
        font-weight: 700;
    }

    [class*="st-key-side-step-"] summary [data-testid="stExpanderIcon"],
    [class*="st-key-side-step-"] summary [data-testid="stIconMaterial"],
    [class*="st-key-side-step-"] summary span:has(> [data-testid="stIconMaterial"]) {
        display: none;
    }

    [class*="st-key-side-step-"] details[open] summary::before {
        box-shadow: 0 0 0 3px rgba(79, 208, 138, 0.3), 0 2px 8px rgba(23, 134, 79, 0.35);
    }

    /* Sidebar switches stay on one line. */
    [data-testid="stSidebar"] [data-testid="stButtonGroup"] > div {
        flex-wrap: nowrap !important;
    }

    [data-testid="stSidebar"] [data-testid="stButtonGroup"] button {
        flex: 1 0 auto !important;
        width: auto !important;
        max-width: none !important;
        min-width: 0;
        padding-left: 0.55rem !important;
        padding-right: 0.55rem !important;
        white-space: nowrap;
    }

    .st-key-page-switch [data-testid="stButtonGroup"] button {
        min-height: 2.6rem;
        font-size: 1.02rem;
    }

    [data-testid="stButtonGroup"] button:focus-visible {
        outline: 3px solid rgba(43, 170, 99, 0.45) !important;
        outline-offset: 2px;
    }

    @media (prefers-reduced-motion: reduce) {
        [class*="st-key-glass-"],
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            transition: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# UI style switch (Default | Crystal), in the same pill style as the other
# switches. Its key is "ui_mode", so the choice is read at the top of the script
# on the next run; switching keeps every other selection in session state.
UI_MODE_LABELS = {
    "default": ":material/light_mode: Default",
    "crystal": ":material/diamond: Crystal",
}
st.container(key="ui-style-switch", horizontal=True, horizontal_alignment="right").segmented_control(
    "Switch UI Style",
    list(UI_MODE_LABELS),
    required=True,
    format_func=UI_MODE_LABELS.get,
    default=None if "ui_mode" in st.session_state else "default",
    key="ui_mode",
    label_visibility="collapsed",
    help="Switch UI Style",
)
st.markdown(
    """
    <style>
    .st-key-ui-style-switch {
        margin-bottom: -0.4rem;
    }

    .st-key-ui-style-switch [data-testid="stButtonGroup"] > div {
        flex-wrap: nowrap !important;
    }

    .st-key-ui-style-switch [data-testid="stButtonGroup"] button {
        min-height: 2rem !important;
        padding: 0.15rem 0.8rem !important;
        white-space: nowrap;
    }

    .st-key-ui-style-switch [data-testid="stButtonGroup"] button p {
        font-size: 0.85rem;
    }

    /* Streamlit's header buttons (Deploy, menu, toolbar actions such as GitHub)
       get the same pill look as the switch; their behaviour is unchanged. */
    [data-testid="stToolbar"] :is([data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stToolbarActions"]) button {
        min-height: 2.55rem;
        border: 1px solid rgba(61, 186, 116, 0.35) !important;
        background: rgba(20, 48, 42, 0.07) !important;
        box-shadow: inset 0 2px 5px rgba(20, 48, 42, 0.12);
        transition: background 220ms ease, box-shadow 220ms ease;
    }

    [data-testid="stToolbar"] [data-testid="stAppDeployButton"] button,
    [data-testid="stToolbar"] [data-testid="stToolbarActions"] button {
        padding: 0 1.05rem !important;
        border-radius: 9999px !important;
    }

    [data-testid="stToolbar"] [data-testid="stMainMenu"] button {
        width: 2.55rem;
        border-radius: 50% !important;
        justify-content: center;
    }

    [data-testid="stToolbar"] :is([data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stToolbarActions"]) button:hover {
        background: linear-gradient(120deg, #17864F 0%, #2BAA63 60%, #4FD08A 100%) !important;
        box-shadow: 0 3px 10px rgba(23, 134, 79, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }

    [data-testid="stToolbar"] :is([data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stToolbarActions"]) button:hover * {
        color: #FFFFFF !important;
    }

    [data-testid="stToolbar"] [data-testid="stAppDeployButton"] button p {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
if CRYSTAL_MODE:
    st.markdown(
        "<style>" + (APP_DIR / "crystal_mode.css").read_text(encoding="utf-8") + "</style>",
        unsafe_allow_html=True,
    )

app_view = st.container(key="page-switch").segmented_control(
    "Page",
    ["Presentation", "Demo"],
    required=True,
    default="Demo",
    format_func={
        "Presentation": ":material/slideshow: " + tr("Presentation"),
        "Demo": ":material/monitoring: " + tr("Demo"),
    }.get,
    label_visibility="collapsed",
    width="stretch",
)

if app_view == "Presentation":
    if not PRESENTATION_FILE.exists():
        st.error(f"Presentation file not found: {PRESENTATION_FILE}")
        st.stop()
    if not DEFAULT_DATA.exists():
        st.error(f"Default data file not found: {DEFAULT_DATA}")
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
    presentation_results = dict(st.session_state.get("presentation_results", {}))
    source_data = load_default_data(str(DEFAULT_DATA), DEFAULT_DATA.stat().st_mtime)
    source_data_payload = json.loads(
        source_data.to_json(orient="split", index=False)
    )
    presentation_results["source_data_columns"] = source_data_payload["columns"]
    presentation_results["source_data_rows"] = source_data_payload["data"]

    total_cells = source_data.size
    missing_cells = int(source_data.isna().sum().sum())
    key_columns = ["ouID", "deID", "cocID", "sexID"]
    duplicate_keys = int(source_data.duplicated(key_columns).sum())
    year_columns = [str(year) for year in HISTORICAL_YEARS]
    numeric_year_values = source_data[year_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_year_values = int(
        (numeric_year_values.isna() | (numeric_year_values < 0)).sum().sum()
    )
    year_value_count = numeric_year_values.size
    district_count = int(source_data["ouID"].nunique())
    age_count = int(source_data["cocID"].nunique())
    sex_count = int(source_data["sexID"].nunique())
    expected_rows = district_count * age_count * sex_count

    presentation_results.update(
        {
            "quality_completeness": f"{(total_cells - missing_cells) / total_cells:.2%}",
            "quality_completeness_note": (
                f"{missing_cells:,} ຄ່າວ່າງ ຈາກ {total_cells:,} ເຊວ "
                f"({missing_cells:,} missing of {total_cells:,} cells)"
            ),
            "quality_unique_keys": f"{(len(source_data) - duplicate_keys) / len(source_data):.2%}",
            "quality_unique_keys_note": (
                f"{len(source_data) - duplicate_keys:,} ລະຫັດບໍ່ຊ້ຳ · "
                f"{duplicate_keys:,} ລະຫັດຊ້ຳ (duplicate keys)"
            ),
            "quality_valid_values": f"{(year_value_count - invalid_year_values) / year_value_count:.2%}",
            "quality_valid_values_note": (
                f"{year_value_count - invalid_year_values:,} ຄ່າເປັນຕົວເລກ ແລະ ບໍ່ຕິດລົບ "
                f"(numeric and nonnegative)"
            ),
            "quality_structural_coverage": (
                f"{len(source_data) / expected_rows:.2%}" if expected_rows else "—"
            ),
            "quality_structural_coverage_note": (
                f"{district_count:,} ເມືອງ × {age_count:,} ອາຍຸ × {sex_count:,} ເພດ "
                f"= {len(source_data):,} ແຖວ"
            ),
        }
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
        json.dumps(presentation_results, ensure_ascii=False, sort_keys=True),
    )
    st.iframe(presentation_html, height=900, width="stretch")
    st.stop()

st.sidebar.segmented_control(
    "ພາສາ / Language",
    list(LANGUAGES),
    required=True,
    default=DEFAULT_LANGUAGE,
    format_func=LANGUAGES.get,
    key="demo_language",
    width="stretch",
)
st.markdown(
    "<style>.stApp[data-test-script-state=\"running\" i]::before "
    f"{{ content: \"{tr('Calculating…')}\"; }}</style>",
    unsafe_allow_html=True,
)

# Option widgets whose labels change with the language get one key per
# language: Streamlit matches their state by the displayed text, so a shared key
# would lose the selection. The chosen value is copied over when the language
# switches.
language_changed = (
    st.session_state.get("_previous_language", current_language()) != current_language()
)
st.session_state["_previous_language"] = current_language()


def lang_key(name: str) -> str:
    key = f"{name}__{current_language()}"
    saved = st.session_state.get(f"_{name}_value")
    if saved is not None and (language_changed or key not in st.session_state):
        st.session_state[key] = saved
    return key


def remember(name: str, value: object) -> object:
    st.session_state[f"_{name}_value"] = value
    return value


# Demo view: the "Switch UI Style" button joins Streamlit's header, just left of
# the Deploy button / toolbar. A tiny script measures where the toolbar starts
# (it is wider once deployed, e.g. with the GitHub icon) and the CSS uses that.
st.markdown(
    """
    <style>
    .st-key-ui-style-switch {
        position: fixed;
        top: 12px;
        right: var(--ui-switch-right, 130px);
        z-index: 999991;
        width: auto !important;
        margin: 0 !important;
    }

    .st-key-ui-style-measure {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="ui-style-measure"):
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          function place() {
            const toolbar = doc.querySelector('[data-testid="stToolbar"]');
            if (!toolbar) return;
            const width = window.parent.innerWidth;
            let left = width;
            // Deploy, the main menu and toolbar actions (Share, GitHub, ...).
            // The temporary "Running..." status widget is ignored.
            toolbar
              .querySelectorAll(
                '[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], ' +
                  '[data-testid="stToolbarActions"] > *'
              )
              .forEach((el) => {
                const box = el.getBoundingClientRect();
                if (box.width > 0 && box.height > 0) left = Math.min(left, box.left);
              });
            const right = Math.max(width - left + 12, 16);
            doc.documentElement.style.setProperty("--ui-switch-right", right + "px");
          }
          place();
          setInterval(place, 1000);
          window.parent.addEventListener("resize", place);
        })();
        </script>
        """,
        height=0,
    )

masthead = st.container()
masthead.markdown(
    f"""
    <header class="atlas-masthead">
        <h1>{tr("Laos population forecast")}</h1>
        <p>{tr("By district, age and sex, from Ministry of Health DHIS2 data.")}</p>
    </header>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header(tr("Data and forecast"))


def side_step(number: int, title: str, value: str, expanded: bool = False):
    """A numbered sidebar section; the collapsed label shows the current setting."""
    return st.sidebar.expander(
        f"**{title}**  \n{value}", expanded=expanded, key=f"side-step-{number}"
    )


uploaded_name = getattr(st.session_state.get("upload_csv"), "name", None) or DEFAULT_DATA.name
with side_step(1, tr("Data source"), uploaded_name):
    uploaded_file = st.file_uploader(
        tr("Upload population CSV"), type=["csv"], key="upload_csv"
    )
    st.caption(tr("Without an upload, the app uses {name}.", name=DEFAULT_DATA.name))

with side_step(2, tr("Forecast years"), f"2026–{st.session_state.get('future_end', 2030)}"):
    future_end = st.slider(tr("Forecast through"), 2026, 2040, 2030, key="future_end")

side_threshold = st.session_state.get("correction_threshold", DEFAULT_CORRECTION_THRESHOLD)
with side_step(3, tr("Outlier correction"), tr("threshold {value}", value=f"{side_threshold:g}")):
    correction_threshold = st.slider(
        tr("Outlier correction threshold"),
        min_value=2.5,
        max_value=6.0,
        value=DEFAULT_CORRECTION_THRESHOLD,
        step=0.25,
        help=tr(
            "Lower values flag and potentially correct more observations. Higher values "
            "require more exceptional robust scores. Every correction still requires an "
            "independent demographic confirmation."
        ),
        key="correction_threshold",
    )
model_key = lang_key("forecast_model")
model_labels = {ALL_MODELS_OPTION: tr(ALL_MODELS_OPTION), **MODEL_LABELS}
with side_step(4, tr("Forecast model"), model_labels[st.session_state.get(model_key, ALL_MODELS_OPTION)]):
    selected_model_option = st.selectbox(
        tr("Forecast model"),
        [ALL_MODELS_OPTION, *MODEL_NAMES],
        format_func=model_labels.get,
        help=tr("Choose one model to inspect, or All models to compare their results and accuracy."),
        key=model_key,
        label_visibility="collapsed",
    )
remember("forecast_model", selected_model_option)
selected_models = (
    list(MODEL_NAMES)
    if selected_model_option == ALL_MODELS_OPTION
    else [selected_model_option]
)

def baseline_label(option: str) -> str:
    if option == ORIGINAL_INPUT_OPTION:
        return tr("Original data")
    return tr(
        "Outlier-corrected data (threshold {threshold})",
        threshold=f"{correction_threshold:g}",
    )


baseline_key = lang_key("baseline_type")
forecast_input_option = st.segmented_control(
    tr("Baseline data type"),
    [ORIGINAL_INPUT_OPTION, CORRECTED_INPUT_OPTION],
    required=True,
    default=None if baseline_key in st.session_state else CORRECTED_INPUT_OPTION,
    format_func={
        option: baseline_label(option)
        for option in (ORIGINAL_INPUT_OPTION, CORRECTED_INPUT_OPTION)
    }.get,
    help=tr(
        "Choose the baseline used by the forecast. Outlier-corrected data changes only "
        "temporal anomalies independently confirmed by a demographic check; the original "
        "upload is never overwritten."
    ),
    width="stretch",
    key=baseline_key,
)
forecast_input_option = remember(
    "baseline_type", forecast_input_option or CORRECTED_INPUT_OPTION
)
forecast_input_label = baseline_label(forecast_input_option)

try:
    if uploaded_file is None:
        if not DEFAULT_DATA.exists():
            st.error(tr("Default data file not found: {path}", path=DEFAULT_DATA))
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
    st.error(tr("Could not load the population data: {error}", error=error))
    st.stop()

filter_count = sum(
    bool(st.session_state.get(f"filter_{column}"))
    for column in ("ouName", "sexName", "cocName")
)
with side_step(
    5,
    tr("Filters"),
    tr("{count} active", count=filter_count) if filter_count else tr("All data"),
) as filter_container:
    filtered_original_data = filter_data(data, filter_container)

map_background_options = {
    "osm": tr("OpenStreetMap"),
    "none": tr("None"),
}
map_background_key = lang_key("map_background")
with side_step(
    6,
    tr("Map background"),
    map_background_options[st.session_state.get(map_background_key) or "osm"],
):
    show_basemap = (
        remember(
            "map_background",
            st.segmented_control(
                tr("Map background"),
                list(map_background_options),
                required=True,
                format_func=map_background_options.get,
                default=None if map_background_key in st.session_state else "osm",
                key=map_background_key,
                width="stretch",
                label_visibility="collapsed",
            )
            or "osm",
        )
        == "osm"
    )
    st.caption(
        tr(
            "OpenStreetMap needs an internet connection. Choose None to show the maps "
            "offline."
        )
    )

if filtered_original_data.empty:
    st.warning(tr("The selected filters contain no records. Change or clear the filters."))
    st.stop()

with st.sidebar.expander(tr("5–7 minute demo path"), icon=":material/route:"):
    st.markdown(block("demo_path", current_language()))

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
    sex_colors = {"Female": ATLAS_MADDER, "Male": ATLAS_MEKONG}
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
                marker_color=sex_colors.get(str(sex_name), ATLAS_SLATE),
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
            marker_color=ATLAS_FOREST,
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
            marker_color=ATLAS_FOREST,
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
        template=PLOTLY_TEMPLATE,
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        font={"size": 18, "color": ATLAS_SLATE},
    )
    outlier_evidence_figure.update_layout(
        template=PLOTLY_TEMPLATE,
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
            line={"color": ATLAS_SLATE, "width": 2, "dash": "dash"},
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
            line={"color": ATLAS_FOREST, "width": 2.5},
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
            marker={"color": ATLAS_MADDER, "size": 13, "symbol": "x", "line": {"width": 2}},
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
                "color": ATLAS_FOREST,
                "size": 13,
                "symbol": "diamond",
                "line": {"color": CHART_SURFACE, "width": 1.5},
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
        template=PLOTLY_TEMPLATE,
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        font={"size": 18, "color": ATLAS_SLATE},
    )
    correction_comparison_figure.update_layout(
        template=PLOTLY_TEMPLATE,
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
    "outlier_evidence_plotly": slide_json(outlier_evidence_figure),
    "correction_comparison_plotly": slide_json(correction_comparison_figure),
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
    "validation_plotly": slide_json(
        interactive_validation_figure(
            validation_result,
            list(MODEL_NAMES),
            lao_presentation=True,
        )
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

# Time ribbon in the masthead: train → test → forecast, one cell per year.
ribbon_model = (
    presentation_best_model
    if selected_model_option == ALL_MODELS_OPTION
    else selected_model_option
)
ribbon_mape = validation_result.metrics.loc[ribbon_model, "MAPE (%)"]
ribbon_train_years = list(validation_result.train_years)
ribbon_test_years = list(validation_result.test_years)

# Land area of the districts in the current selection (all of Lao PDR when no
# district filter is set), measured from the map boundaries.
ribbon_area_km2 = 0.0
if BOUNDARY_FILE.exists() and "ouID" in filtered_data.columns:
    boundary_names = map_boundaries()["names"]
    district_areas = boundary_names.loc[
        boundary_names["level"] == population_map.DISTRICT_LEVEL
    ].set_index("ouID")["area_km2"]
    ribbon_area_km2 = float(
        district_areas.reindex(filtered_data["ouID"].unique()).sum()
    )


# ------------------------------------------------------------------ the three steps
def crystal(figure: go.Figure, height: int | None = None) -> go.Figure:
    """Crystal chart style: transparent glass background, soft grid, clear hover."""
    if CRYSTAL_MODE:
        figure.update_layout(template=PLOTLY_TEMPLATE)
        for trace in figure.data:
            # Value labels outside bars sit on the dark panel: make them light.
            if getattr(trace, "textposition", None) == "outside" and not trace.textfont.color:
                trace.textfont.color = CHART_TEXT
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={
            "bgcolor": CHART_HOVER,
            "bordercolor": "rgba(20,48,42,0.18)",
            "font_color": CHART_TEXT,
        },
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
    )
    if height:
        figure.update_layout(height=height)
    figure.update_xaxes(gridcolor="rgba(20,48,42,0.07)", zeroline=False, showline=False)
    figure.update_yaxes(gridcolor="rgba(20,48,42,0.07)", zeroline=False, showline=False)
    return figure


_glass_count = 0


def glass(parent=None):
    """A frosted-glass card; the .st-key-glass-* CSS gives it the crystal look."""
    global _glass_count
    _glass_count += 1
    return (parent or st).container(border=True, key=f"glass-{_glass_count}")


def card_title(icon: str, title: str, caption: str | None = None) -> None:
    st.markdown(f":material/{icon}: **{title}**")
    if caption:
        st.caption(caption)


def step_header(number: int, title: str, text: str) -> None:
    st.markdown(
        f'<div class="step-hero"><span class="step-hero__number">{number}</span>'
        f"<div><h2>{title}</h2><p>{text}</p></div></div>",
        unsafe_allow_html=True,
    )


CRYSTAL_CHART_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def sex_view_switch(name: str) -> bool:
    """Number / Share (%) switch for a year-and-sex chart; True means share."""
    options = {"number": tr("Number"), "share": tr("Share (%)")}
    key = lang_key(name)
    view = remember(
        name,
        st.segmented_control(
            tr("Show"),
            list(options),
            required=True,
            format_func=options.get,
            default=None if key in st.session_state else "number",
            key=key,
            label_visibility="collapsed",
        )
        or "number",
    )
    return view == "share"


def sex_year_figure(
    by_sex: pd.DataFrame, show_share: bool, forecast_years: list[int] | None = None
) -> go.Figure:
    """Stacked men/women bars per year with the total on top.

    ``by_sex`` has one row per year and Male/Female columns.  Years listed in
    ``forecast_years`` get hatched bars on a gold background.
    """
    years = [int(year) for year in by_sex.index]
    forecast_years = [year for year in (forecast_years or []) if year in years]
    is_forecast = [year in forecast_years for year in years]
    totals = by_sex.sum(axis=1)
    growth = totals.pct_change() * 100
    figure = go.Figure()
    if forecast_years:
        figure.add_vrect(
            x0=forecast_years[0] - 0.5,
            x1=forecast_years[-1] + 0.5,
            fillcolor="rgba(184, 144, 47, 0.12)",
            line_width=0,
            layer="below",
        )
        figure.add_annotation(
            x=(forecast_years[0] + forecast_years[-1]) / 2,
            y=1.0,
            yref="paper",
            yanchor="bottom",
            text=f"<b>{tr('Forecast')}</b> {forecast_years[0]}–{forecast_years[-1]}",
            showarrow=False,
            font={"color": FORECAST_LABEL_COLOUR, "size": 12},
        )
    for sex_name, color in (("Male", ATLAS_MEKONG), ("Female", ATLAS_MADDER)):
        if sex_name not in by_sex:
            continue
        values = by_sex[sex_name]
        shares = values / totals * 100
        figure.add_trace(
            go.Bar(
                x=years,
                y=shares if show_share else values,
                customdata=np.column_stack([values, shares]),
                name=tr(sex_name),
                marker={
                    "color": color,
                    "opacity": 0.82,
                    "line": {"color": CHART_SURFACE, "width": 1.2},
                    "pattern": {
                        "shape": ["/" if forecast else "" for forecast in is_forecast],
                        "fgcolor": "rgba(255, 255, 255, 0.55)",
                        "fillmode": "overlay",
                        "size": 7,
                        "solidity": 0.25,
                    },
                },
                text=(
                    [f"{share:.1f}%" for share in shares]
                    if show_share
                    else [f"{value / 1e6:.1f}M" for value in values]
                ),
                textposition="inside",
                insidetextanchor="middle",
                textangle=0,
                textfont={"color": "#FFFFFF", "size": 10},
                hovertemplate=(
                    f"{tr(sex_name)}: %{{customdata[0]:,.0f}} (%{{customdata[1]:.1f}}%)<extra></extra>"
                ),
            )
        )
    if not show_share:
        # Total on top of each stack, joined by a thin line to show the trend.
        # With many years, only the first, last actual and last totals are written.
        last_actual = max((year for year, forecast in zip(years, is_forecast) if not forecast), default=None)
        label_years = set(years) if len(years) <= 11 else {years[0], years[-1], last_actual}
        figure.add_trace(
            go.Scatter(
                x=years,
                y=totals,
                customdata=[["—" if np.isnan(value) else f"{value:+.2f}%"] for value in growth],
                name=tr("Total"),
                mode="lines+markers+text",
                line={"color": CHART_TEXT, "width": 1.5, "dash": "dot"},
                marker={
                    "size": 7,
                    "color": [CHART_SURFACE if not forecast else ATLAS_GOLD for forecast in is_forecast],
                    "line": {"color": CHART_TEXT, "width": 2},
                },
                text=[f"{value / 1e6:.2f}M" if year in label_years else "" for year, value in zip(years, totals)],
                textposition="top center",
                textfont={"color": CHART_TEXT, "size": 11},
                hovertemplate=(
                    f"<b>{tr('Total')}: %{{y:,.0f}}</b>"
                    f"<br>{tr('Growth vs previous year')}: %{{customdata[0]}}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        barmode="stack",
        bargap=0.22,
        hovermode="x unified",
        uniformtext={"minsize": 9, "mode": "hide"},
    )
    figure.update_xaxes(tickmode="array", tickvals=years)
    if show_share:
        figure.update_yaxes(range=[0, 100], ticksuffix="%")
    else:
        figure.update_yaxes(tickformat=",.0f", range=[0, float(totals.max()) * 1.14])
    return figure

step1_tab, step2_tab, step3_tab = st.tabs(
    [
        tr("Dataset and sources"),
        tr("Forecasting and visualization"),
        tr("Summary dashboard"),
    ],
    key="step-tabs",
)
with step1_tab:
    step_header(
        1,
        tr("Dataset and data sources"),
        tr(
            "Population by age, sex, district and year from the Ministry of Health DHIS2 "
            "system: what the data covers, where it comes from and how complete it is."
        ),
    )
    step1_overview = st.container()
    step1_source = st.container()
with step2_tab:
    step_header(
        2,
        tr("Forecasting details and visualization"),
        tr(
            "How the models are built and tested, how outliers are handled, the forecast "
            "results, and maps and charts of the population."
        ),
    )
    methodology_tab, data_tab, overview_tab, pyramid_tab, analysis_tab = st.tabs(
        [
            ":material/functions: " + tr("Methodology"),
            ":material/rule: " + tr("Outliers"),
            ":material/query_stats: " + tr("Results"),
            ":material/map: " + tr("Maps & charts"),
            ":material/analytics: " + tr("Further analysis"),
        ]
    )
with step3_tab:
    step_header(
        3,
        tr("Summary and interactive dashboard"),
        tr(
            "The headline numbers, key findings and downloads in one place. Everything "
            "follows the sidebar settings and the baseline data type."
        ),
    )
    step3_dashboard = st.container()
    step3_findings = st.container()
    export_tab = st.container()
    step3_info = st.container()

# ------------------------------------------------------------------ step 1 content
with step1_overview:
    latest_year = HISTORICAL_YEARS[-1]
    latest_column = str(latest_year)
    district_filter_active = bool(st.session_state.get("filter_ouName"))
    province_count = data["ouName"].astype(str).str[:2].nunique()
    boundary_info = map_boundaries() if BOUNDARY_FILE.exists() else None
    country_area_km2 = 0.0
    if boundary_info is not None:
        boundary_table = boundary_info["names"].copy()
        country_area_km2 = float(
            boundary_table.loc[
                boundary_table["level"] == population_map.DISTRICT_LEVEL, "area_km2"
            ].sum()
        )

    tiles = st.container(key="dataset-tiles").columns(5)
    tiles[0].metric(
        ":material/table_rows: " + tr("Records"),
        f"{len(data):,}",
        delta=tr("{count} values", count=f"{len(data) * len(HISTORICAL_YEARS):,}"),
        delta_color="off",
        delta_arrow="off",
        help=tr("district–age–sex series"),
        border=True,
    )
    tiles[1].metric(
        ":material/location_city: " + tr("Districts"),
        f"{data['ouID'].nunique():,}",
        delta=tr("{count} provinces", count=province_count),
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    tiles[2].metric(
        ":material/groups: " + tr("Age groups"),
        f"{data['cocID'].nunique():,}",
        delta=tr("{count} sexes", count=data["sexID"].nunique()),
        delta_color="off",
        delta_arrow="off",
        help=tr("{count} sexes, single years 0–80+", count=data["sexID"].nunique()),
        border=True,
    )
    tiles[3].metric(
        ":material/calendar_month: " + tr("Years"),
        f"{len(HISTORICAL_YEARS)}",
        delta=f"{HISTORICAL_YEARS[0]}–{latest_year}",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    tiles[4].metric(
        ":material/landscape: " + tr("Area (km²)"),
        f"{country_area_km2:,.0f}" if country_area_km2 else "—",
        delta=tr("Lao PDR"),
        delta_color="off",
        delta_arrow="off",
        help=tr("measured from the map boundaries"),
        border=True,
    )

    map_column, source_column = st.columns([1.3, 1])
    with map_column:
        with glass():
            card_title(
                "public",
                tr("Lao PDR: area covered by the data")
                if not district_filter_active
                else tr("Selected districts in Lao PDR"),
                tr(
                    "The whole country (level 1) is shown when all districts are selected. "
                    "Choose districts in the sidebar to highlight them. Hover for area, "
                    "population ({year}) and density.",
                    year=latest_year,
                ),
            )
            if boundary_info is None:
                st.warning(tr("Map boundary file not found: {path}", path=BOUNDARY_FILE))
            else:
                district_population = data.groupby("ouID")[latest_column].sum()
                area_table = boundary_table.copy()
                area_table["Population"] = area_table["ouID"].map(district_population)
                province_codes = area_table["name"].astype(str).str[:2]
                district_codes = data.groupby("ouID")["ouName"].first().astype(str).str[:2]
                province_population = district_population.groupby(district_codes).sum()
                is_province = area_table["level"] == population_map.PROVINCE_LEVEL
                area_table.loc[is_province, "Population"] = province_codes[is_province].map(
                    province_population
                )
                is_country = area_table["level"] == population_map.COUNTRY_LEVEL
                area_table.loc[is_country, "Population"] = district_population.sum()
                area_table.loc[is_country, "area_km2"] = country_area_km2
                area_table["Population"] = area_table["Population"].fillna(0)
                area_table["Density"] = area_table["Population"] / area_table["area_km2"]
                highlight_ids = (
                    list(filtered_original_data["ouID"].unique()) if district_filter_active else None
                )
                st.container(key="dark-map").plotly_chart(
                    population_map.area_overview_figure(
                        boundary_info,
                        area_table,
                        highlight_ids,
                        {
                            "area": tr("Area (km²)"),
                            "population": tr("Population"),
                            "density": tr("People per km²"),
                        },
                        PLOTLY_TEMPLATE,
                        height=470,
                        basemap=show_basemap,
                    ),
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": True}, theme=CHART_THEME,
                )
                if district_filter_active:
                    selected_area = area_table.loc[
                        area_table["ouID"].isin(highlight_ids), "area_km2"
                    ].sum()
                    st.caption(
                        tr(
                            "{count} selected district(s): {area} km², {share} of the country.",
                            count=len(highlight_ids),
                            area=f"{selected_area:,.0f}",
                            share=f"{selected_area / country_area_km2:.1%}",
                        )
                    )

    with source_column:
        with glass():
            card_title("source", tr("Data source"))
            st.markdown(block("data_source", current_language()))
            st.caption(tr("Loaded file: {name}", name=source_name))
        with glass():
            card_title("verified", tr("Data quality checks"))
            total_cells = data.size
            missing_cells = int(data.isna().sum().sum())
            duplicate_keys = int(data.duplicated(["ouID", "deID", "cocID", "sexID"]).sum())
            year_values = data[[str(year) for year in HISTORICAL_YEARS]].apply(
                pd.to_numeric, errors="coerce"
            )
            invalid_values = int((year_values.isna() | (year_values < 0)).sum().sum())
            expected_rows = data["ouID"].nunique() * data["cocID"].nunique() * data["sexID"].nunique()
            quality = pd.DataFrame(
                [
                    (tr("Completeness"), (total_cells - missing_cells) / total_cells, tr("{count} empty cells", count=f"{missing_cells:,}")),
                    (tr("Unique keys"), (len(data) - duplicate_keys) / len(data), tr("{count} duplicate keys", count=f"{duplicate_keys:,}")),
                    (tr("Valid values"), (year_values.size - invalid_values) / year_values.size, tr("numeric and not negative")),
                    (tr("Structural coverage"), len(data) / expected_rows if expected_rows else np.nan, tr("districts × ages × sexes")),
                ],
                columns=[tr("Check"), tr("Score"), tr("Detail")],
            )
            st.dataframe(
                quality,
                hide_index=True,
                width="stretch",
                column_config={
                    tr("Score"): st.column_config.ProgressColumn(
                        format="percent", min_value=0.0, max_value=1.0
                    ),
                },
            )

    st.caption(
        tr(
            "Charts below use the source data as loaded (before outlier correction), with "
            "the sidebar filters applied."
        )
    )
    year_chart_column, age_chart_column = st.columns(2)
    with year_chart_column:
        with glass():
            card_title("stacked_bar_chart", tr("Population by year and sex"))
            by_sex = (
                filtered_original_data.groupby("sexName")[[str(year) for year in HISTORICAL_YEARS]]
                .sum()
                .T
            )
            show_share = sex_view_switch("sex_chart_view")
            st.plotly_chart(
                crystal(sex_year_figure(by_sex, show_share), 360),
                width="stretch",
                config=CRYSTAL_CHART_CONFIG, theme=CHART_THEME,
            )
    with age_chart_column:
        with glass():
            card_title("area_chart", tr("Population by single age, {year}", year=latest_year))
            single_ages = pd.to_numeric(
                filtered_original_data["cocCode"].astype(str).str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )
            by_age = (
                filtered_original_data.assign(age=single_ages)
                .groupby(["age", "sexName"])[latest_column]
                .sum()
                .unstack(fill_value=0)
                .sort_index()
            )
            age_figure = go.Figure()
            for sex_name, rgb in SEX_RGB.items():
                if sex_name in by_age:
                    age_figure.add_trace(
                        go.Scatter(
                            x=by_age.index,
                            y=by_age[sex_name],
                            name=tr(sex_name),
                            mode="lines",
                            line={"color": f"rgb({rgb})", "width": 2.5, "shape": "spline"},
                            fill="tozeroy",
                            fillgradient={
                                "type": "vertical",
                                "colorscale": [[0, f"rgba({rgb},0.02)"], [1, f"rgba({rgb},0.30)"]],
                            },
                            hovertemplate=f"{tr('Age')} %{{x}} · {tr(sex_name)}<br>{tr('Population')}: %{{y:,.0f}}<extra></extra>",
                        )
                    )
            age_figure.update_xaxes(title_text=tr("Age"))
            age_figure.update_yaxes(tickformat=",.0f")
            st.plotly_chart(crystal(age_figure, 360), width="stretch", config=CRYSTAL_CHART_CONFIG, theme=CHART_THEME)

    with glass():
        card_title(
            "leaderboard",
            tr("Largest districts, {year}", year=latest_year),
            tr("Top 12 districts in the current selection by population."),
        )
        top_districts = (
            filtered_original_data.groupby("ouName")[latest_column].sum().nlargest(12).sort_values()
        )
        district_figure = go.Figure(
            go.Bar(
                x=top_districts.values,
                y=top_districts.index,
                orientation="h",
                marker={
                    "color": top_districts.values,
                    "colorscale": [[0, "rgba(61,186,116,0.55)"], [1, "rgba(11,93,59,0.9)"]],
                    "line": {"color": CHART_SURFACE, "width": 1.2},
                },
                text=[f"{value:,.0f}" for value in top_districts.values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"<b>%{{y}}</b><br>{tr('Population')}: %{{x:,.0f}}<extra></extra>",
            )
        )
        district_figure.update_xaxes(tickformat=",.0f")
        st.plotly_chart(
            crystal(district_figure, 420), width="stretch", config=CRYSTAL_CHART_CONFIG, theme=CHART_THEME
        )

    with st.expander(":material/menu_book: " + tr("Data dictionary")):
        dictionary_rows = [
            ("ouID", "text", tr("District identifier in DHIS2")),
            ("ouName", "text", tr("District code and name (the first two digits are the province)")),
            ("deID / deName", "text", tr("Data element (Population)")),
            ("cocID / cocName / cocCode", "text / number", tr("Age: single years 0–79 and 80+")),
            ("sexID / sexName / sexCode", "text", tr("Sex: Male (M) or Female (F)")),
            (f"{HISTORICAL_YEARS[0]} … {latest_year}", "number", tr("Population in each year (one column per year)")),
        ]
        st.dataframe(
            pd.DataFrame(dictionary_rows, columns=[tr("Column"), tr("Type"), tr("Description")]),
            hide_index=True,
            width="stretch",
        )

# Summary dashboard (step 3): stat tiles, the population trajectory
# (train / test / forecast years as background bands) and the age structure.
summary_years = [*HISTORICAL_YEARS, *future_years]
history_count = len(HISTORICAL_YEARS)
summary_population = np.concatenate(
    [
        np.asarray(result.actual, dtype=float),
        np.asarray(result.future[ribbon_model], dtype=float),
    ]
)
summary_density = (
    summary_population / ribbon_area_km2
    if ribbon_area_km2
    else np.full_like(summary_population, np.nan)
)
summary_growth = np.concatenate(
    [[np.nan], np.diff(summary_population) / summary_population[:-1] * 100]
)
first_population = summary_population[0]
last_actual_population = summary_population[history_count - 1]
end_population = summary_population[-1]
first_year, last_actual_year = HISTORICAL_YEARS[0], HISTORICAL_YEARS[-1]
forecast_model_label = MODEL_LABELS[ribbon_model]
average_forecast_growth = (
    (end_population / last_actual_population) ** (1 / len(future_years)) - 1
) * 100


def sparkline(values) -> list[float]:
    """Change from the first value: the tile charts show the trend's shape."""
    values = np.asarray(values, dtype=float)
    return [round(float(value), 2) for value in values - values[0]]


test_errors = (
    np.abs(validation_result.predictions[ribbon_model] - validation_result.actual)
    / validation_result.actual
    * 100
)

with step3_dashboard:
    st.subheader(":material/monitoring: " + tr("Interactive summary dashboard"))
    tiles = st.columns(4)
    tiles[0].metric(
        tr("Population in {year}", year=last_actual_year),
        f"{last_actual_population:,.0f}",
        delta=tr(
            "{change} since {year}",
            change=f"{(last_actual_population / first_population - 1) * 100:+.1f}%",
            year=first_year,
        ),
        delta_color="normal",
        chart_data=sparkline(summary_population[:history_count]),
        chart_type="area",
        border=True,
    )
    tiles[1].metric(
        tr("Forecast for {year}", year=future_end),
        f"{end_population:,.0f}",
        delta=tr(
            "{change} vs {year}",
            change=f"{(end_population / last_actual_population - 1) * 100:+.1f}%",
            year=last_actual_year,
        ),
        delta_color="normal",
        chart_data=sparkline(summary_population),
        chart_type="area",
        border=True,
        help=tr("Model used for the forecast: {model}.", model=forecast_model_label),
    )
    if ribbon_area_km2:
        tiles[2].metric(
            tr("People per km², {year}", year=future_end),
            f"{summary_density[-1]:,.1f}",
            delta=tr(
                "{change} vs {year}",
                change=f"{summary_density[-1] - summary_density[history_count - 1]:+.1f}",
                year=last_actual_year,
            ),
            delta_color="normal",
            chart_data=sparkline(summary_density),
            chart_type="area",
            border=True,
        )
    tiles[3].metric(
        tr("Test-year MAPE"),
        f"{ribbon_mape:.4f}%",
        delta=forecast_model_label,
        delta_color="off",
        delta_arrow="off",
        chart_data=[round(float(value), 4) for value in test_errors],
        chart_type="bar",
        border=True,
        help=tr(
            "Mean absolute percentage error on the unseen {start}–{end} test years. The "
            "bars show the error in each test year.",
            start=ribbon_test_years[0],
            end=ribbon_test_years[-1],
        ),
    )

    measure_options = ["population", "density"] if ribbon_area_km2 else ["population"]
    measure_labels = {
        "population": tr("Population"),
        "density": tr("People per km²"),
    }
    measure_key = lang_key("summary_measure")
    summary_measure = remember(
        "summary_measure",
        st.segmented_control(
            tr("Show"),
            measure_options,
            required=True,
            format_func=measure_labels.get,
            default=None if measure_key in st.session_state else "population",
            key=measure_key,
        )
        or "population",
    )
    trajectory_column, age_column = st.columns([1.7, 1])
    with glass(trajectory_column):
        y_values = summary_density if summary_measure == "density" else summary_population
        hover_data = np.column_stack([summary_population, summary_density, summary_growth])
        hover_rows = (
            f"{tr('Population')}: %{{customdata[0]:,.0f}}"
            + (f"<br>{tr('People per km²')}: %{{customdata[1]:,.1f}}" if ribbon_area_km2 else "")
            + f"<br>{tr('Annual growth')}: %{{customdata[2]:+.2f}}%"
        )

        trajectory = go.Figure()
        phases = (
            [
                (ribbon_train_years, tr("Train"), "rgba(43, 217, 138, 0.04)", "#4FE0A0"),
                (ribbon_test_years, tr("Test"), "rgba(43, 217, 138, 0.09)", "#4FE0A0"),
                (future_years, tr("Forecast"), "rgba(25, 195, 217, 0.08)", "#5FD8E8"),
            ]
            if CRYSTAL_MODE
            else [
                (ribbon_train_years, tr("Train"), "rgba(30, 123, 79, 0.07)", ATLAS_FOREST),
                (ribbon_test_years, tr("Test"), "rgba(30, 123, 79, 0.16)", ATLAS_FOREST),
                (future_years, tr("Forecast"), "rgba(184, 144, 47, 0.16)", "#7A5F1C"),
            ]
        )
        for years, label, fill, text_color in phases:
            trajectory.add_vrect(
                x0=years[0] - 0.5,
                x1=years[-1] + 0.5,
                fillcolor=fill,
                line_width=0,
                layer="below",
            )
            trajectory.add_annotation(
                x=(years[0] + years[-1]) / 2,
                y=1.0,
                yref="paper",
                yanchor="bottom",
                text=f"<b>{label}</b> {years[0]}–{years[-1]}",
                showarrow=False,
                font={"color": text_color, "size": 12},
            )
        history_x = summary_years[:history_count]
        future_x = summary_years[history_count - 1 :]
        if CRYSTAL_MODE:
            # Soft gradient area under the actual line, then glow + line on top.
            trajectory.add_trace(
                go.Scatter(
                    x=history_x,
                    y=y_values[:history_count],
                    mode="lines",
                    line={"width": 0, "shape": "spline"},
                    fill="tozeroy",
                    fillgradient={
                        "type": "vertical",
                        "colorscale": [[0, "rgba(43, 217, 138, 0.0)"], [1, "rgba(43, 217, 138, 0.28)"]],
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            trajectory.add_trace(
                go.Scatter(
                    x=future_x,
                    y=y_values[history_count - 1 :],
                    mode="lines",
                    line={"width": 0, "shape": "spline"},
                    fill="tozeroy",
                    fillgradient={
                        "type": "vertical",
                        "colorscale": [[0, "rgba(25, 195, 217, 0.0)"], [1, "rgba(25, 195, 217, 0.18)"]],
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            for x_values, y_line, colour in (
                (history_x, y_values[:history_count], "rgba(43, 217, 138, 0.25)"),
                (future_x, y_values[history_count - 1 :], "rgba(25, 195, 217, 0.25)"),
            ):
                trajectory.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_line,
                        mode="lines",
                        line={"color": colour, "width": 10, "shape": "spline"},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        trajectory.add_trace(
            go.Scatter(
                x=history_x,
                y=y_values[:history_count],
                customdata=hover_data[:history_count],
                name=tr("Actual"),
                mode="lines+markers",
                line={"color": CRYSTAL_GREEN, "width": 3, "shape": "spline" if CRYSTAL_MODE else "linear"},
                marker={"size": 7, "color": CHART_SURFACE, "line": {"color": CRYSTAL_GREEN, "width": 2}},
                hovertemplate=f"<b>%{{x}}</b> · {tr('Actual')}<br>{hover_rows}<extra></extra>",
            )
        )
        trajectory.add_trace(
            go.Scatter(
                x=future_x,
                y=y_values[history_count - 1 :],
                customdata=hover_data[history_count - 1 :],
                name=tr("Forecast ({model})", model=forecast_model_label),
                mode="lines+markers",
                line={"color": CRYSTAL_CYAN, "width": 3, "dash": "dash", "shape": "spline" if CRYSTAL_MODE else "linear"},
                marker={"size": 7, "color": CRYSTAL_CYAN},
                hovertemplate=f"<b>%{{x}}</b> · {tr('Forecast')}<br>{hover_rows}<extra></extra>",
            )
        )
        trajectory.update_layout(
            template=PLOTLY_TEMPLATE,
            title={
                "text": tr(
                    "Population trajectory {start}–{end}", start=first_year, end=future_end
                ),
                "x": 0,
                "xanchor": "left",
                "y": 0.98,
            },
            height=400,
            margin={"l": 10, "r": 10, "t": 78, "b": 10},
            hovermode="closest",
            legend={"orientation": "h", "x": 1, "xanchor": "right", "y": -0.12},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel={"bgcolor": CHART_HOVER, "font_color": CHART_TEXT},
        )
        trajectory.update_xaxes(
            tickmode="array",
            tickvals=summary_years,
            showgrid=False,
            range=[summary_years[0] - 0.5, summary_years[-1] + 0.5],
        )
        y_low, y_high = float(np.nanmin(y_values)), float(np.nanmax(y_values))
        y_pad = (y_high - y_low) * 0.12 or abs(y_high) * 0.05 or 1.0
        trajectory.update_yaxes(
            tickformat=",.1f" if summary_measure == "density" else ",.0f",
            gridcolor=STREAMLIT_CHART_GRID,
            zeroline=False,
            range=[y_low - y_pad, y_high + y_pad],
        )
        st.plotly_chart(
            trajectory,
            width="stretch",
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, theme=CHART_THEME,
        )

    with glass(age_column):
        age_labels = ["0–4", "5–18", "19–25", "26–40", "41–60", "61–80+"]
        age_bins = [-1, 4, 18, 25, 40, 60, np.inf]
        summary_rows = cached_row_forecast(filtered_data, future_end, round_and_clip=False)
        summary_ages = pd.cut(
            pd.to_numeric(
                summary_rows["cocCode"].astype(str).str.extract(r"(\d+)", expand=False),
                errors="coerce",
            ),
            bins=age_bins,
            labels=age_labels,
        )
        end_column = f"{future_end}{MODEL_EXPORT_SUFFIXES[ribbon_model]}"
        age_structure = (
            pd.DataFrame(
                {
                    "Age group": summary_ages,
                    "start": summary_rows[str(last_actual_year)].astype(float),
                    "end": summary_rows[end_column].astype(float),
                }
            )
            .groupby("Age group", observed=False)[["start", "end"]]
            .sum()
            .reindex(age_labels)
        )
        age_shares = age_structure / age_structure.sum() * 100
        age_chart = go.Figure()
        for column, year, color in (
            ("start", last_actual_year, ATLAS_FOREST),
            ("end", future_end, ATLAS_GOLD),
        ):
            age_chart.add_trace(
                go.Bar(
                    y=age_labels,
                    x=age_shares[column],
                    customdata=age_structure[column],
                    name=str(year),
                    orientation="h",
                    marker={"color": color, "line": {"color": CHART_SURFACE, "width": 1}},
                    hovertemplate=(
                        f"<b>%{{y}}</b> · {year}<br>{tr('Share')}: %{{x:.1f}}%"
                        f"<br>{tr('Population')}: %{{customdata:,.0f}}<extra></extra>"
                    ),
                )
            )
        age_chart.update_layout(
            template=PLOTLY_TEMPLATE,
            title={
                "text": tr(
                    "Age structure: {start} vs {end}", start=last_actual_year, end=future_end
                ),
                "x": 0,
                "xanchor": "left",
                "y": 0.98,
            },
            barmode="group",
            bargap=0.25,
            height=400,
            margin={"l": 10, "r": 10, "t": 78, "b": 10},
            legend={
                "orientation": "h",
                "x": 1,
                "xanchor": "right",
                "y": 1.0,
                "yanchor": "bottom",
                "traceorder": "reversed",
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel={"bgcolor": CHART_HOVER, "font_color": CHART_TEXT},
        )
        age_chart.update_xaxes(
            title_text=tr("Share of population (%)"),
            ticksuffix="%",
            gridcolor=STREAMLIT_CHART_GRID,
            zeroline=False,
        )
        age_chart.update_yaxes(title_text=tr("Age group"))
        st.plotly_chart(age_chart, width="stretch", config={"displaylogo": False}, theme=CHART_THEME)

    st.markdown(
        '<p class="atlas-meta">'
        + tr(
            "Average annual growth {start}–{end}: {rate}.",
            start=last_actual_year,
            end=future_end,
            rate=f"{average_forecast_growth:+.2f}%",
        )
        + " "
        + tr(
            "{rows} data rows. Baseline: {baseline}. Source: DHIS2, Ministry of Health.",
            rows=f"{len(filtered_data):,}",
            baseline=forecast_input_label,
        )
        + (
            " "
            + tr(
                "Land area {area} km², measured from the map boundaries.",
                area=f"{ribbon_area_km2:,.0f}",
            )
            if ribbon_area_km2
            else ""
        )
        + "</p>",
        unsafe_allow_html=True,
    )

with overview_tab:
    if selected_model_option == ALL_MODELS_OPTION:
        best_model = validation_result.metrics.loc[selected_models, "MAPE (%)"].idxmin()
        st.subheader(":material/leaderboard: " + tr("Time-based validation model comparison"))
        st.caption(
            tr(
                "Models are trained on 2016–2022 and evaluated on the unseen 2023–2025 holdout "
                "period. The final 2026+ forecasts are then retrained on all 2016–2025 data."
            )
        )
        all_model_summary = validation_result.metrics.loc[list(MODEL_NAMES)].reset_index()
        all_model_summary["Model"] = all_model_summary["Model"].map(MODEL_LABELS)
        st.dataframe(
            tr_table(all_model_summary).style.format(
                tr_formats(
                    {
                        "MAE": "{:,.2f}",
                        "RMSE": "{:,.2f}",
                        "MAPE (%)": "{:.4f}",
                        "R²": "{:.4f}",
                        "Accuracy (%)": "{:.4f}",
                    }
                )
            ).highlight_min(subset=["MAPE (%)"], color=ATLAS_FOREST_SOFT),
            hide_index=True,
            width="stretch",
            height=143,
        )
        # st.caption(f"Best fit by MAPE: {MODEL_LABELS[best_model]}")
        st.markdown(
            '<hr style="border: none; border-top: 1px solid #D3DED5;">',
            unsafe_allow_html=True
        )
        # st.divider()
    # st.caption(f"Source: {source_name} · {len(data):,} total source rows")



with overview_tab:
    st.subheader(":material/fact_check: " + tr("Time-based validation: 2016–2022 train → 2023–2025 test"))
    st.caption(
        tr(
            "This chart measures performance on later historical years that each model did "
            "not see during fitting. Model selection uses holdout MAPE, not training fit."
        )
    )
    validation_chart = interactive_validation_figure(validation_result, selected_models)
    glass().plotly_chart(
        validation_chart,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }, theme=CHART_THEME,
    )
    validation_table = validation_result.metrics.loc[selected_models].reset_index()
    validation_table["Model"] = validation_table["Model"].map(MODEL_LABELS)
    st.dataframe(
        tr_table(validation_table).style.format(
            tr_formats(
                {
                    "MAE": "{:,.2f}",
                    "RMSE": "{:,.2f}",
                    "MAPE (%)": "{:.4f}",
                    "R²": "{:.4f}",
                    "Accuracy (%)": "{:.4f}",
                }
            )
        ).highlight_min(subset=["MAPE (%)"], color=ATLAS_FOREST_SOFT),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        tr(
            "Accuracy is displayed as 100 − MAPE for continuity with the original project; "
            "it is not classification accuracy."
        )
    )
    st.markdown(
        '<hr style="border: none; border-top: 1px solid #D3DED5;">',
        unsafe_allow_html=True,
    )
    st.subheader(":material/timeline: " + tr("Final fit and forecast: retrained on all 2016–2025 observations"))
    interactive_forecast_chart = interactive_forecast_figure(result, selected_models)
    live_results["forecast_plotly"] = slide_json(
        interactive_forecast_figure(
            result,
            selected_models,
            lao_presentation=True,
        )
    )
    glass().plotly_chart(
        interactive_forecast_chart,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }, theme=CHART_THEME,
    )
    st.markdown(
            '<hr style="border: none; border-top: 1px solid #D3DED5;">',
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
    #         '<hr style="border: none; border-top: 1px solid #D3DED5;">',
    #         unsafe_allow_html=True
    #     )
    # st.caption(
    #     "Accuracy is shown as 100 − MAPE, matching the notebook. It is an "
    #     "in-sample descriptive score, not a validation-set accuracy measure."
    # )

    st.subheader(":material/compare_arrows: " + tr("Actual versus predicted population (2016–2025) by model"))
    interactive_actual_predicted_chart = interactive_actual_vs_predicted_figure(
        result, selected_models
    )
    live_results["actual_predicted_plotly"] = slide_json(
        interactive_actual_vs_predicted_figure(
            result,
            selected_models,
            lao_presentation=True,
        )
    )
    glass().plotly_chart(
        interactive_actual_predicted_chart,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }, theme=CHART_THEME,
    )
    st.session_state["presentation_results"] = live_results

    st.markdown(
            '<hr style="border: none; border-top: 1px solid #D3DED5;">',
            unsafe_allow_html=True
        )
    st.subheader(":material/table_chart: " + tr("Total population forecast 2026–{end} by model", end=future_end))
    future_table = pd.DataFrame(
        {
            tr("Year"): future_years,
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
        '<hr style="border: none; border-top: 1px solid #D3DED5;">',
        unsafe_allow_html=True,
    )
    st.subheader(":material/trending_up: " + tr("Growth rates and averages"))
    growth_rate_table, average_growth_rates = population_growth_rate_summary(
        result, selected_models
    )
    average_columns = st.columns(len(average_growth_rates))
    for column, (label, average_rate) in zip(
        average_columns, average_growth_rates.items()
    ):
        column.metric(tr("{label} average", label=tr(label)), f"{average_rate:+.2f}%")

    growth_rate_table = tr_table(growth_rate_table)
    growth_columns = [column for column in growth_rate_table if column != tr("Year")]
    st.dataframe(
        growth_rate_table.style.format(
            {column: "{:+.2f}%" for column in growth_columns}, na_rep="—"
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        tr(
            "Each value is the year-over-year growth rate. Actual values run to 2025; model "
            "forecasts continue to the selected end year."
        )
    )

with pyramid_tab:
    st.subheader(":material/groups: " + tr("Population pyramid by district"))
    st.caption(
        tr(
            "Compare male and female populations in the age groups 0–4, 5–18, 19–25, 26–40, "
            "41–60, and 61–80+."
        )
    )
    if "ouName" not in data.columns:
        st.error(tr("The uploaded data must contain an ouName district column."))
    else:
        pyramid_districts = sorted(data["ouName"].dropna().astype(str).unique())
        district_column, year_column, age_display_column = st.columns(3)
        selected_pyramid_district = district_column.selectbox(
            tr("District"),
            ["All", *pyramid_districts],
            format_func=lambda district, all_label=tr("All districts"): (
                all_label if district == "All" else district
            ),
            key=lang_key("population_pyramid_district"),
            help=tr("Choose All districts for the nationwide pyramid, or select one district."),
        )
        remember("population_pyramid_district", selected_pyramid_district)
        selected_pyramid_year = year_column.selectbox(
            tr("Year"),
            [*HISTORICAL_YEARS, *future_years],
            index=len(HISTORICAL_YEARS) - 1,
            key="population_pyramid_year",
        )
        selected_age_display = age_display_column.selectbox(
            tr("Age display"),
            ["By age group", "By single age"],
            format_func={option: tr(option) for option in ("By age group", "By single age")}.get,
            key=lang_key("population_pyramid_age_display"),
        )
        remember("population_pyramid_age_display", selected_age_display)
        # Use the same forecast input as the overview (corrected or original),
        # but not the sidebar filters: the pyramid has its own district selector.
        pyramid_source = (
            corrected_data
            if forecast_input_option == CORRECTED_INPUT_OPTION
            else data
        )
        pyramid_data = pyramid_source
        pyramid_year_label = str(selected_pyramid_year)
        is_future_pyramid = selected_pyramid_year > HISTORICAL_YEARS[-1]
        if is_future_pyramid:
            pyramid_model = (
                presentation_best_model
                if selected_model_option == ALL_MODELS_OPTION
                else selected_model_option
            )
            pyramid_predictions = cached_row_forecast(pyramid_source, future_end)
            prediction_column = (
                f"{selected_pyramid_year}{MODEL_EXPORT_SUFFIXES[pyramid_model]}"
            )
            pyramid_data = pyramid_predictions.copy()
            pyramid_data[str(selected_pyramid_year)] = pyramid_data[prediction_column]
            # Raw model values (not rounded, negatives kept) for the totals, so
            # they equal the overview forecast for the same district or nation.
            raw_predictions = cached_row_forecast(
                pyramid_source, future_end, round_and_clip=False
            )
            if selected_pyramid_district != "All":
                raw_predictions = raw_predictions[
                    raw_predictions["ouName"].astype(str).eq(selected_pyramid_district)
                ]
            raw_sex_codes = raw_predictions["sexCode"].astype(str).str.strip().str.upper()
            raw_male = raw_predictions.loc[raw_sex_codes.eq("M"), prediction_column].sum()
            raw_female = raw_predictions.loc[raw_sex_codes.eq("F"), prediction_column].sum()
            pyramid_year_label = tr(
                "{year} forecast, {model}",
                year=selected_pyramid_year,
                model=MODEL_LABELS[pyramid_model],
            )
            if selected_model_option == ALL_MODELS_OPTION:
                st.info(
                    tr(
                        "All models is selected, so this future pyramid uses the current "
                        "lowest-MAPE model: {model}.",
                        model=MODEL_LABELS[pyramid_model],
                    )
                )
            else:
                st.caption(
                    tr(
                        "The {year} pyramid uses the selected {model} forecast.",
                        year=selected_pyramid_year,
                        model=MODEL_LABELS[pyramid_model],
                    )
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
            if is_future_pyramid:
                combined_total = int(round(raw_male + raw_female))
                pyramid_male_total = int(round(raw_male))
                pyramid_female_total = combined_total - pyramid_male_total
            else:
                combined_total = pyramid_male_total + pyramid_female_total
            male_total_column, female_total_column, combined_total_column = st.columns(3)
            male_total_column.metric(tr("Male population"), f"{pyramid_male_total:,}")
            female_total_column.metric(tr("Female population"), f"{pyramid_female_total:,}")
            combined_total_column.metric(tr("Total population"), f"{combined_total:,}")
            st.caption(tr("Data source: **{source}**.", source=forecast_input_label))
            if is_future_pyramid:
                st.caption(
                    tr(
                        "The totals are the model's forecast, the same as the overview. The "
                        "bars show each district–age–sex series rounded to whole people, with "
                        "series forecast below zero drawn as 0, so the bars can add up to "
                        "slightly more than the total."
                    )
                )
            glass().plotly_chart(pyramid_chart, width="stretch", config={"displaylogo": False}, theme=CHART_THEME)

    st.divider()
    st.subheader(":material/map: " + tr("Population map"))
    st.caption(
        tr(
            "Total population and population density by province (level 2) or district "
            "(level 3). Hover over an area to see its name, population and density."
        )
    )
    if not BOUNDARY_FILE.exists():
        st.warning(tr("Map boundary file not found: {path}", path=BOUNDARY_FILE))
    elif not {"ouID", "ouName"}.issubset(data.columns):
        st.error(tr("The map requires ouID and ouName columns in the data."))
    else:
        map_level_column, map_year_column = st.columns(2)
        map_level_key = lang_key("population_map_level")
        selected_map_level = remember(
            "population_map_level",
            map_level_column.segmented_control(
                tr("Administrative level"),
                list(MAP_LEVEL_OPTIONS),
                required=True,
                format_func={option: tr(option) for option in MAP_LEVEL_OPTIONS}.get,
                default=None if map_level_key in st.session_state else "Province (level 2)",
                key=map_level_key,
            )
            or "Province (level 2)",
        )
        selected_map_year = map_year_column.selectbox(
            tr("Year"),
            [*HISTORICAL_YEARS, *future_years],
            index=len(HISTORICAL_YEARS) - 1,
            key="population_map_year",
        )
        map_source = (
            corrected_data
            if forecast_input_option == CORRECTED_INPUT_OPTION
            else data
        )
        if selected_map_year > HISTORICAL_YEARS[-1]:
            map_model = (
                presentation_best_model
                if selected_model_option == ALL_MODELS_OPTION
                else selected_model_option
            )
            # Raw model values, so the map adds up to the overview forecast.
            map_values = cached_row_forecast(map_source, future_end, round_and_clip=False)
            map_value_column = f"{selected_map_year}{MODEL_EXPORT_SUFFIXES[map_model]}"
            map_year_label = tr(
                "{year} forecast, {model}",
                year=selected_map_year,
                model=MODEL_LABELS[map_model],
            )
        else:
            map_values = map_source
            map_value_column = str(selected_map_year)
            map_year_label = str(selected_map_year)

        boundaries = map_boundaries()
        map_names = boundaries["names"]
        district_table = district_totals(map_values, map_value_column)
        if MAP_LEVEL_OPTIONS[selected_map_level] == "provinces":
            map_table = province_totals(district_table, map_names)
        else:
            map_table = district_table.merge(
                map_names[["ouID", "name", "displayName", "area_km2"]], on="ouID", how="left"
            )
            map_table["name"] = map_table["name"].fillna(map_table["ouName"])
            map_table["displayName"] = map_table["displayName"].fillna(map_table["ouName"])

        missing_areas = int(
            (~map_table["ouID"].isin(
                [feature["id"] for feature in boundaries[MAP_LEVEL_OPTIONS[selected_map_level]]["features"]]
            )).sum()
        )
        is_province_map = MAP_LEVEL_OPTIONS[selected_map_level] == "provinces"
        map_geojson = boundaries[MAP_LEVEL_OPTIONS[selected_map_level]]
        # Density = people per km² of land, using each area's size measured on
        # its boundary (population_map.load_boundaries).
        map_table["Density"] = map_table["Population"] / map_table["area_km2"]
        total_population = map_table["Population"].sum()
        nationwide_density = total_population / map_table["area_km2"].sum()
        map_config = {"displaylogo": False, "scrollZoom": True}
        population_column, density_column = st.columns(2)
        glass(population_column).plotly_chart(
            population_map_figure(
                map_table,
                map_geojson,
                tr(
                    "Population by province — {year}"
                    if is_province_map
                    else "Population by district — {year}",
                    year=map_year_label,
                ),
                PLOTLY_TEMPLATE,
                height=620,
                basemap=show_basemap,
                value_label=tr("Population"),
                hover_lines=[(tr("People per km²"), "Density", ",.1f")],
            ),
            width="stretch",
            config=map_config, theme=CHART_THEME,
        )
        glass(density_column).plotly_chart(
            population_map_figure(
                map_table,
                map_geojson,
                tr(
                    "Population density by province — {year}"
                    if is_province_map
                    else "Population density by district — {year}",
                    year=map_year_label,
                ),
                PLOTLY_TEMPLATE,
                height=620,
                basemap=show_basemap,
                value_label=tr("People per km²"),
                value_column="Density",
                value_format=",.1f",
                colorscale=DENSITY_COLORSCALE,
                hover_lines=[
                    (tr("Population"), "Population", ",.0f"),
                    (tr("Area (km²)"), "area_km2", ",.0f"),
                ],
            ),
            width="stretch",
            config=map_config, theme=CHART_THEME,
        )
        st.caption(
            tr(
                "Data source: **{source}**. Nationwide total: **{total}** people, "
                "**{density}** people per km². Colours use a log scale so smaller areas can "
                "still be told apart.",
                source=forecast_input_label,
                total=f"{total_population:,.0f}",
                density=f"{nationwide_density:,.1f}",
            )
            + " "
            + tr(
                "Density divides each area's population by its land area, measured from the "
                "map boundaries."
            )
            + (
                " " + tr("{count} area(s) in the data have no boundary on the map.", count=missing_areas)
                if missing_areas
                else ""
            )
        )
        with st.expander(
            tr("Population table by province" if is_province_map else "Population table by district")
        ):
            st.dataframe(
                map_table[["name", "displayName", "Population", "area_km2", "Density"]]
                .assign(
                    Population=lambda table: table["Population"].round().astype("int64"),
                    area_km2=lambda table: table["area_km2"].round().astype("int64"),
                    Density=lambda table: table["Density"].round(1),
                )
                .sort_values("Density", ascending=False)
                .rename(
                    columns={
                        "name": tr("Name (English)"),
                        "displayName": tr("Name (Lao)"),
                        "Population": tr("Population"),
                        "area_km2": tr("Area (km²)"),
                        "Density": tr("People per km²"),
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    tr("Population"): st.column_config.NumberColumn(format="localized"),
                    tr("Area (km²)"): st.column_config.NumberColumn(format="localized"),
                    tr("People per km²"): st.column_config.NumberColumn(format="%.1f"),
                },
            )

with step1_source:
    st.subheader(":material/table_view: " + tr("Forecast source data"))
    st.caption(
        tr(
            "Forecasts currently use **{source}**. The original source is preserved and "
            "remains available for comparison.",
            source=forecast_input_label,
        )
    )
    original_source_tab, corrected_source_tab = st.tabs(
        [tr("Original filtered data"), tr("Corrected filtered data")]
    )
    with original_source_tab:
        st.dataframe(
            filtered_original_data, hide_index=True, width="stretch", height=420
        )
    with corrected_source_tab:
        st.dataframe(
            filtered_corrected_data, hide_index=True, width="stretch", height=420
        )

with data_tab:
    st.subheader(":material/auto_fix_high: " + tr("Conservative outlier correction"))
    st.caption(
        tr(
            "Automatic correction uses the selected robust-score threshold of {threshold}. A "
            "value is changed only when a cohort-aware historical anomaly is independently "
            "confirmed by a district-share, sex-ratio, or adjacent-age check.",
            threshold=f"{correction_threshold:g}",
        )
    )
    with st.expander(tr("Outlier correction steps"), expanded=True):
        st.markdown(block("correction_steps", current_language()))

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
    correction_metrics[0].metric(tr("Corrected observations"), f"{len(visible_corrections):,}")
    correction_metrics[1].metric(tr("Affected series"), f"{corrected_series:,}")
    correction_metrics[2].metric(tr("Share of historical cells"), f"{corrected_percent:.3f}%")
    correction_metrics[3].metric(
        tr("Net historical-cell adjustment"),
        f"{visible_corrections['Difference'].sum():+,.0f}"
        if not visible_corrections.empty
        else "0",
    )

    if visible_corrections.empty:
        st.success(tr("No observations in the current filters met the conservative correction rule."))
    else:
        st.bar_chart(
            visible_corrections.groupby("Year").size().rename(tr("Corrections")),
            height=220,
        )
        st.dataframe(
            tr_table(visible_corrections, ("Supporting check", "Correction method")).style.format(
                tr_formats(
                    {
                        "Original value": "{:,.0f}",
                        "Corrected value": "{:,.0f}",
                        "Difference": "{:+,.0f}",
                        "Change (%)": "{:+.3f}%",
                        "Temporal robust score": "{:+.2f}",
                        "Supporting robust score": "{:+.2f}",
                    }
                )
            ),
            hide_index=True,
            width="stretch",
            height=420,
        )

    correction_download, corrected_data_download = st.columns(2)
    correction_download.download_button(
        tr("Download correction audit"),
        visible_corrections.to_csv(index=False).encode("utf-8-sig"),
        file_name="population_outlier_correction_audit.csv",
        mime="text/csv",
        width="stretch",
    )
    corrected_data_download.download_button(
        tr("Download corrected source data"),
        dataframe_to_csv_bytes(corrected_data),
        file_name="dataDistrict_Corrected.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("#### :material/compare: " + tr("Forecast impact: original versus corrected data"))
    st.caption(
        tr(
            "This comparison uses the current district, age, and sex filters. It shows how the "
            "conservative corrections affect each of the existing three models."
        )
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
        tr_table(forecast_impact).style.format(
            tr_formats(
                {
                    "Original forecast": "{:,.0f}",
                    "Corrected-data forecast": "{:,.0f}",
                    "Difference": "{:+,.0f}",
                    "Change (%)": "{:+.4f}%",
                }
            )
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader(":material/troubleshoot: " + tr("Nationwide outlier review"))
    st.caption(
        tr(
            "Checks always use the complete nationwide dataset as their reference. Sidebar "
            "selections and the year selection below only filter the displayed flags. A flag "
            "indicates a record to review; it is not automatically an error."
        )
    )
    control_column, year_column = st.columns([1, 3])
    control_column.metric(
        tr("Active robust-score threshold"), f"{correction_threshold:g}"
    )
    selected_outlier_years = year_column.multiselect(
        tr("Years to display"),
        HISTORICAL_YEARS,
        default=HISTORICAL_YEARS,
        key="outlier_years",
    )
    if not selected_outlier_years:
        st.info(tr("Select at least one year to display outlier results."))
        visible_outlier_results = {
            "nationwide": pd.DataFrame(),
            "historical": pd.DataFrame(),
            "demographic": pd.DataFrame(),
        }
    else:
        outlier_results = correction_result.outliers
        nationwide_check, historical_check, demographic_check = st.tabs(
            [
                tr("Nationwide district comparison"),
                tr("Historical change"),
                tr("Demographic consistency"),
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
    st.subheader(":material/download: " + tr("Downloads and exports"))
    st.write(
        tr(
            "Download the population forecasts as CSV or HTML. The files include every "
            "forecast from Linear Regression, Polynomial Regression and Exponential Growth."
        )
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
    with st.expander(":material/table_view: " + tr("Preview the row-level forecasts")):
        st.dataframe(
            predicted[preview_columns], hide_index=True, width="stretch", height=480
        )

    csv_column, html_column = st.columns(2)
    csv_column.download_button(
        tr("Download all-model CSV"),
        dataframe_to_csv_bytes(predicted),
        file_name="dataDistrict_Predicted_All_Models.csv",
        mime="text/csv",
        width="stretch",
    )
    html_column.download_button(
        tr("Download Polynomial HTML"),
        dataframe_to_html_bytes(html_table),
        file_name="dataDistrict_Predicted_Polynomial.html",
        mime="text/html",
        width="stretch",
    )

with methodology_tab:
    st.subheader(":material/functions: " + tr("Forecasting models"))
    st.caption(
        tr(
            "Each model is fitted to the yearly population totals, and separately to every "
            "district–age–sex series for the row-level forecasts."
        )
    )
    model_cards = (
        ("Linear", "show_chart", r"\hat{y}_t = \beta_0 + \beta_1 t",
         "A straight line: about the same number of people is added every year."),
        ("Polynomial", "ssid_chart", r"\hat{y}_t = \beta_0 + \beta_1 (t-\bar{t}) + \beta_2 (t-\bar{t})^2",
         "A degree-2 curve on centred years, so growth can speed up or slow down."),
        ("Exponential", "trending_up", r"\ln \hat{y}_t = \beta_0 + \beta_1 t",
         "A constant growth rate: about the same percentage is added every year."),
    )
    for column, (name, icon, formula, description) in zip(st.columns(3), model_cards):
        with glass(column):
            st.markdown(f":material/{icon}: **{MODEL_LABELS[name]}**")
            st.latex(formula)
            st.caption(tr(description))
            st.markdown(
                tr(
                    "Test-year MAPE: **{mape}**",
                    mape=f"{validation_result.metrics.loc[name, 'MAPE (%)']:.4f}%",
                )
            )
    with glass():
        card_title("schedule", tr("How the models are tested and used"))
        st.markdown(block("validation_scheme", current_language()))
    st.info(
        tr(
            "Course purpose: help every student understand the end-to-end AI workflow and "
            "apply it to a practical task that addresses a real-world problem."
        )
    )
    st.subheader(":material/account_tree: " + tr("CRISP-DM project workflow"))
    st.markdown(block("crisp_dm", current_language()))

    st.subheader(":material/lan: " + tr("Application architecture"))
    st.code(block("architecture", current_language()), language="text")

    st.subheader(":material/warning: " + tr("Limitations and responsible use"))
    st.markdown(block("limitations", current_language()))

    st.subheader(":material/menu_book: " + tr("Method and software references"))
    st.markdown(block("references", current_language()))

# ------------------------------------------------------------------ shared summary tables
name_column = "displayName" if current_language() == "lo" else "name"
summary_rows_raw = cached_row_forecast(filtered_data, future_end, round_and_clip=False)
summary_end_column = f"{future_end}{MODEL_EXPORT_SUFFIXES[ribbon_model]}"
province_summary = pd.DataFrame()
district_summary = pd.DataFrame()
if BOUNDARY_FILE.exists():
    summary_names = map_boundaries()["names"]
    start_by_district = district_totals(summary_rows_raw, str(last_actual_year))
    end_by_district = district_totals(summary_rows_raw, summary_end_column)
    province_start = province_totals(start_by_district, summary_names)
    province_end = province_totals(end_by_district, summary_names)
    province_summary = province_start[["name", "displayName", "area_km2", "Population"]].rename(
        columns={"Population": "start"}
    )
    province_summary["end"] = province_end["Population"].to_numpy()
    province_summary = province_summary[province_summary["start"] > 0].copy()
    province_summary["change"] = province_summary["end"] - province_summary["start"]
    province_summary["change_pct"] = province_summary["change"] / province_summary["start"] * 100
    province_summary["density_end"] = province_summary["end"] / province_summary["area_km2"]
    district_summary = end_by_district.merge(
        summary_names[["ouID", "name", "displayName", "area_km2"]], on="ouID", how="left"
    )
    district_summary["density_end"] = district_summary["Population"] / district_summary["area_km2"]

year_summary = pd.DataFrame(
    {
        "year": summary_years,
        "phase": [tr("Actual")] * history_count + [tr("Forecast")] * len(future_years),
        "population": summary_population,
        "growth": summary_growth,
        "density": summary_density,
    }
)
year_summary_display = year_summary.assign(
    growth=[("—" if np.isnan(value) else f"{value:+.2f}%") for value in summary_growth]
).rename(
    columns={
        "year": tr("Year"),
        "phase": tr("Type"),
        "population": tr("Population"),
        "growth": tr("Annual growth (%)"),
        "density": tr("People per km²"),
    }
)
province_summary_display = (
    province_summary.sort_values("change_pct", ascending=False)
    .rename(
        columns={
            name_column: tr("Province"),
            "start": tr("Population {year}", year=last_actual_year),
            "end": tr("Forecast {year}", year=future_end),
            "change": tr("Change"),
            "change_pct": tr("Change (%)"),
            "density_end": tr("People per km² ({year})", year=future_end),
            "area_km2": tr("Area (km²)"),
        }
    )
    .drop(columns=["displayName" if name_column == "name" else "name"])
    if not province_summary.empty
    else pd.DataFrame()
)

# ------------------------------------------------------------------ step 3: forecast vs past trend
with step3_dashboard:
    st.subheader(":material/compare_arrows: " + tr("Does the forecast follow the past trend?"))
    st.caption(
        tr(
            "Actual {start}–{last} and forecast {first}–{end} ({model}), using the baseline "
            "data and sidebar filters. Hatched bars and dashed lines are forecasts.",
            start=HISTORICAL_YEARS[0],
            last=last_actual_year,
            first=future_years[0],
            end=future_end,
            model=forecast_model_label,
        )
    )
    trend_year_column, trend_age_column = st.columns([1.35, 1])
    trend_sex_column = "sexName" if "sexName" in summary_rows_raw else None
    with glass(trend_year_column):
        card_title(
            "stacked_bar_chart",
            tr("Population by year and sex, {start}–{end}", start=HISTORICAL_YEARS[0], end=future_end),
        )
        trend_share = sex_view_switch("trend_sex_chart_view")
        forecast_columns = [f"{year}{MODEL_EXPORT_SUFFIXES[ribbon_model]}" for year in future_years]
        trend_by_sex = summary_rows_raw.groupby(trend_sex_column)[
            [str(year) for year in HISTORICAL_YEARS] + forecast_columns
        ].sum().T
        trend_by_sex.index = [*HISTORICAL_YEARS, *future_years]
        st.plotly_chart(
            crystal(sex_year_figure(trend_by_sex, trend_share, future_years), 400),
            width="stretch",
            config=CRYSTAL_CHART_CONFIG, theme=CHART_THEME,
        )
    with glass(trend_age_column):
        card_title(
            "area_chart",
            tr("Population by single age: {start} vs {end}", start=last_actual_year, end=future_end),
        )
        trend_ages = pd.to_numeric(
            summary_rows_raw["cocCode"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        trend_by_age = (
            summary_rows_raw.assign(age=trend_ages)
            .groupby(["age", trend_sex_column])[[str(last_actual_year), summary_end_column]]
            .sum()
        )
        trend_age_figure = go.Figure()
        for sex_name, color in (("Male", ATLAS_MEKONG), ("Female", ATLAS_MADDER)):
            if sex_name not in trend_by_age.index.get_level_values(1):
                continue
            sex_rows = trend_by_age.xs(sex_name, level=1).sort_index()
            for column, year, dash, width in (
                (str(last_actual_year), last_actual_year, "solid", 2.5),
                (summary_end_column, future_end, "dash", 2.5),
            ):
                label = f"{tr(sex_name)} {year}"
                trend_age_figure.add_trace(
                    go.Scatter(
                        x=sex_rows.index,
                        y=sex_rows[column],
                        name=label,
                        mode="lines",
                        line={"color": color, "width": width, "dash": dash},
                        opacity=1 if dash == "solid" else 0.85,
                        hovertemplate=(
                            f"{tr('Age')} %{{x}} · {label}<br>{tr('Population')}: %{{y:,.0f}}<extra></extra>"
                        ),
                    )
                )
        trend_age_figure.update_layout(hovermode="x unified")
        trend_age_figure.update_xaxes(title_text=tr("Age"))
        trend_age_figure.update_yaxes(tickformat=",.0f")
        st.plotly_chart(
            crystal(trend_age_figure, 448),
            width="stretch",
            config=CRYSTAL_CHART_CONFIG, theme=CHART_THEME,
        )

# ------------------------------------------------------------------ step 3: key findings
with step3_findings:
    st.subheader(":material/lightbulb: " + tr("Key findings"))
    other_mapes = ", ".join(
        f"{MODEL_LABELS[name]} {validation_result.metrics.loc[name, 'MAPE (%)']:.4f}%"
        for name in MODEL_NAMES
        if name != ribbon_model
    )
    older_share_start = float(age_shares.loc[["41–60", "61–80+"], "start"].sum())
    older_share_end = float(age_shares.loc[["41–60", "61–80+"], "end"].sum())
    findings = [
        (
            "trending_up",
            tr("Population keeps growing"),
            tr(
                "From {start} in {start_year} to {end} in {end_year} ({change}), about {rate} a year.",
                start=f"{last_actual_population:,.0f}",
                start_year=last_actual_year,
                end=f"{end_population:,.0f}",
                end_year=future_end,
                change=f"{(end_population / last_actual_population - 1) * 100:+.1f}%",
                rate=f"{average_forecast_growth:+.2f}%",
            ),
        ),
        (
            "verified",
            tr("Most accurate model: {model}", model=forecast_model_label),
            tr(
                "Test-year MAPE {mape}. Other models: {others}.",
                mape=f"{ribbon_mape:.4f}%",
                others=other_mapes,
            ),
        ),
        (
            "elderly",
            tr("The population is getting older"),
            tr(
                "People aged 41 and over: {start} of the population in {start_year}, {end} in {end_year}.",
                start=f"{older_share_start:.1f}%",
                start_year=last_actual_year,
                end=f"{older_share_end:.1f}%",
                end_year=future_end,
            ),
        ),
    ]
    if ribbon_area_km2:
        findings.append(
            (
                "landscape",
                tr("Density rises"),
                tr(
                    "From {start} to {end} people per km² between {start_year} and {end_year}.",
                    start=f"{summary_density[history_count - 1]:,.1f}",
                    end=f"{summary_density[-1]:,.1f}",
                    start_year=last_actual_year,
                    end_year=future_end,
                ),
            )
        )
    if not province_summary.empty:
        fastest = province_summary.nlargest(3, "change_pct")
        findings.append(
            (
                "location_city",
                tr("Fastest-growing provinces"),
                ", ".join(
                    f"{row[name_column]} ({row['change_pct']:+.1f}%)"
                    for _, row in fastest.iterrows()
                ),
            )
        )
    corrected_count = len(correction_result.corrections)
    findings.append(
        (
            "rule",
            tr("Outliers corrected"),
            tr(
                "{count} values ({share}) were corrected; the original values are kept in the audit log.",
                count=f"{corrected_count:,}",
                share=f"{corrected_count / (len(data) * len(HISTORICAL_COLUMNS)):.2%}",
            ),
        )
    )
    for row_start in range(0, len(findings), 3):
        columns = st.columns(3)
        for column, (icon, title, text) in zip(columns, findings[row_start : row_start + 3]):
            with glass(column):
                st.markdown(f":material/{icon}: **{title}**")
                st.markdown(text)

    st.subheader(":material/table_chart: " + tr("Summary statistics"))
    by_year_tab, by_province_tab = st.tabs(
        [":material/calendar_month: " + tr("By year"), ":material/location_city: " + tr("By province")]
    )
    with by_year_tab:
        st.dataframe(
            year_summary_display,
            hide_index=True,
            width="stretch",
            column_config={
                tr("Year"): st.column_config.NumberColumn(format="%d"),
                tr("Population"): st.column_config.NumberColumn(format="localized"),
                tr("People per km²"): st.column_config.NumberColumn(format="%.1f"),
            },
        )
    with by_province_tab:
        if province_summary_display.empty:
            st.info(tr("Map boundary file not found: {path}", path=BOUNDARY_FILE))
        else:
            st.dataframe(
                province_summary_display,
                hide_index=True,
                width="stretch",
                column_config={
                    tr("Population {year}", year=last_actual_year): st.column_config.NumberColumn(format="localized"),
                    tr("Forecast {year}", year=future_end): st.column_config.NumberColumn(format="localized"),
                    tr("Change"): st.column_config.NumberColumn(format="%+.0f"),
                    tr("Change (%)"): st.column_config.NumberColumn(format="%+.2f%%"),
                    tr("People per km² ({year})", year=future_end): st.column_config.NumberColumn(format="%.1f"),
                    tr("Area (km²)"): st.column_config.NumberColumn(format="localized"),
                },
            )

# ------------------------------------------------------------------ step 3: more downloads
with export_tab:
    st.markdown(":material/folder_zip: **" + tr("Summary and audit files") + "**")
    summary_downloads = st.columns(4)
    summary_downloads[0].download_button(
        ":material/calendar_month: " + tr("Summary by year (CSV)"),
        year_summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="population_summary_by_year.csv",
        mime="text/csv",
        width="stretch",
    )
    summary_downloads[1].download_button(
        ":material/location_city: " + tr("Summary by province (CSV)"),
        province_summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="population_summary_by_province.csv",
        mime="text/csv",
        width="stretch",
        disabled=province_summary.empty,
    )
    summary_downloads[2].download_button(
        ":material/rule: " + tr("Download correction audit"),
        correction_result.corrections.to_csv(index=False).encode("utf-8-sig"),
        file_name="population_outlier_correction_audit_all.csv",
        mime="text/csv",
        width="stretch",
        key="summary_audit_download",
    )
    summary_downloads[3].download_button(
        ":material/database: " + tr("Download corrected source data"),
        dataframe_to_csv_bytes(corrected_data),
        file_name="dataDistrict_Corrected.csv",
        mime="text/csv",
        width="stretch",
        key="summary_corrected_download",
    )

with step3_info:
    with st.expander(":material/info: " + tr("About this forecast: limitations and references")):
        st.markdown("#### :material/warning: " + tr("Use responsibly"))
        st.markdown(block("use_responsibly", current_language()))
        st.markdown("#### :material/rule: " + tr("Limitations"))
        st.markdown(block("limitations", current_language()))
        st.caption(
            tr(
                "Areas and densities use land areas measured from the DHIS2 map boundaries, "
                "which total about 2% less than the official 236,800 km²."
            )
        )
        st.markdown("#### :material/menu_book: " + tr("References"))
        st.markdown(block("references", current_language()))

# ------------------------------------------------------------------ step 2: further analysis
with analysis_tab:
    st.subheader(":material/analytics: " + tr("Further analysis"))
    growth_column, models_column = st.columns(2)
    with growth_column:
        with glass():
            card_title(
                "location_city",
                tr("Population change by province, {start}–{end}", start=last_actual_year, end=future_end),
                tr("Forecast with {model}.", model=forecast_model_label),
            )
            if province_summary.empty:
                st.info(tr("Map boundary file not found: {path}", path=BOUNDARY_FILE))
            else:
                ranked = province_summary.sort_values("change_pct")
                province_figure = go.Figure(
                    go.Bar(
                        x=ranked["change_pct"],
                        y=ranked[name_column],
                        orientation="h",
                        customdata=ranked[["start", "end"]].to_numpy(),
                        marker={
                            "color": ranked["change_pct"],
                            "colorscale": [[0, "rgba(184,144,47,0.45)"], [1, "rgba(184,144,47,0.95)"]],
                            "line": {"color": CHART_SURFACE, "width": 1.2},
                        },
                        text=[f"{value:+.1f}%" for value in ranked["change_pct"]],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate=(
                            f"<b>%{{y}}</b><br>{last_actual_year}: %{{customdata[0]:,.0f}}"
                            f"<br>{future_end}: %{{customdata[1]:,.0f}}<br>{tr('Change')}: %{{x:+.2f}}%<extra></extra>"
                        ),
                    )
                )
                province_figure.update_xaxes(ticksuffix="%")
                st.plotly_chart(crystal(province_figure, 560), width="stretch", config=CRYSTAL_CHART_CONFIG, theme=CHART_THEME)
    with models_column:
        with glass():
            card_title(
                "stacked_line_chart",
                tr("The three models compared, {start}–{end}", start=HISTORICAL_YEARS[0], end=future_end),
                tr("How far the models disagree grows with the forecast horizon."),
            )
            models_figure = go.Figure()
            models_figure.add_trace(
                go.Scatter(
                    x=HISTORICAL_YEARS,
                    y=result.actual,
                    name=tr("Actual"),
                    mode="lines+markers",
                    line={"color": CHART_TEXT, "width": 3, "shape": "spline"},
                    marker={"size": 6},
                    hovertemplate=f"<b>%{{x}}</b> · {tr('Actual')}<br>%{{y:,.0f}}<extra></extra>",
                )
            )
            dash_styles = {"Linear": "dash", "Polynomial": "dashdot", "Exponential": "dot"}
            for name in MODEL_NAMES:
                models_figure.add_trace(
                    go.Scatter(
                        x=[HISTORICAL_YEARS[-1], *future_years],
                        y=[result.actual[-1], *result.future[name]],
                        name=MODEL_LABELS[name],
                        mode="lines+markers",
                        line={
                            "color": MODEL_STYLES[name]["color"],
                            "width": 3.5 if name == ribbon_model else 2,
                            "dash": dash_styles[name],
                            "shape": "spline",
                        },
                        marker={"size": 6},
                        opacity=1 if name == ribbon_model else 0.7,
                        hovertemplate=f"<b>%{{x}}</b> · {MODEL_LABELS[name]}<br>%{{y:,.0f}}<extra></extra>",
                    )
                )
            models_figure.add_vrect(
                x0=HISTORICAL_YEARS[-1], x1=future_end, fillcolor="rgba(184,144,47,0.10)", line_width=0, layer="below"
            )
            models_figure.update_yaxes(tickformat=",.0f")
            st.plotly_chart(crystal(models_figure, 560), width="stretch", config=CRYSTAL_CHART_CONFIG, theme=CHART_THEME)
    if not district_summary.empty:
        with glass():
            card_title(
                "density_medium",
                tr("Most densely populated districts, {year}", year=future_end),
                tr("People per km², forecast with {model}.", model=forecast_model_label),
            )
            densest = district_summary.nlargest(12, "density_end").sort_values("density_end")
            density_figure = go.Figure(
                go.Bar(
                    x=densest["density_end"],
                    y=densest[name_column].fillna(densest["ouName"]),
                    orientation="h",
                    customdata=densest[["Population", "area_km2"]].to_numpy(),
                    marker={
                        "color": densest["density_end"],
                        "colorscale": [[0, "rgba(29,106,174,0.45)"], [1, "rgba(14,63,110,0.95)"]],
                        "line": {"color": CHART_SURFACE, "width": 1.2},
                    },
                    text=[f"{value:,.0f}" for value in densest["density_end"]],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        f"<b>%{{y}}</b><br>{tr('People per km²')}: %{{x:,.1f}}"
                        f"<br>{tr('Population')}: %{{customdata[0]:,.0f}}"
                        f"<br>{tr('Area (km²)')}: %{{customdata[1]:,.0f}}<extra></extra>"
                    ),
                )
            )
            density_figure.update_xaxes(tickformat=",.0f")
            st.plotly_chart(crystal(density_figure, 440), width="stretch", config=CRYSTAL_CHART_CONFIG, theme=CHART_THEME)

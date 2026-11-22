"""Choropleth maps of population and population density by province (level 2)
and district (level 3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COUNTRY_LEVEL = 1
PROVINCE_LEVEL = 2
DISTRICT_LEVEL = 3

# Single-hue sequential green ramp, light (small population) to dark (large).
MAP_COLORSCALE = [
    [0.0, "#E3F2E6"],
    [0.2, "#B7DEC2"],
    [0.4, "#84C59B"],
    [0.6, "#4FA674"],
    [0.8, "#237F4F"],
    [1.0, "#0B4F30"],
]

# Single-hue sequential blue ramp for population density, so the density map
# reads as a different measure from the green population map beside it.
DENSITY_COLORSCALE = [
    [0.0, "#DCEAF5"],
    [0.2, "#AECBE6"],
    [0.4, "#7AA8D4"],
    [0.6, "#4A84BE"],
    [0.8, "#1D6AAE"],
    [1.0, "#0E3F6E"],
]

EARTH_RADIUS_KM = 6378.137

# About 300 m at Laos' latitude: plenty for a country-wide map and it cuts the
# 31 MB source file down to a size the browser can redraw quickly.
SIMPLIFY_TOLERANCE = 0.003


def _simplify_ring(ring: list, tolerance: float) -> list:
    """Douglas–Peucker simplification of one closed ring."""
    points = np.asarray(ring, dtype=float)[:, :2]
    if len(points) <= 4:
        return points.tolist()

    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[end] - points[start]
        offsets = points[start + 1 : end] - points[start]
        length = np.hypot(*segment)
        if length == 0:
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
        else:
            distances = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]) / length
        farthest = int(np.argmax(distances))
        if distances[farthest] > tolerance:
            index = start + 1 + farthest
            keep[index] = True
            stack.extend([(start, index), (index, end)])

    simplified = points[keep]
    if len(simplified) < 4:
        # A tiny ring collapsed: keep four evenly spaced original points.
        simplified = points[np.linspace(0, len(points) - 1, 4).astype(int)]
    return np.round(simplified, 5).tolist()


def _ring_area_km2(ring: list) -> float:
    """Area of one lon/lat ring on the sphere, in km² (unsigned).

    Spherical-excess approximation used by GeoJSON tools (Chamberlain & Duquette,
    "Some algorithms for polygons on a sphere", 2007).
    """
    points = np.radians(np.asarray(ring, dtype=float)[:, :2])
    if len(points) < 3:
        return 0.0
    longitudes, latitudes = points[:, 0], points[:, 1]
    following = np.roll(np.arange(len(points)), -1)
    total = np.sum(
        (longitudes[following] - longitudes) * (2 + np.sin(latitudes) + np.sin(latitudes[following]))
    )
    return abs(total) * EARTH_RADIUS_KM**2 / 2


def _geometry_area_km2(geometry: dict) -> float:
    """Land area of a Polygon/MultiPolygon in km² (outer rings minus holes)."""
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    area = 0.0
    for polygon in polygons:
        if not polygon:
            continue
        area += _ring_area_km2(polygon[0])
        area -= sum(_ring_area_km2(hole) for hole in polygon[1:])
    return area


def _simplify_geometry(geometry: dict, tolerance: float) -> dict:
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}.")
    simplified = [
        [_simplify_ring(ring, tolerance) for ring in polygon] for polygon in polygons
    ]
    return {"type": "MultiPolygon", "coordinates": simplified}


def load_boundaries(path: str | Path, tolerance: float = SIMPLIFY_TOLERANCE) -> dict:
    """Load the DHIS2 org-unit GeoJSON and return simplified boundaries.

    Returns ``{"country": FeatureCollection, "provinces": FeatureCollection,
    "districts": FeatureCollection, "names": DataFrame}``.  Each feature's ``id`` is its ``ouID`` so it can be
    joined to the population data directly.  ``names`` also holds each area's
    size in km² (``area_km2``), measured on the full-detail boundary.
    """
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    collections = {COUNTRY_LEVEL: [], PROVINCE_LEVEL: [], DISTRICT_LEVEL: []}
    names = []
    for feature in source["features"]:
        properties = feature.get("properties", {})
        level = properties.get("level")
        if level not in collections or not feature.get("geometry"):
            continue
        ou_id = feature.get("ouID") or properties.get("id")
        collections[level].append(
            {
                "type": "Feature",
                "id": ou_id,
                "properties": {"name": properties.get("name", "")},
                "geometry": _simplify_geometry(feature["geometry"], tolerance),
            }
        )
        names.append(
            {
                "ouID": ou_id,
                "level": level,
                "name": properties.get("name", ""),
                "displayName": properties.get("displayName", properties.get("name", "")),
                "area_km2": _geometry_area_km2(feature["geometry"]),
            }
        )
    return {
        "country": {"type": "FeatureCollection", "features": collections[COUNTRY_LEVEL]},
        "provinces": {"type": "FeatureCollection", "features": collections[PROVINCE_LEVEL]},
        "districts": {"type": "FeatureCollection", "features": collections[DISTRICT_LEVEL]},
        "names": pd.DataFrame(names),
    }


def district_totals(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Total population per district (ouID) for one year column.

    Values are not rounded here, so province and national sums of forecast
    years stay equal to the overview forecast; round only for display.
    """
    totals = (
        df.groupby(["ouID", "ouName"], as_index=False)[value_column]
        .sum()
        .rename(columns={value_column: "Population"})
    )
    totals["Population"] = totals["Population"].astype(float)
    return totals


def province_totals(district_table: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Add district totals up to provinces.

    District names start with their province code ("0101 Chanthabouli" belongs
    to province "01"), which matches the province names in the GeoJSON.
    """
    provinces = names[names["level"] == PROVINCE_LEVEL].copy()
    provinces["code"] = provinces["name"].astype(str).str[:2]
    totals = district_table.assign(code=district_table["ouName"].astype(str).str[:2])
    totals = totals.groupby("code", as_index=False)["Population"].sum()
    return provinces.merge(totals, on="code", how="left").fillna({"Population": 0})


def _log_ticks(low: float, high: float) -> list[float]:
    """1–2–5 tick values between 10**low and 10**high."""
    ticks = []
    for exponent in range(int(np.floor(low)) - 1, int(np.ceil(high)) + 1):
        for step in (1, 2, 5):
            value = step * 10.0**exponent
            if low <= np.log10(value) <= high:
                ticks.append(value)
    return ticks


# OpenStreetMap standard tiles behind every map. The tiles load from the
# internet; Plotly shows the required "(c) OpenStreetMap contributors" credit.
BASEMAP_STYLE = "open-street-map"
# Plain style used when the map background is switched off (works offline).
PLAIN_STYLE = "white-bg"


def _map_view(geojson_layers: list[dict], height: int, width: int = 520) -> dict:
    """Centre and zoom that fit all features of the given GeoJSON layers."""
    longitudes, latitudes = [], []
    for layer in geojson_layers:
        for feature in layer.get("features", []):
            for polygon in feature["geometry"]["coordinates"]:
                ring = np.asarray(polygon[0], dtype=float)
                longitudes.extend([ring[:, 0].min(), ring[:, 0].max()])
                latitudes.extend([ring[:, 1].min(), ring[:, 1].max()])
    if not longitudes:
        return {"center": {"lon": 103.8, "lat": 18.2}, "zoom": 5}
    west, east = min(longitudes), max(longitudes)
    south, north = min(latitudes), max(latitudes)

    def mercator_y(latitude: float) -> float:
        return float(np.log(np.tan(np.pi / 4 + np.radians(latitude) / 2)))

    lon_span = max(east - west, 1e-3)
    lat_span = max(mercator_y(north) - mercator_y(south), 1e-4)
    zoom_lon = np.log2(width * 360 / (512 * lon_span))  # MapLibre uses 512 px tiles
    zoom_lat = np.log2(height * 2 * np.pi / (512 * lat_span))
    center_y = (mercator_y(north) + mercator_y(south)) / 2
    center_lat = np.degrees(2 * np.arctan(np.exp(center_y)) - np.pi / 2)
    return {
        "center": {"lon": (west + east) / 2, "lat": float(center_lat)},
        "zoom": float(min(zoom_lon, zoom_lat) - 0.15),
    }


def population_map_figure(
    table: pd.DataFrame,
    geojson: dict,
    title: str,
    template: str,
    height: int = 680,
    value_label: str = "Population",
    value_column: str = "Population",
    value_format: str = ",.0f",
    colorscale: list | None = None,
    hover_lines: list[tuple[str, str, str]] | None = None,
    basemap: bool = True,
) -> go.Figure:
    """Choropleth of ``value_column`` in ``table`` (also needs ouID, name, displayName).

    ``value_label`` names the value in the hover text and colour bar, so the
    caller can pass it in the display language.  ``hover_lines`` adds extra
    hover rows as ``(label, column, number format)``.
    """
    values = table[value_column].astype(float)
    # Log colour scale so small areas are not all the same pale shade; the
    # colour bar still shows real values.
    color_values = np.log10(values.clip(lower=1e-3))
    low, high = float(color_values.min()), float(color_values.max())
    tick_values = _log_ticks(low, high)

    extra = hover_lines or []
    customdata = np.column_stack(
        [table["displayName"], table["name"], values]
        + [table[column].astype(float) for _, column, _ in extra]
    )
    extra_hover = "".join(
        f"<br>{label}: %{{customdata[{position}]:{number_format}}}"
        for position, (label, _, number_format) in enumerate(extra, start=3)
    )

    figure = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=table["ouID"],
            z=color_values,
            zmin=low,
            zmax=high,
            colorscale=colorscale or MAP_COLORSCALE,
            marker={"opacity": 0.68 if basemap else 1.0, "line": {"color": "#ffffff", "width": 0.7}},
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                f"{value_label}: %{{customdata[2]:{value_format}}}"
                f"{extra_hover}<extra></extra>"
            ),
            colorbar={
                "title": {"text": value_label, "side": "top"},
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.02,
                "yanchor": "top",
                "tickvals": [np.log10(value) for value in tick_values],
                "ticktext": [
                    f"{value:,.0f}" if value >= 1 else f"{value:,.1f}" for value in tick_values
                ],
                "thickness": 12,
                "len": 0.9,
            },
        )
    )
    figure.update_layout(
        template=template,
        title={"text": title, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 0, "r": 0, "t": 48, "b": 70},
        hoverlabel={"bgcolor": "#ffffff", "font_size": 14},
        map={
            "style": BASEMAP_STYLE if basemap else PLAIN_STYLE,
            **_map_view([geojson], height - 118),
        },
    )
    return figure


def area_overview_figure(
    boundaries: dict,
    area_table: pd.DataFrame,
    highlight_ids: list[str] | None,
    labels: dict[str, str],
    template: str,
    height: int = 560,
    basemap: bool = True,
) -> go.Figure:
    """Map of Laos for the dataset step, on OpenStreetMap.

    With no ``highlight_ids`` the whole country (level 1) is filled; otherwise
    the selected districts are filled on top of the country and province
    outlines.  ``area_table`` holds ouID, name, displayName, area_km2,
    Population and Density for every area that can be hovered.  ``labels``
    gives the hover labels in the display language (area, population,
    density).
    """
    hover = (
        "<b>%{customdata[0]}</b><br>%{customdata[1]}"
        f"<br>{labels['area']}: %{{customdata[2]:,.0f}} km²"
        f"<br>{labels['population']}: %{{customdata[3]:,.0f}}"
        f"<br>{labels['density']}: %{{customdata[4]:,.1f}}<extra></extra>"
    )

    def layer(geojson: dict, table: pd.DataFrame, fill: str, line: str, width: float, hoverable: bool):
        return go.Choroplethmap(
            geojson=geojson,
            locations=table["ouID"],
            z=np.ones(len(table)),
            colorscale=[[0, fill], [1, fill]],
            showscale=False,
            marker={"line": {"color": line, "width": width}},
            customdata=table[["displayName", "name", "area_km2", "Population", "Density"]].to_numpy(),
            hovertemplate=hover if hoverable else None,
            hoverinfo=None if hoverable else "skip",
        )

    country = area_table[area_table["level"] == COUNTRY_LEVEL]
    provinces = area_table[area_table["level"] == PROVINCE_LEVEL]
    districts = area_table[area_table["level"] == DISTRICT_LEVEL]
    figure = go.Figure()
    if highlight_ids:
        figure.add_trace(layer(boundaries["country"], country, "rgba(30,123,79,0.06)", "rgba(11,93,59,0.9)", 1.6, False))
        figure.add_trace(layer(boundaries["provinces"], provinces, "rgba(0,0,0,0)", "rgba(11,93,59,0.45)", 0.8, False))
        selected = districts[districts["ouID"].isin(highlight_ids)]
        figure.add_trace(layer(boundaries["districts"], selected, "rgba(30,123,79,0.55)", "#0B5D3B", 1.4, True))
    else:
        country_fill = "rgba(30,123,79,0.28)" if basemap else "rgba(122,196,150,0.9)"
        figure.add_trace(layer(boundaries["country"], country, country_fill, "rgba(11,93,59,0.95)", 2.0, True))
        province_line = "rgba(11,93,59,0.55)" if basemap else "rgba(255,255,255,0.85)"
        figure.add_trace(layer(boundaries["provinces"], provinces, "rgba(0,0,0,0)", province_line, 0.9, False))
    figure.update_layout(
        template=template,
        height=height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"bgcolor": "rgba(255,255,255,0.95)", "font_size": 14},
        map={
            "style": BASEMAP_STYLE if basemap else PLAIN_STYLE,
            **_map_view([boundaries["country"]], height),
        },
    )
    return figure

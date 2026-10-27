"""Robust demographic outlier checks for the population dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DIMENSION_COLUMNS = ("ouName", "cocName", "sexName")


@dataclass(frozen=True)
class OutlierResults:
    """Flagged observations produced by the three complementary checks."""

    nationwide: pd.DataFrame
    historical: pd.DataFrame
    demographic: pd.DataFrame


@dataclass(frozen=True)
class OutlierCorrectionResult:
    """Conservatively corrected data plus a cell-level audit trail."""

    data: pd.DataFrame
    corrections: pd.DataFrame
    outliers: OutlierResults


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Outlier checks require these columns: " + ", ".join(missing)
        )


def _long_population(df: pd.DataFrame, year_columns: list[str]) -> pd.DataFrame:
    _require_columns(df, [*DIMENSION_COLUMNS, *year_columns])
    identifier_columns = [
        column
        for column in ("ouID", "ouName", "cocID", "cocName", "cocCode", "sexID", "sexName", "sexCode")
        if column in df.columns
    ]
    long = df.melt(
        id_vars=identifier_columns,
        value_vars=year_columns,
        var_name="Year",
        value_name="Population",
    )
    long["Year"] = pd.to_numeric(long["Year"], errors="raise").astype(int)
    long["Population"] = pd.to_numeric(long["Population"], errors="raise")
    return long


def _robust_scores(values: pd.Series, groups: list[pd.Series]) -> tuple[pd.Series, pd.Series]:
    """Return group medians and signed robust scores with stable scale fallbacks."""
    grouped = values.groupby(groups, sort=False, dropna=False)
    median = grouped.transform("median")
    deviation = values - median
    absolute_deviation = deviation.abs()
    mad = absolute_deviation.groupby(groups, sort=False, dropna=False).transform("median")

    q1 = grouped.transform("quantile", q=0.25)
    q3 = grouped.transform("quantile", q=0.75)
    robust_scale = mad / 0.67448975
    robust_scale = robust_scale.mask(robust_scale <= 0, (q3 - q1) / 1.3489795)

    standard_deviation = grouped.transform("std")
    robust_scale = robust_scale.mask(robust_scale <= 0, standard_deviation)
    score = deviation / robust_scale.replace(0, np.nan)
    return median, score.fillna(0.0)


def _age_number(df: pd.DataFrame) -> pd.Series:
    if "cocCode" in df.columns:
        return pd.to_numeric(df["cocCode"], errors="coerce")
    return pd.to_numeric(df["cocName"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def _identity_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ("ouID", "ouName", "sexID", "sexName") if column in df.columns]


def nationwide_district_outliers(
    long: pd.DataFrame, threshold: float = 3.5
) -> pd.DataFrame:
    """Compare each district's age-sex share with nationwide district peers."""
    district_keys = [column for column in ("ouID", "ouName") if column in long.columns]
    totals = long.groupby([*district_keys, "Year"], dropna=False)["Population"].transform("sum")
    work = long.assign(Value=np.where(totals > 0, long["Population"] / totals * 100, np.nan))
    group_columns = ["cocName", "sexName", "Year"]
    median, score = _robust_scores(work["Value"], [work[column] for column in group_columns])
    work["Expected"] = median
    work["Robust score"] = score
    work = work[work["Robust score"].abs() > threshold].copy()
    work["Check"] = "Nationwide district comparison"
    work["Reason"] = "Unusual district age-sex share"
    work["Observed"] = work["Value"]
    work["Unit"] = "% of district population"
    return _result_columns(work)


def historical_change_outliers(
    long: pd.DataFrame, threshold: float = 3.5
) -> pd.DataFrame:
    """Compare cohort-aware annual changes with the same transition nationwide."""
    work = long.copy()
    work["Age"] = _age_number(work)
    work = work[work["Age"].between(1, 79, inclusive="both")].copy()
    current = work.rename(columns={"Population": "Current population"})
    previous = work.copy()
    previous["Age"] = previous["Age"] + 1
    previous["Year"] = previous["Year"] + 1
    previous = previous.rename(columns={"Population": "Previous population"})

    join_columns = [*_identity_columns(work), "Age", "Year"]
    previous_columns = [*join_columns, "Previous population"]
    paired = current.merge(previous[previous_columns], on=join_columns, how="inner")
    paired["Value"] = np.log1p(paired["Current population"]) - np.log1p(
        paired["Previous population"]
    )
    group_columns = ["Age", "sexName", "Year"]
    median, score = _robust_scores(
        paired["Value"], [paired[column] for column in group_columns]
    )
    paired["Expected log change"] = median
    paired["Robust score"] = score
    paired = paired[paired["Robust score"].abs() > threshold].copy()
    paired["Check"] = "Historical change"
    paired["Reason"] = "Unusual cohort change from the previous age and year"
    paired["Observed"] = np.expm1(paired["Value"]) * 100
    paired["Expected"] = np.expm1(paired["Expected log change"]) * 100
    paired["Population"] = paired["Current population"]
    paired["Unit"] = "% cohort change"
    return _result_columns(paired)


def _sex_ratio_outliers(long: pd.DataFrame, threshold: float) -> pd.DataFrame:
    work = long.copy()
    if "sexCode" in work.columns:
        sex_key = work["sexCode"].astype(str).str.strip().str.upper()
    else:
        sex_key = work["sexName"].astype(str).str.strip().str.upper().str[:1]
    work["Sex key"] = sex_key
    pivot_keys = [column for column in ("ouID", "ouName", "cocName", "Year") if column in work.columns]
    pivot = work.pivot_table(
        index=pivot_keys,
        columns="Sex key",
        values="Population",
        aggfunc="sum",
    ).reset_index()
    if "M" not in pivot.columns or "F" not in pivot.columns:
        return _empty_results()
    total = pivot["M"] + pivot["F"]
    pivot["Value"] = np.where(total > 0, pivot["M"] / total * 100, np.nan)
    group_columns = ["cocName", "Year"]
    median, score = _robust_scores(
        pivot["Value"], [pivot[column] for column in group_columns]
    )
    pivot["Expected"] = median
    pivot["Robust score"] = score
    pivot = pivot[pivot["Robust score"].abs() > threshold].copy()
    pivot["sexName"] = "Male/Female"
    pivot["Population"] = total.loc[pivot.index]
    pivot["Check"] = "Demographic consistency"
    pivot["Reason"] = "Unusual male share"
    pivot["Observed"] = pivot["Value"]
    pivot["Unit"] = "% male"
    return _result_columns(pivot)


def _age_profile_outliers(long: pd.DataFrame, threshold: float) -> pd.DataFrame:
    work = long.copy()
    work["Age"] = _age_number(work)
    work = work[work["Age"].between(0, 79, inclusive="both")].copy()
    keys = _identity_columns(work)
    indexed = work.set_index([*keys, "Year", "Age"])["Population"]
    previous = indexed.rename_axis(index={"Age": "Previous age"}).reset_index()
    previous["Age"] = previous["Previous age"] + 1
    previous = previous.rename(columns={"Population": "Previous age population"})
    following = indexed.rename_axis(index={"Age": "Following age"}).reset_index()
    following["Age"] = following["Following age"] - 1
    following = following.rename(columns={"Population": "Following age population"})

    join_columns = [*keys, "Year", "Age"]
    profile = work.merge(
        previous[[*join_columns, "Previous age population"]],
        on=join_columns,
        how="inner",
    ).merge(
        following[[*join_columns, "Following age population"]],
        on=join_columns,
        how="inner",
    )
    profile["Value"] = np.log1p(profile["Population"]) - (
        np.log1p(profile["Previous age population"])
        + np.log1p(profile["Following age population"])
    ) / 2
    group_columns = ["Age", "sexName", "Year"]
    median, score = _robust_scores(
        profile["Value"], [profile[column] for column in group_columns]
    )
    profile["Expected local deviation"] = median
    profile["Robust score"] = score
    profile = profile[profile["Robust score"].abs() > threshold].copy()
    profile["Check"] = "Demographic consistency"
    profile["Reason"] = "Unusual break between adjacent ages"
    profile["Observed"] = (np.exp(profile["Value"]) - 1) * 100
    profile["Expected"] = (np.exp(profile["Expected local deviation"]) - 1) * 100
    profile["Unit"] = "% above/below adjacent-age baseline"
    return _result_columns(profile)


def demographic_consistency_outliers(
    long: pd.DataFrame, threshold: float = 3.5
) -> pd.DataFrame:
    """Combine sex-ratio and adjacent-age profile checks."""
    return pd.concat(
        [
            _sex_ratio_outliers(long, threshold),
            _age_profile_outliers(long, threshold),
        ],
        ignore_index=True,
    ).sort_values(["Year", "District", "Age group", "Reason"], ignore_index=True)


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Check",
            "Reason",
            "District",
            "Age group",
            "Sex",
            "Year",
            "Population",
            "Observed",
            "Expected",
            "Unit",
            "Robust score",
        ]
    )


def _result_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_results()
    result = pd.DataFrame(
        {
            "Check": df["Check"],
            "Reason": df["Reason"],
            "District": df["ouName"],
            "Age group": df["cocName"],
            "Sex": df["sexName"],
            "Year": df["Year"].astype(int),
            "Population": df["Population"],
            "Observed": df["Observed"],
            "Expected": df["Expected"],
            "Unit": df["Unit"],
            "Robust score": df["Robust score"],
        }
    )
    return result.sort_values(
        ["Year", "District", "Age group", "Sex"], ignore_index=True
    )


def detect_population_outliers(
    df: pd.DataFrame, year_columns: list[str], threshold: float = 3.5
) -> OutlierResults:
    """Run all checks against the full nationwide reference dataset."""
    if threshold <= 0:
        raise ValueError("The robust-score threshold must be positive.")
    long = _long_population(df, year_columns)
    return OutlierResults(
        nationwide=nationwide_district_outliers(long, threshold),
        historical=historical_change_outliers(long, threshold),
        demographic=demographic_consistency_outliers(long, threshold),
    )


CORRECTION_COLUMNS = [
    "District",
    "Age group",
    "Sex",
    "Year",
    "Original value",
    "Corrected value",
    "Difference",
    "Change (%)",
    "Temporal robust score",
    "Supporting check",
    "Supporting robust score",
    "Correction method",
]


def _theil_sen_log_estimate(
    years: np.ndarray, values: np.ndarray, target_year: int
) -> float:
    """Estimate one endpoint from the median pairwise log-growth slope."""
    log_values = np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))
    slopes = []
    for left in range(len(years) - 1):
        for right in range(left + 1, len(years)):
            year_gap = float(years[right] - years[left])
            if year_gap:
                slopes.append((log_values[right] - log_values[left]) / year_gap)
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(log_values - slope * years))
    return float(np.expm1(intercept + slope * target_year))


def correct_population_outliers(
    df: pd.DataFrame,
    year_columns: list[str],
    threshold: float = 3.5,
    max_corrections_per_series: int = 2,
) -> OutlierCorrectionResult:
    """Correct only temporal anomalies confirmed by an independent check.

    Internal years use two-sided log-linear interpolation. Endpoints use a
    Theil-Sen trend on the remaining years. Adjacent candidates are not both
    corrected, and each district-age-sex series is capped to avoid replacing
    a genuine demographic trajectory.
    """
    if threshold <= 0:
        raise ValueError("The robust-score threshold must be positive.")
    if max_corrections_per_series < 1:
        raise ValueError("max_corrections_per_series must be at least 1.")
    _require_columns(df, [*DIMENSION_COLUMNS, *year_columns])

    years = np.asarray([int(column) for column in year_columns], dtype=int)
    if len(np.unique(years)) != len(years):
        raise ValueError("Historical year columns must be unique.")
    year_order = np.argsort(years)
    years = years[year_order]
    ordered_columns = [year_columns[position] for position in year_order]

    results = detect_population_outliers(df, ordered_columns, threshold)
    exact_keys = ["District", "Age group", "Sex", "Year"]
    broad_keys = ["District", "Age group", "Year"]

    historical = results.historical.copy()
    if historical.empty:
        return OutlierCorrectionResult(
            df.copy(), pd.DataFrame(columns=CORRECTION_COLUMNS), results
        )
    historical = historical.rename(columns={"Robust score": "Temporal robust score"})
    historical = historical[
        [*exact_keys, "Temporal robust score"]
    ].drop_duplicates(exact_keys)

    support_frames = []
    if not results.nationwide.empty:
        nationwide = results.nationwide.copy()
        nationwide["Supporting check"] = "Nationwide district comparison"
        nationwide = nationwide.rename(
            columns={"Robust score": "Supporting robust score"}
        )
        support_frames.append(
            nationwide[[*exact_keys, "Supporting check", "Supporting robust score"]]
        )

    demographic = results.demographic.copy()
    exact_demographic = demographic[demographic["Sex"].ne("Male/Female")].copy()
    if not exact_demographic.empty:
        exact_demographic["Supporting check"] = exact_demographic["Reason"]
        exact_demographic = exact_demographic.rename(
            columns={"Robust score": "Supporting robust score"}
        )
        support_frames.append(
            exact_demographic[
                [*exact_keys, "Supporting check", "Supporting robust score"]
            ]
        )

    supported = []
    if support_frames:
        exact_support = pd.concat(support_frames, ignore_index=True)
        supported.append(historical.merge(exact_support, on=exact_keys, how="inner"))

    sex_ratio = demographic[demographic["Sex"].eq("Male/Female")].copy()
    if not sex_ratio.empty:
        sex_ratio["Supporting check"] = sex_ratio["Reason"]
        sex_ratio = sex_ratio.rename(
            columns={"Robust score": "Supporting robust score"}
        )
        supported.append(
            historical.merge(
                sex_ratio[
                    [*broad_keys, "Supporting check", "Supporting robust score"]
                ],
                on=broad_keys,
                how="inner",
            )
        )

    if not supported:
        return OutlierCorrectionResult(
            df.copy(), pd.DataFrame(columns=CORRECTION_COLUMNS), results
        )

    candidates = pd.concat(supported, ignore_index=True)
    candidates["Combined score"] = (
        candidates["Temporal robust score"].abs()
        + candidates["Supporting robust score"].abs()
    )
    candidates = candidates.sort_values("Combined score", ascending=False).drop_duplicates(
        exact_keys
    )

    candidate_groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in candidates.to_dict("records"):
        identity = (
            str(row["District"]),
            str(row["Age group"]),
            str(row["Sex"]),
        )
        candidate_groups.setdefault(identity, []).append(row)

    source_indices: dict[tuple[str, str, str], list[object]] = {}
    for index, district, age_group, sex in zip(
        df.index, df["ouName"], df["cocName"], df["sexName"]
    ):
        identity = (str(district), str(age_group), str(sex))
        source_indices.setdefault(identity, []).append(index)

    corrected = df.copy()
    audit_rows: list[dict[str, object]] = []

    for identity, row_candidates in candidate_groups.items():
        matching_indices = source_indices.get(identity, [])
        if not matching_indices:
            continue
        row_candidates.sort(key=lambda row: float(row["Combined score"]), reverse=True)

        selected = []
        for candidate in row_candidates:
            candidate_year = int(candidate["Year"])
            if any(abs(candidate_year - int(item["Year"])) <= 1 for item in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_corrections_per_series:
                break

        for index in matching_indices:
            original_values = df.loc[index, ordered_columns].to_numpy(dtype=float)
            for candidate in selected:
                year = int(candidate["Year"])
                position = int(np.where(years == year)[0][0])
                original = float(original_values[position])
                if 0 < position < len(years) - 1:
                    left = max(float(original_values[position - 1]), 0.0)
                    right = max(float(original_values[position + 1]), 0.0)
                    estimate = float(
                        np.expm1((np.log1p(left) + np.log1p(right)) / 2)
                    )
                    method = "Two-sided log-linear interpolation"
                else:
                    retained = np.ones(len(years), dtype=bool)
                    retained[position] = False
                    estimate = _theil_sen_log_estimate(
                        years[retained], original_values[retained], year
                    )
                    method = "Theil-Sen endpoint estimate"

                corrected_value = int(np.rint(max(estimate, 0.0)))
                corrected.at[index, str(year)] = corrected_value
                change = (
                    (corrected_value - original) / original * 100
                    if original != 0
                    else np.nan
                )
                audit_rows.append(
                    {
                        "District": identity[0],
                        "Age group": identity[1],
                        "Sex": identity[2],
                        "Year": year,
                        "Original value": original,
                        "Corrected value": corrected_value,
                        "Difference": corrected_value - original,
                        "Change (%)": change,
                        "Temporal robust score": float(
                            candidate["Temporal robust score"]
                        ),
                        "Supporting check": str(candidate["Supporting check"]),
                        "Supporting robust score": float(
                            candidate["Supporting robust score"]
                        ),
                        "Correction method": method,
                    }
                )

    corrections = pd.DataFrame(audit_rows, columns=CORRECTION_COLUMNS)
    if not corrections.empty:
        corrections = corrections.sort_values(
            ["Year", "District", "Age group", "Sex"], ignore_index=True
        )
    return OutlierCorrectionResult(corrected, corrections, results)

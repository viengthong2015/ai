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

import unittest

import pandas as pd

from outlier_detection import correct_population_outliers, detect_population_outliers


class OutlierDetectionTests(unittest.TestCase):
    def setUp(self):
        records = []
        for district_number in range(12):
            for age in range(6):
                for sex_code, sex_name, sex_offset in (
                    ("M", "Male", 8),
                    ("F", "Female", 0),
                ):
                    record = {
                        "ouID": f"D{district_number}",
                        "ouName": f"District {district_number}",
                        "cocCode": str(age),
                        "cocName": f"{age} years",
                        "sexCode": sex_code,
                        "sexName": sex_name,
                    }
                    for year in range(2020, 2023):
                        record[str(year)] = (
                            900
                            + district_number * 7
                            + age * 12
                            + sex_offset
                            + (year - 2020) * 4
                            + (district_number % 3)
                        )
                    records.append(record)
        self.data = pd.DataFrame(records)
        anomaly = (
            self.data["ouName"].eq("District 11")
            & self.data["cocName"].eq("3 years")
            & self.data["sexName"].eq("Male")
        )
        self.data.loc[anomaly, "2022"] = 8_000

    def test_all_three_checks_find_injected_anomaly(self):
        results = detect_population_outliers(
            self.data, ["2020", "2021", "2022"], threshold=3.5
        )
        for result in (
            results.nationwide,
            results.historical,
            results.demographic,
        ):
            self.assertIn("District 11", set(result["District"]))
            self.assertTrue((result["Robust score"].abs() > 3.5).all())

    def test_results_use_review_friendly_schema(self):
        results = detect_population_outliers(
            self.data, ["2020", "2021", "2022"], threshold=3.5
        )
        expected_columns = {
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
        }
        self.assertEqual(expected_columns, set(results.nationwide.columns))
        self.assertEqual(expected_columns, set(results.historical.columns))
        self.assertEqual(expected_columns, set(results.demographic.columns))

    def test_threshold_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "threshold must be positive"):
            detect_population_outliers(self.data, ["2020"], threshold=0)

    def test_conservative_correction_repairs_confirmed_anomaly(self):
        original = self.data.copy(deep=True)

        result = correct_population_outliers(
            self.data, ["2020", "2021", "2022"], threshold=3.5
        )

        corrected_row = result.data[
            result.data["ouName"].eq("District 11")
            & result.data["cocName"].eq("3 years")
            & result.data["sexName"].eq("Male")
        ].iloc[0]
        self.assertLess(corrected_row["2022"], 2_000)
        correction = result.corrections[
            result.corrections["District"].eq("District 11")
            & result.corrections["Age group"].eq("3 years")
            & result.corrections["Sex"].eq("Male")
            & result.corrections["Year"].eq(2022)
        ].iloc[0]
        self.assertEqual(correction["Original value"], 8_000)
        self.assertEqual(correction["Correction method"], "Theil-Sen endpoint estimate")
        pd.testing.assert_frame_equal(self.data, original)

    def test_single_unconfirmed_flag_is_not_automatically_corrected(self):
        data = self.data.copy()
        anomaly = (
            data["ouName"].eq("District 10")
            & data["cocName"].eq("5 years")
            & data["sexName"].eq("Female")
        )
        data.loc[anomaly, "2022"] = data.loc[anomaly, "2022"] * 2

        result = correct_population_outliers(
            data, ["2020", "2021", "2022"], threshold=3.5
        )

        corrected = result.data.loc[anomaly, "2022"].iloc[0]
        self.assertGreater(corrected, 1_000)


if __name__ == "__main__":
    unittest.main()

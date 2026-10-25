import unittest

import numpy as np
import pandas as pd

from population_forecast import (
    HISTORICAL_YEARS,
    _polynomial_design,
    annual_growth_rates,
    dataframe_to_html_bytes,
    forecast_totals,
)


class AnnualGrowthRatesTests(unittest.TestCase):
    def test_calculates_year_over_year_percentages(self):
        rates = annual_growth_rates(np.array([100.0, 110.0, 121.0, 108.9]))

        np.testing.assert_allclose(rates, np.array([10.0, 10.0, -10.0]))

    def test_returns_nan_when_previous_population_is_zero(self):
        rates = annual_growth_rates(np.array([0.0, 10.0, 15.0]))

        self.assertTrue(np.isnan(rates[0]))
        self.assertEqual(rates[1], 50.0)

    def test_rejects_multidimensional_input(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            annual_growth_rates(np.array([[100.0, 110.0]]))


class PolynomialForecastStabilityTests(unittest.TestCase):
    def test_polynomial_design_is_centered_and_well_conditioned(self):
        design = _polynomial_design(HISTORICAL_YEARS)
        design_with_intercept = np.column_stack((np.ones(len(design)), design))

        self.assertEqual(design.dtype, np.float64)
        self.assertLess(np.linalg.cond(design_with_intercept), 20)

    def test_quadratic_history_is_fitted_without_rank_loss(self):
        offsets = np.arange(len(HISTORICAL_YEARS), dtype=float)
        population = 1_000_000 + 10_000 * offsets + 250 * offsets**2
        frame = pd.DataFrame(
            [{str(year): value for year, value in zip(HISTORICAL_YEARS, population)}]
        )

        result = forecast_totals(frame)

        self.assertLess(result.metrics.loc["Polynomial", "MAPE (%)"], 1e-10)
        self.assertTrue(np.isfinite(result.future["Polynomial"]).all())


class HtmlExportTests(unittest.TestCase):
    def test_generates_styled_standalone_report(self):
        table = pd.DataFrame(
            {
                "District": ["Example", "Grand Total"],
                "2026_Poly": [1234, 1234],
                "Total": [5678, 5678],
            }
        )

        html = dataframe_to_html_bytes(table).decode("utf-8")

        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn('<meta name="viewport"', html)
        self.assertIn('class="dataframe population-table"', html)
        self.assertIn("Polynomial Population Forecast", html)
        self.assertIn("position: sticky", html)
        self.assertIn("1,234", html)
        self.assertIn("Grand Total", html)
        self.assertIn('id="download-csv"', html)
        self.assertIn("Download table as CSV", html)
        self.assertIn('link.download = "Polynomial_Population_Forecast.csv"', html)
        self.assertIn('id="page-size"', html)
        self.assertIn('id="previous-page"', html)
        self.assertIn('id="next-page"', html)
        self.assertIn("function renderPage()", html)
        self.assertIn("row.hidden =", html)


if __name__ == "__main__":
    unittest.main()

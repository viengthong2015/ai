import unittest

import numpy as np
import pandas as pd

from population_forecast import annual_growth_rates, dataframe_to_html_bytes


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

import unittest

import numpy as np

from population_forecast import annual_growth_rates


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


if __name__ == "__main__":
    unittest.main()

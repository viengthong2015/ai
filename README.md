# Laos Population Forecast (Streamlit)

This application turns the workflow in `estimatePop.ipynb` into an interactive
Streamlit dashboard. It compares Linear Regression, degree-2 Polynomial
Regression, and Exponential Growth models using the 2016–2025 population data.
The sidebar dropdown can display one model or use **All models** to compare
their results and accuracy. The overview includes a dedicated historical column
chart comparing actual values with the selected models' fitted predictions,
including signed percentage-difference labels for every predicted value.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads `data/dataDistrict.csv` by default. You can also upload a CSV with
the required year columns `2016` through `2025`, filter the records, change the
forecast end year, inspect model metrics and residuals, and download row-level
CSV or HTML forecasts.

The source-data tab also runs three nationwide, robust outlier checks:
district age-sex shares, cohort-aware historical changes, and demographic
consistency (sex ratio and adjacent-age profile). Sidebar filters affect only
the displayed flags, while detection thresholds always use the full dataset.

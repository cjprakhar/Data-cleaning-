# CleanSight — Data Cleaning & Quality Dashboard

A local, portfolio-focused data cleaning application built with Python, Pandas, Streamlit and Plotly.

## Features

- Upload Excel (`.xlsx`, `.xls`) and CSV files.
- Choose an Excel worksheet when a workbook has multiple sheets.
- Preview records, column types, unique values and missing data.
- Identify exact duplicate rows, blank text cells, possible numbers stored as text, date-like text and numeric IQR outliers.
- Profile numeric and categorical columns with interactive charts.
- Clean with user-controlled options: trim whitespace, turn blank strings into missing values, standardize text case, convert text numbers, parse dates, normalize column names, remove exact duplicate rows, and fill or drop missing values.
- Review before/after metrics and a cleaning audit log.
- Download the cleaned dataset as CSV or Excel, plus the column profile and a JSON cleaning report.
- Data stays in the local Streamlit session; this starter project does not send uploaded data to an external data service.

## Windows setup

Open Command Prompt in this folder and run:

```bat
.venv\Scripts\python.exe -m streamlit run app.py
```

The setup script creates `.venv` and installs the packages automatically unless `--skip-install` is supplied.

## Important notes

- Make a copy of source data before using any cleaning tool. Cleaning steps may remove rows or change values.
- Date parsing and numeric interpretation are suggestions; inspect the preview before exporting.
- Numeric text cleanup assumes commas are thousands separators and removes common currency symbols and `%` signs without rescaling percentages. Verify locale-specific formats such as `1.234,56` manually.
- IQR outliers are flagged for review and are never automatically removed.
- This is a local portfolio MVP. It does not yet provide authentication, a database, multi-user collaboration, or deployment security controls.

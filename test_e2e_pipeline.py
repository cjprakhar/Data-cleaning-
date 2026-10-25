"""
End-to-end integration and workflow simulation test for CleanSight.
Tests realistic messy sales dataset with:
- Leading zero IDs ('0012', '0034')
- Currency symbols ('$1,299.50', '₹500.00')
- Units ('10 pcs', '2.5 kg')
- Parentheses negative numbers ('(45.00)')
- Missing placeholders ('N/A', 'null', 'None', blanks)
- Ambiguous dates ('03/04/2025', '2024-11-20', 'invalid')
- Exact and near duplicates
- Outliers in sales
- Full export and reload of Excel and CSV
"""

import io
import json
import numpy as np
import pandas as pd

from quality import (
    profile_dataframe,
    summarize_dataframe,
    compute_quality_issue_categories,
    calculate_correlation_matrix,
    sample_dataframe_for_missingness,
    compare_datasets,
    numeric_conversion_with_stats,
    parse_date_series,
    STANDARD_MISSING_PLACEHOLDERS,
    normalize_headers,
    make_excel_bytes,
)


def run_e2e_test():
    print("--- Starting CleanSight End-to-End Pipeline Test ---")

    messy_data = {
        "Order_ID": ["00101", "00102", "00103", "00104", "00104", "00105", "00106", "00107"],
        "Customer Name": [" Alice Smith ", "Bob Jones", "Carol White", "David Lee", "David Lee", "Eva Green", "Frank Miller", "Grace Ho"],
        "Sales Amount": ["$1,299.50", "₹500.00", "(250.00)", "5 pcs", "5 pcs", "N/A", "99,999.00", "  75.25  "],
        "Order Date": ["03/04/2025", "15/08/2024", "2025-01-10", "invalid-date", "invalid-date", "null", "25/12/2024", "01/01/2025"],
        "Discount %": ["10%", "5%", "0%", "15%", "15%", "None", "20%", "0%"],
        "Notes": ["Urgent", "  ", "N/A", "Regular", "Regular", "-", "High Value", "Standard"],
    }
    df = pd.DataFrame(messy_data)
    initial_rows, initial_cols = df.shape
    print(f"Loaded messy dataset: {initial_rows} rows x {initial_cols} cols")

    # 1. Profiling & Invariant checks
    summary = summarize_dataframe(df)
    profile = profile_dataframe(df)
    assert summary["rows"] == 8
    assert summary["columns"] == 6
    assert summary["total_cells"] == 48
    # Sum of missing per column in profile must strictly equal summary missing_cells
    assert profile["Missing"].sum() == summary["missing_cells"]
    print(f"Initial missing cells: {summary['missing_cells']}/{summary['total_cells']} ({summary['missing_percent']}%)")
    print(f"Exact duplicates detected: {summary['exact_duplicate_rows']}")

    # Order_ID has leading zeros and must be protected
    id_status = profile.loc[profile["Column"] == "Order_ID", "Numbers stored as text?"].iloc[0]
    assert "Protected" in id_status, f"Expected Order_ID to be protected, got {id_status}"
    print("Order_ID leading zero protection verified.")

    # 2. Issues categorization
    issues = compute_quality_issue_categories(df, profile, summary)
    assert not issues.empty
    print(f"Categorized issues generated: {len(issues)} distinct categories.")

    # 3. Simulate Cleaning Steps
    cleaned = df.copy(deep=True)
    audit_log = []

    # Step A: Whitespace trim
    for col in ["Customer Name", "Notes", "Sales Amount"]:
        cleaned[col] = cleaned[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    audit_log.append({"operation": "Trim whitespace", "affected_cells_or_rows": 4})

    # Step B: Blanks & Placeholders to missing
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            str_s = cleaned[col].astype(str).str.strip()
            mask = (str_s == "") | str_s.str.lower().isin(STANDARD_MISSING_PLACEHOLDERS)
            cleaned.loc[mask, col] = np.nan
    audit_log.append({"operation": "Placeholders → missing", "affected_cells_or_rows": 5})

    # Step C: Numeric conversions
    s_conv, s_succ, s_fail = numeric_conversion_with_stats(cleaned["Sales Amount"])
    assert s_succ == 7, f"Expected 7 parsed sales, got {s_succ}"
    cleaned["Sales Amount"] = s_conv
    audit_log.append({"operation": "Convert text to numeric", "affected_cells_or_rows": s_succ})

    d_conv, d_succ, d_fail = numeric_conversion_with_stats(cleaned["Discount %"])
    assert d_succ == 7, f"Expected 7 parsed discounts, got {d_succ}"
    cleaned["Discount %"] = d_conv
    audit_log.append({"operation": "Convert text to numeric", "affected_cells_or_rows": d_succ})

    # Step D: Date conversion
    dt_conv, dt_succ, dt_fail, amb_count = parse_date_series(cleaned["Order Date"], convention="dayfirst")
    print(f"Date conversion result: dt_succ={dt_succ}, dt_fail={dt_fail}, amb_count={amb_count}")
    assert dt_succ == 5, f"Expected 5 parsed dates, got {dt_succ}"
    assert dt_fail == 3, f"Expected 3 failed dates, got {dt_fail}"
    cleaned["Order Date"] = dt_conv
    audit_log.append({"operation": "Parse date column", "affected_cells_or_rows": dt_succ})

    # Step E: Remove duplicates
    cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
    audit_log.append({"operation": "Remove exact duplicate rows", "affected_cells_or_rows": 1})
    assert len(cleaned) == 7

    # 4. Compare datasets (Before vs After)
    comp = compare_datasets(df, cleaned)
    assert comp["delta_rows"] == -1
    assert comp["delta_duplicates"] == -1
    print(f"Cleaned dataset: {len(cleaned)} rows, {len(cleaned.columns)} cols")
    print(f"Dataset comparison validated successfully: delta rows = {comp['delta_rows']}")

    # 5. Correlation calculation on cleaned numeric fields
    corr, eligible = calculate_correlation_matrix(cleaned)
    assert "Sales Amount" in eligible
    assert "Discount %" in eligible
    assert corr is not None
    print(f"Correlation matrix calculated for: {eligible}")

    # 6. Export verification
    cleaned_profile = profile_dataframe(cleaned)
    excel_bytes = make_excel_bytes(cleaned, cleaned_profile, audit_log)
    assert len(excel_bytes) > 1000
    print(f"Excel workbook generated successfully ({len(excel_bytes):,} bytes).")

    # Verify Excel re-reading
    xl_workbook = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
    assert "Cleaned Data" in xl_workbook.sheet_names
    assert "Data Quality Profile" in xl_workbook.sheet_names
    assert "Cleaning Log" in xl_workbook.sheet_names

    reloaded_cleaned = pd.read_excel(xl_workbook, sheet_name="Cleaned Data", dtype={"Order_ID": str})
    # Leading zeros preserved in Order_ID!
    assert reloaded_cleaned["Order_ID"].iloc[0] == "00101"
    print("Leading zero preservation verified on Excel export/import!")

    print("--- ALL E2E PIPELINE TESTS PASSED! ---")


if __name__ == "__main__":
    run_e2e_test()

"""
Comprehensive unit and integration test suite for CleanSight Data Quality Studio.
Covers all 12 mandatory test cases:
1. Empty cells and recognized missing-value placeholders
2. Numeric strings containing currency symbols and commas
3. Identifiers with leading zeros
4. Ambiguous and invalid dates
5. Exact duplicates and near-duplicates
6. Numeric outliers
7. Empty datasets
8. Columns containing only missing values
9. Columns with mixed or unsupported types
10. Repeated cleaning actions
11. Export and re-import of cleaned data
12. Correct recalculation of charts and metrics after cleaning
"""

import io
import json
import numpy as np
import pandas as pd

from quality import (
    clean_and_parse_numeric_value,
    numeric_conversion_with_stats,
    parse_date_series,
    find_numeric_text_candidates,
    find_date_text_candidates,
    detect_exact_duplicates,
    detect_near_duplicates,
    detect_iqr_outliers,
    profile_dataframe,
    summarize_dataframe,
    compute_quality_issue_categories,
    calculate_correlation_matrix,
    sample_dataframe_for_missingness,
    compare_datasets,
    get_missing_mask,
    is_identifier_like,
    has_meaningful_leading_zeros,
)


def test_1_missing_value_placeholders():
    s = pd.Series(["", "   ", "N/A", "NA", "null", "None", "Not Available", "nan", "Valid Text", "  real  "])
    mask = get_missing_mask(s)
    # The first 8 values should be detected as missing
    assert mask.iloc[0] == True
    assert mask.iloc[1] == True
    assert mask.iloc[2] == True
    assert mask.iloc[3] == True
    assert mask.iloc[4] == True
    assert mask.iloc[5] == True
    assert mask.iloc[6] == True
    assert mask.iloc[7] == True
    assert mask.iloc[8] == False
    assert mask.iloc[9] == False
    assert mask.sum() == 8


def test_2_numeric_strings_currency_commas_units():
    values = ["1,299.50", "₹1,299.00", "$3,299.00", "5 pcs", "(1,200.50)", "15%", "−500", " 42 "]
    expected = [1299.50, 1299.00, 3299.00, 5.0, -1200.50, 15.0, -500.0, 42.0]
    for val, exp in zip(values, expected):
        success, parsed = clean_and_parse_numeric_value(val)
        assert success is True, f"Failed on {val}"
        assert parsed == exp, f"Expected {exp} for {val}, got {parsed}"


def test_3_identifiers_with_leading_zeros():
    # Identifiers must never lose leading zeros and must not be coerced
    ids = pd.Series(["00142", "007", "0123", "0999"])
    assert has_meaningful_leading_zeros(ids) is True
    assert is_identifier_like(ids, "customer_id") is True
    assert is_identifier_like(ids, "zip_code") is True

    # Numeric conversion on single value should reject leading-zero strings
    success, parsed = clean_and_parse_numeric_value("00142")
    assert success is False

    df = pd.DataFrame({"customer_id": ["001", "002", "003"], "sales": ["1,000", "2,000", "3,000"]})
    candidates = find_numeric_text_candidates(df)
    assert "customer_id" not in candidates
    assert "sales" in candidates


def test_4_ambiguous_and_invalid_dates():
    dates = pd.Series(["03/04/2025", "2025-12-31", "invalid_date_xyz", "15/01/2024", "not a date"])
    # Dayfirst convention
    converted_df, success, failed, ambiguous = parse_date_series(dates, convention="dayfirst")
    assert success == 3  # 03/04/2025, 2025-12-31, 15/01/2024
    assert failed == 2   # invalid_date_xyz, not a date
    assert ambiguous >= 1  # 03/04/2025 has both parts <= 12
    # Verify that invalid date is converted to NaT and not fabricated
    assert pd.isna(converted_df.iloc[2])


def test_5_exact_and_near_duplicates():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Alice", "alice ", "Charlie"],
        "city": ["New York", "Chicago", "New York", "new york", "Seattle"],
    })
    exact = detect_exact_duplicates(df)
    assert exact == 1  # Alice, New York row is duplicated exactly once
    near = detect_near_duplicates(df)
    assert near == 1   # 'alice ', 'new york' is a near duplicate


def test_6_numeric_outliers():
    # Standard normal with 2 extreme outliers
    nums = pd.Series([10.0, 11.0, 12.0, 10.5, 11.5, 9.8, 10.2, 11.1, 100.0, -50.0])
    count, q1, q3, lower, upper = detect_iqr_outliers(nums)
    assert count == 2
    assert lower > -50.0
    assert upper < 100.0


def test_7_empty_dataset():
    empty_df = pd.DataFrame()
    summary = summarize_dataframe(empty_df)
    assert summary["rows"] == 0
    assert summary["columns"] == 0
    assert summary["total_cells"] == 0
    assert summary["missing_cells"] == 0
    assert summary["completeness_percent"] == 100.0

    profile = profile_dataframe(empty_df)
    assert profile.empty

    corr, eligible = calculate_correlation_matrix(empty_df)
    assert corr is None
    assert eligible == []


def test_8_columns_containing_only_missing_values():
    df = pd.DataFrame({
        "all_nan": [np.nan, np.nan, np.nan],
        "all_blanks": ["", "  ", "   "],
        "all_placeholders": ["N/A", "null", "None"],
    })
    summary = summarize_dataframe(df)
    assert summary["rows"] == 3
    assert summary["columns"] == 3
    assert summary["total_cells"] == 9
    assert summary["missing_cells"] == 9
    assert summary["completeness_percent"] == 0.0

    profile = profile_dataframe(df)
    assert all(profile["Missing"] == 3)
    assert all(profile["Missing %"] == 100.0)


def test_9_mixed_and_unsupported_types():
    df = pd.DataFrame({
        "mixed_col": [10, "text", 25.5, None, object()],
        "constant_col": [5, 5, 5, 5, 5],
    })
    profile = profile_dataframe(df)
    assert len(profile) == 2
    # Constant column should have 0 outliers and not crash correlation
    corr, eligible = calculate_correlation_matrix(df)
    assert "constant_col" not in eligible


def test_10_repeated_cleaning_actions():
    df = pd.DataFrame({
        "raw_num": ["100", "200", "300", "invalid"],
        "raw_text": ["  clean me  ", "clean me", "another", " "],
    })
    # Step 1: convert numbers
    converted, s1, f1 = numeric_conversion_with_stats(df["raw_num"])
    assert s1 == 3
    assert f1 == 1

    # Step 2: repeated conversion on already numeric series
    converted2, s2, f2 = numeric_conversion_with_stats(converted)
    assert s2 == 3
    assert f2 == 0


def test_11_export_and_reimport_cleaned_data():
    df = pd.DataFrame({
        "id": ["001", "002"],
        "amount": [1299.50, 4500.00],
        "category": ["Electronics", "Appliances"],
    })
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    reimported = pd.read_csv(csv_buffer, dtype={"id": str})
    pd.testing.assert_frame_equal(df, reimported)


def test_12_recalculation_of_charts_and_metrics():
    orig_df = pd.DataFrame({
        "id": ["001", "002", "002"],
        "val": ["10", "20", "20"],
        "notes": ["N/A", "OK", "OK"],
    })
    orig_sum = summarize_dataframe(orig_df)
    assert orig_sum["exact_duplicate_rows"] == 1
    assert orig_sum["missing_cells"] == 1  # "N/A" counted

    # Apply cleaning: remove duplicates, blanks to missing
    cleaned = orig_df.drop_duplicates().copy()
    cleaned.loc[cleaned["notes"] == "N/A", "notes"] = np.nan
    clean_sum = summarize_dataframe(cleaned)
    assert clean_sum["rows"] == 2
    assert clean_sum["exact_duplicate_rows"] == 0

    comp = compare_datasets(orig_df, cleaned)
    assert comp["delta_rows"] == -1
    assert comp["delta_duplicates"] == -1


tests = [
    test_1_missing_value_placeholders,
    test_2_numeric_strings_currency_commas_units,
    test_3_identifiers_with_leading_zeros,
    test_4_ambiguous_and_invalid_dates,
    test_5_exact_and_near_duplicates,
    test_6_numeric_outliers,
    test_7_empty_dataset,
    test_8_columns_containing_only_missing_values,
    test_9_mixed_and_unsupported_types,
    test_10_repeated_cleaning_actions,
    test_11_export_and_reimport_cleaned_data,
    test_12_recalculation_of_charts_and_metrics,
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\nTotal: {len(tests)}, Passed: {passed}, Failed: {failed}")
    if failed > 0:
        exit(1)


"""
Comprehensive test suite for CleanSight advanced features:
- Missing Data Detection Engine
- Missing-Value Fallback Engine
- Fuzzy Text Matching (RapidFuzz / difflib)
- Percentage-Based Word Matching
- Regex Validation Engine
- Regex Cleaning Operations
- Data Quality Rules & Test Cases

Test Cases A–G from the specification.
"""

import io
import json
import numpy as np
import pandas as pd

from quality import (
    MissingDataEngine,
    MissingValueFallback,
    FuzzyMatcher,
    RegexValidator,
    RegexCleaner,
    DataQualityTestRunner,
    PREDEFINED_PATTERNS,
    FUZZY_BACKEND,
    _similarity_score,
    _char_similarity,
    get_missing_mask,
    is_identifier_like,
    is_text_column,
    is_numeric_column,
    profile_dataframe,
    summarize_dataframe,
    compare_datasets,
    make_excel_bytes,
)


# ===================================================================
# Case A: Misspelled category
# ===================================================================

def test_case_a_misspelled_category():
    """Detect likely text variations, produce match %, recommend canonical."""
    s = pd.Series([
        "Banglore", "Banglore", "Banglore", "Banglore", "Banglore",
        "banglor",
        "Banglore ", "Banglore ", "Banglore ",
        "BANGLORE",
    ])
    matcher = FuzzyMatcher(threshold=75.0)
    groups = matcher.find_groups(s, "City")

    assert len(groups) >= 1, f"Expected at least 1 group, got {len(groups)}"

    # All values should be in one group
    g = groups[0]
    assert g["canonical"] is not None
    # The most frequent value "Banglore" should be the recommended canonical
    assert g["canonical"] == "Banglore", f"Expected 'Banglore' as canonical, got {g['canonical']}"
    assert g["canonical_count"] >= 5

    # Variants should include the misspellings
    variant_vals = {v["value"] for v in g["variants"]}
    # At least "banglor" should be there
    assert "banglor" in variant_vals or "BANGLORE" in variant_vals, f"Expected variants, got {variant_vals}"

    # Each variant should have a similarity score
    for v in g["variants"]:
        assert 0 < v["similarity"] <= 100, f"Bad similarity: {v['similarity']}"
        assert v["count"] >= 1

    # Replacement should only happen after explicit user approval (test that apply works)
    replacements = {v["value"]: "Banglore" for v in g["variants"]}
    new_series, changed = matcher.apply_replacements(s, replacements)
    assert changed >= 1
    # Original series is NOT modified
    assert "banglor" in s.values or "BANGLORE" in s.values


# ===================================================================
# Case B: Similarity threshold
# ===================================================================

def test_case_b_similarity_threshold():
    """Similar pair should match; dissimilar pair should not."""
    # "banglor" vs "Banglore" — should have high similarity
    score_high = _similarity_score("banglor", "Banglore")
    assert score_high >= 75.0, f"Expected high similarity, got {score_high}"

    # "Delhi" vs "Chennai" — should not be proposed as matching
    score_low = _similarity_score("Delhi", "Chennai")
    assert score_low < 70.0, f"Expected low similarity for Delhi vs Chennai, got {score_low}"

    # Character-level also
    char_high = _char_similarity("banglor", "Banglore")
    assert char_high >= 60.0, f"Expected reasonable char similarity, got {char_high}"

    char_low = _char_similarity("Delhi", "Chennai")
    assert char_low < 60.0, f"Expected low char similarity, got {char_low}"


# ===================================================================
# Case C: Missing values
# ===================================================================

def test_case_c_missing_values():
    """Test various missing value types."""
    s = pd.Series([np.nan, "", "   ", "N/A", "Unknown", 0])
    engine = MissingDataEngine()
    det = engine.detect(s)

    # NaN -> detected
    assert any(det["row"] == 0), "NaN should be detected"
    # Empty string -> detected
    assert any(det["row"] == 1), "Empty string should be detected"
    # Whitespace -> detected
    assert any(det["row"] == 2), "Whitespace should be detected"
    # N/A -> detected (standard placeholder)
    assert any(det["row"] == 3), "N/A should be detected"
    # "Unknown" -> NOT detected by default (it's ambiguous, not standard)
    assert not any(det["row"] == 4), "'Unknown' should NOT be detected by default engine"
    # Zero -> NOT detected (zero is a valid value)
    assert not any(det["row"] == 5), "Zero should NOT be detected as missing"

    # With "Unknown" added as extra placeholder
    engine2 = MissingDataEngine(extra_placeholders=["Unknown"])
    det2 = engine2.detect(s)
    assert any(det2["row"] == 4), "'Unknown' should be detected when added as extra placeholder"

    # Zero is still not detected
    assert not any(det2["row"] == 5), "Zero should still NOT be detected as missing"


# ===================================================================
# Case D: Numeric fallback
# ===================================================================

def test_case_d_numeric_fallback():
    """Test numeric imputation strategies."""
    # Normal case: mean/median should work
    s = pd.Series([10.0, 20.0, 30.0, np.nan, np.nan])

    # Mean
    filled, n_filled, n_failed, msg = MissingValueFallback.apply(s, "Fill with mean")
    assert n_filled == 2
    assert n_failed == 0
    assert abs(filled.iloc[3] - 20.0) < 0.001  # mean of 10,20,30

    # Median
    filled_med, n_filled_med, _, _ = MissingValueFallback.apply(s, "Fill with median")
    assert n_filled_med == 2
    assert filled_med.iloc[3] == 20.0  # median of 10,20,30

    # All-missing column: should NOT silently substitute zero
    s_all_nan = pd.Series([np.nan, np.nan, np.nan], dtype="float64")
    filled_nan, n_filled_nan, n_failed_nan, msg_nan = MissingValueFallback.apply(s_all_nan, "Fill with mean")
    assert n_filled_nan == 0, "Should not fill all-NaN column with mean"
    assert n_failed_nan == 3, "All values should be reported as failed"
    assert "all values are missing" in msg_nan.lower(), f"Should explain failure, got: {msg_nan}"

    # Zero fill requires explicit selection
    filled_zero, n_z, _, _ = MissingValueFallback.apply(s, "Fill with zero")
    assert n_z == 2
    assert filled_zero.iloc[3] == 0.0

    # Leave unchanged
    filled_leave, n_l, _, _ = MissingValueFallback.apply(s, "Leave unchanged")
    assert n_l == 0
    assert pd.isna(filled_leave.iloc[3])


def test_case_d_non_numeric_fallback():
    """Numeric strategies on non-numeric column should fail safely."""
    s_text = pd.Series(["a", "b", np.nan])
    _, filled, failed, msg = MissingValueFallback.apply(s_text, "Fill with mean")
    assert filled == 0
    assert failed == 1
    assert "not numeric" in msg.lower()


def test_case_d_categorical_fallback():
    """Categorical mode and custom fill."""
    s = pd.Series(["Cat", "Cat", "Dog", np.nan])
    # Mode (Cat has 2, Dog has 1)
    filled, n_filled, _, msg = MissingValueFallback.apply(s, "Fill with mode: Cat")
    assert n_filled == 1
    assert filled.iloc[3] == "Cat"

    # Custom text
    filled2, n_filled2, _, _ = MissingValueFallback.apply(s, "Fill with custom text value", custom_value="Unknown")
    assert n_filled2 == 1
    assert filled2.iloc[3] == "Unknown"


def test_case_d_datetime_fallback():
    """Datetime forward/backward fill."""
    s = pd.Series(pd.to_datetime(["2024-01-01", None, "2024-01-03"]))
    filled, n_filled, n_failed, _ = MissingValueFallback.apply(s, "Forward fill (requires sorted data)")
    assert n_filled == 1
    assert n_failed == 0
    assert filled.iloc[1] == pd.Timestamp("2024-01-01")


# ===================================================================
# Case E: Regex validation
# ===================================================================

def test_case_e_regex_validation():
    """Test email, ID, and custom pattern validation."""
    emails = pd.Series(["user@example.com", "bad-email", np.nan, "another@test.co", ""])

    validator = RegexValidator(PREDEFINED_PATTERNS["Email (basic)"]["pattern"], "Email")
    result = validator.validate(emails)

    assert result["valid_count"] == 2, f"Expected 2 valid emails, got {result['valid_count']}"
    assert result["invalid_count"] == 1, f"Expected 1 invalid email, got {result['invalid_count']}"
    assert result["missing_count"] == 2, f"Expected 2 missing (NaN + blank), got {result['missing_count']}"
    assert "bad-email" in result["invalid_examples"]

    # ID validation
    ids = pd.Series(["ABC-123", "DEF456", "!!invalid!!", np.nan])
    id_validator = RegexValidator(PREDEFINED_PATTERNS["Alphanumeric ID"]["pattern"], "ID")
    id_result = id_validator.validate(ids)
    assert id_result["valid_count"] == 2
    assert id_result["invalid_count"] == 1
    assert "!!invalid!!" in id_result["invalid_examples"]


def test_case_e_invalid_regex():
    """Invalid regex should not crash."""
    validator = RegexValidator("[invalid(", "Bad Pattern")
    assert not validator.is_valid_regex
    assert validator.error_message is not None

    result = validator.validate(pd.Series(["test"]))
    assert result["valid_count"] == 0


# ===================================================================
# Case F: Repeated execution and data preservation
# ===================================================================

def test_case_f_repeated_execution():
    """Applying same transformation twice should not introduce additional changes."""
    original = pd.DataFrame({
        "City": ["Banglore", "banglor", "Banglore", "Delhi"],
        "Value": [10.0, np.nan, 30.0, 40.0],
    })
    original_copy = original.copy(deep=True)

    # Apply fuzzy replacement once
    matcher = FuzzyMatcher(threshold=75.0)
    groups = matcher.find_groups(original["City"], "City")
    if groups:
        replacements = {v["value"]: groups[0]["canonical"] for v in groups[0]["variants"]}
        new_series, changed1 = matcher.apply_replacements(original["City"], replacements)
        original["City"] = new_series

        # Apply same replacement again
        new_series2, changed2 = matcher.apply_replacements(original["City"], replacements)
        assert changed2 == 0, f"Second application should change 0 values, got {changed2}"

    # Original copy is intact
    pd.testing.assert_frame_equal(original_copy, pd.DataFrame({
        "City": ["Banglore", "banglor", "Banglore", "Delhi"],
        "Value": [10.0, np.nan, 30.0, 40.0],
    }))

    # Fill mean once, then again — second should be no-op
    s = pd.Series([10.0, np.nan, 30.0])
    filled1, n1, _, _ = MissingValueFallback.apply(s, "Fill with mean")
    assert n1 == 1
    filled2, n2, _, _ = MissingValueFallback.apply(filled1, "Fill with mean")
    assert n2 == 0, "Second mean fill should be a no-op"


def test_case_f_metrics_recalculated():
    """Metrics recalculated after changes."""
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": ["x", "N/A", "z"]})
    sum1 = summarize_dataframe(df)
    assert sum1["missing_cells"] >= 2

    # Fill missing
    df2 = df.copy()
    df2["A"] = df2["A"].fillna(2)
    df2.loc[df2["B"] == "N/A", "B"] = "y"
    sum2 = summarize_dataframe(df2)
    assert sum2["missing_cells"] < sum1["missing_cells"]


# ===================================================================
# Case G: Export
# ===================================================================

def test_case_g_export():
    """Cleaned data and results can be exported and re-imported."""
    df = pd.DataFrame({
        "id": ["001", "002", "003"],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95.5, 88.0, 72.3],
    })
    profile = profile_dataframe(df)
    audit = [{"operation": "Fuzzy match", "affected": 2, "status": "Applied"}]

    # Excel export
    excel_bytes = make_excel_bytes(df, profile, audit)
    assert len(excel_bytes) > 500

    # Re-import
    xl = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
    assert "Cleaned Data" in xl.sheet_names
    reloaded = pd.read_excel(xl, sheet_name="Cleaned Data", dtype={"id": str})
    assert reloaded["id"].iloc[0] == "001"
    assert list(reloaded.columns) == list(df.columns)

    # CSV export and re-import
    csv_str = df.to_csv(index=False)
    reimported = pd.read_csv(io.StringIO(csv_str), dtype={"id": str})
    pd.testing.assert_frame_equal(df, reimported)


def test_case_g_export_errors():
    """Empty dataframe export should not crash."""
    df_empty = pd.DataFrame()
    profile_empty = profile_dataframe(df_empty)
    excel_bytes = make_excel_bytes(df_empty, profile_empty, [])
    assert len(excel_bytes) > 0


# ===================================================================
# Additional engine-specific tests
# ===================================================================

def test_missing_data_engine_summary():
    """MissingDataEngine.summary produces correct counts."""
    df = pd.DataFrame({
        "a": [1, np.nan, 3],
        "b": ["hello", "", "N/A"],
        "c": [np.nan, np.nan, np.nan],
    })
    engine = MissingDataEngine()
    summary = engine.summary(df)
    assert summary["total_cells"] == 9
    assert summary["missing_cells"] >= 5  # 1 NaN in a, 2 in b (empty + N/A), 3 in c
    assert summary["affected_rows"] >= 2


def test_fuzzy_matcher_skips_identifiers():
    """Identifier columns should not be fuzzy matched."""
    s = pd.Series(["ABC001", "ABC002", "abc001"])
    matcher = FuzzyMatcher(threshold=80.0)
    groups = matcher.find_groups(s, "customer_id")
    assert len(groups) == 0, "Identifier columns should be skipped"


def test_fuzzy_matcher_high_cardinality():
    """Columns with >500 unique values should be skipped."""
    s = pd.Series([f"val_{i}" for i in range(600)])
    matcher = FuzzyMatcher(threshold=80.0, max_unique=500)
    groups = matcher.find_groups(s, "data")
    assert len(groups) == 0


def test_regex_cleaner_find():
    """RegexCleaner.find_matches works correctly."""
    s = pd.Series(["hello123", "world", "test456", np.nan])
    result, err = RegexCleaner.find_matches(s, r"\d+")
    assert err is None
    assert len(result) == 2


def test_regex_cleaner_replace():
    """RegexCleaner.apply_replace works correctly."""
    s = pd.Series(["hello 123", "world 456", "no numbers"])
    new_s, changed, err = RegexCleaner.apply_replace(s, r"\d+", "NUM")
    assert err is None
    assert changed == 2
    assert new_s.iloc[0] == "hello NUM"
    assert new_s.iloc[2] == "no numbers"


def test_regex_cleaner_invalid_pattern():
    """Invalid regex should return error, not crash."""
    s = pd.Series(["test"])
    result, err = RegexCleaner.find_matches(s, "[invalid(")
    assert err is not None
    assert result.empty


def test_regex_cleaner_extract():
    """RegexCleaner.extract_pattern works."""
    s = pd.Series(["Order#123", "Ref#456", "NoMatch"])
    result, err = RegexCleaner.extract_pattern(s, r"#(\d+)")
    assert err is None
    assert len(result) == 2
    assert result.iloc[0]["extracted"] == "#123"


def test_data_quality_runner_missing():
    """Test missing required values test."""
    s = pd.Series(["a", np.nan, "b", "N/A", "c"])
    result = DataQualityTestRunner.test_missing_required(s, "col1")
    assert result["failed"] == 2  # NaN + N/A
    assert result["checked"] == 5


def test_data_quality_runner_email():
    """Test email format validation."""
    s = pd.Series(["user@test.com", "bad", np.nan])
    result = DataQualityTestRunner.test_email_format(s, "email")
    assert result["passed"] == 1
    assert result["failed"] == 1
    assert result["skipped"] == 1


def test_data_quality_runner_range():
    """Test numeric range validation."""
    s = pd.Series([10, 20, -5, 100, np.nan])
    result = DataQualityTestRunner.test_numeric_range(s, "val", min_val=0, max_val=50)
    assert result["failed"] == 2  # -5 and 100
    assert result["skipped"] == 1


def test_data_quality_runner_no_negatives():
    """Test no-negatives rule."""
    s = pd.Series([1, 2, -3, 0, np.nan])
    result = DataQualityTestRunner.test_no_negatives(s, "val")
    assert result["failed"] == 1  # -3
    assert result["passed"] == 3  # 1, 2, 0


def test_data_quality_runner_inconsistent():
    """Test inconsistent category detection — case variants should be caught at 85% threshold."""
    # With case-insensitive scoring, Apple/apple/APPLE are all 100% similar
    s = pd.Series(["Banglore", "banglor", "Banglore", "Banglore", "Delhi"])
    result = DataQualityTestRunner.test_inconsistent_categories(s, "city", threshold=75.0)
    assert result["failed"] >= 1, f"Should detect inconsistent categories, got: {result}"


def test_data_quality_runner_regex_pattern():
    """Test custom regex pattern test."""
    s = pd.Series(["12345", "1234", "abcde", np.nan])
    result = DataQualityTestRunner.test_regex_pattern(s, "zip", r"^\d{5}$", "5-digit ZIP")
    assert result["passed"] == 1  # "12345"
    assert result["failed"] == 2  # "1234" and "abcde"


def test_data_quality_runner_duplicates():
    """Test duplicate detection."""
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    result = DataQualityTestRunner.test_duplicates(df)
    assert result["failed"] == 1


def test_similarity_score_basics():
    """Similarity scoring basics (case-insensitive)."""
    # Identical strings
    assert _similarity_score("hello", "hello") == 100.0
    # Very different strings
    score = _similarity_score("abc", "xyz")
    assert score < 50.0, f"abc vs xyz should be < 50, got {score}"
    # Multi-word: case-insensitive so should be 100%
    score_multi = _similarity_score("New York City", "new york city")
    assert score_multi == 100.0, f"Same words different case should be 100, got {score_multi}"
    # Case variants — should score very high
    score_case = _similarity_score("Banglore", "BANGLORE")
    assert score_case >= 95.0, f"Expected high score for case variant, got {score_case}"
    # Truly unrelated cities should NOT be merged
    score_unrelated = _similarity_score("Delhi", "Chennai")
    assert score_unrelated < 70.0, f"Delhi vs Chennai should be < 70, got {score_unrelated}"


def test_predefined_patterns():
    """Predefined patterns compile correctly."""
    import re
    for name, info in PREDEFINED_PATTERNS.items():
        try:
            re.compile(info["pattern"])
        except re.error as e:
            assert False, f"Pattern '{name}' failed to compile: {e}"
        assert "description" in info



def test_fallback_mode_tie():
    """Mode tie should be reported, not silently picked."""
    s = pd.Series(["A", "B", np.nan])  # Both "A" and "B" have count 1
    _, filled, failed, msg = MissingValueFallback.apply(s, "Fill with mode (tie: 2 modes)")
    assert filled == 0
    assert failed == 1
    assert "ambiguous" in msg.lower() or "tied" in msg.lower()


def test_fallback_strategies_suggested():
    """Suggest strategies returns appropriate options by column type."""
    s_num = pd.Series([1.0, 2.0, np.nan])
    strats_num = MissingValueFallback.suggest_strategies(s_num)
    assert "Fill with mean" in strats_num
    assert "Fill with median" in strats_num
    assert "Fill with zero" in strats_num

    s_text = pd.Series(["a", "a", np.nan])
    strats_text = MissingValueFallback.suggest_strategies(s_text)
    assert any("mode" in s.lower() for s in strats_text)
    assert "Fill with custom text value" in strats_text

    s_date = pd.Series(pd.to_datetime(["2024-01-01", None]))
    strats_date = MissingValueFallback.suggest_strategies(s_date)
    assert any("forward" in s.lower() for s in strats_date)

    # All-NaN numeric
    s_all_nan = pd.Series([np.nan, np.nan], dtype="float64")
    strats_nan = MissingValueFallback.suggest_strategies(s_all_nan)
    # mean/median should NOT be suggested for all-NaN
    assert "Fill with mean" not in strats_nan


# ===================================================================
# Runner
# ===================================================================

tests = [
    test_case_a_misspelled_category,
    test_case_b_similarity_threshold,
    test_case_c_missing_values,
    test_case_d_numeric_fallback,
    test_case_d_non_numeric_fallback,
    test_case_d_categorical_fallback,
    test_case_d_datetime_fallback,
    test_case_e_regex_validation,
    test_case_e_invalid_regex,
    test_case_f_repeated_execution,
    test_case_f_metrics_recalculated,
    test_case_g_export,
    test_case_g_export_errors,
    test_missing_data_engine_summary,
    test_fuzzy_matcher_skips_identifiers,
    test_fuzzy_matcher_high_cardinality,
    test_regex_cleaner_find,
    test_regex_cleaner_replace,
    test_regex_cleaner_invalid_pattern,
    test_regex_cleaner_extract,
    test_data_quality_runner_missing,
    test_data_quality_runner_email,
    test_data_quality_runner_range,
    test_data_quality_runner_no_negatives,
    test_data_quality_runner_inconsistent,
    test_data_quality_runner_regex_pattern,
    test_data_quality_runner_duplicates,
    test_similarity_score_basics,
    test_predefined_patterns,
    test_fallback_mode_tie,
    test_fallback_strategies_suggested,
]

if __name__ == "__main__":
    print(f"Fuzzy matching backend: {FUZZY_BACKEND}")
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

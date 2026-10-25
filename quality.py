"""
CleanSight Data Quality Engine
Provides robust, reliable, and mathematically consistent functions for:
- Missing-value and placeholder detection
- Data type inference and identifier protection
- Currency, unit, and numeric text parsing
- Date parsing and ambiguity detection
- Duplicate row and near-duplicate detection
- IQR outlier detection
- Profiling summaries and invariant-preserving quality metrics
- Before-and-after comparison analysis
"""

from __future__ import annotations

import io
import re
from typing import Any
import numpy as np
import pandas as pd

# Standard recognized missing-value placeholders (case-insensitive)
STANDARD_MISSING_PLACEHOLDERS = frozenset({
    "n/a",
    "na",
    "null",
    "none",
    "not available",
    "nan",
    "nil",
    "<na>",
    "n.a.",
    "n/d",
    "#n/a",
})

# Ambiguous placeholders that should be flagged for review rather than blindly coerced
AMBIGUOUS_PLACEHOLDERS = frozenset({
    "-",
    "--",
    "---",
    "?",
    "??",
    "???",
    ".",
    "unknown",
    "pending",
    "tbd",
})

# Regex to identify columns that are likely IDs, codes, or postal numbers where leading zeros matter
IDENTIFIER_COL_REGEX = re.compile(
    r"(?i)(^id$|_id$|^id_|\bid\b|\bcode\b|\bzip\b|\bpostal\b|\bphone\b|\bssn\b|\baccount\b|\bserial\b|\bref\b|\bpin\b|\bemployee_id\b|\bcustomer_id\b|\border_id\b)"
)


def is_text_column(series: pd.Series) -> bool:
    """Check if series has text/object/string/categorical data type."""
    return (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def is_numeric_column(series: pd.Series) -> bool:
    """Check if series has true numeric data type (excluding boolean)."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def get_missing_mask(series: pd.Series, include_placeholders: bool = True) -> pd.Series:
    """
    Produce a boolean mask of genuine missing values:
    - Standard pandas nulls / NaNs / NaT / None
    - Whitespace-only strings
    - Recognized missing-value placeholders (e.g. 'N/A', 'null', 'none', 'Not Available')
    """
    base_null = series.isna()
    if not is_text_column(series):
        return base_null

    # For text series, detect blank whitespace and recognized placeholders
    str_series = series.astype(str).str.strip()
    blank_mask = str_series.eq("")
    if include_placeholders:
        placeholder_mask = str_series.str.lower().isin(STANDARD_MISSING_PLACEHOLDERS)
        return base_null | blank_mask | placeholder_mask
    return base_null | blank_mask


def find_ambiguous_placeholders(series: pd.Series) -> dict[str, int]:
    """Find occurrences of ambiguous placeholder strings like '-' or '?'."""
    if not is_text_column(series):
        return {}
    nonempty = series.dropna().astype(str).str.strip()
    counts: dict[str, int] = {}
    lower_s = nonempty.str.lower()
    for placeholder in AMBIGUOUS_PLACEHOLDERS:
        cnt = int((lower_s == placeholder).sum())
        if cnt > 0:
            counts[placeholder] = cnt
    return counts


def has_meaningful_leading_zeros(series: pd.Series) -> bool:
    """Check if a text column contains digit strings with leading zeros (e.g. '00142', '07')."""
    if not is_text_column(series):
        return False
    nonempty = series.dropna().astype(str).str.strip()
    if len(nonempty) == 0:
        return False
    # Check if any values start with '0', are all digits, and have length > 1
    leading_zero_mask = nonempty.str.startswith("0") & (nonempty.str.len() > 1) & nonempty.str.isdigit()
    return bool(leading_zero_mask.any())


def is_identifier_like(series: pd.Series, col_name: str = "") -> bool:
    """Determine if a column looks like an identifier, phone, postal code, or account code."""
    if col_name and IDENTIFIER_COL_REGEX.search(col_name):
        return True
    return has_meaningful_leading_zeros(series)


def clean_and_parse_numeric_value(value: Any) -> tuple[bool, float | int | None]:
    """
    Parse a single value into numeric.
    Handles:
    - Native numbers (int, float)
    - Strings with currency symbols (₹, $, €, £, ¥)
    - Thousand separator commas (1,299.50)
    - Trailing units ('5 pcs', '10.5 kg', '250 units')
    - Accounting parentheses ('(1,200.50)' -> -1200.50)
    - Minus symbols ('−' -> '-')
    - Percentages ('15%' -> 15.0, without scaling)
    Returns:
    - (is_success, numeric_value_or_none)
    """
    if pd.isna(value):
        return False, None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return True, float(value)

    raw_text = str(value).strip()
    if not raw_text:
        return False, None

    # Protect meaningful leading zeros on purely numeric strings (e.g. '00123')
    if raw_text.startswith("0") and len(raw_text) > 1 and raw_text.isdigit():
        return False, None

    text = raw_text

    # Accounting parentheses: '(100)' -> '-100'
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()

    # Normalize unicode minus
    text = text.replace("−", "-")

    # Remove currency symbols and whitespace
    text = re.sub(r"[$₹€£¥\s]", "", text)

    # Percentage: keep numeric part without multiplying or dividing
    if text.endswith("%"):
        text = text[:-1].strip()

    # Trailing unit words like '5 pcs', '10.5 kg', '100 units'
    unit_match = re.match(r"^([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)\s*[a-zA-Z]+$", text)
    if unit_match:
        text = unit_match.group(1)

    # Remove thousand commas only if standard format (e.g. 1,299.50 or 1,299)
    if "," in text:
        # Check standard thousand grouping or simple thousand removal
        text = text.replace(",", "")

    if is_negative and not text.startswith("-"):
        text = "-" + text

    try:
        val = float(text)
        if np.isnan(val) or np.isinf(val):
            return False, None
        return True, val
    except (ValueError, TypeError):
        return False, None


def numeric_conversion_with_stats(
    series: pd.Series, preserve_leading_zeros: bool = True
) -> tuple[pd.Series, int, int]:
    """
    Convert a series to numeric, tracking successfully parsed and failed values.
    Returns: (converted_series, success_count, failed_count)
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
        return converted, int(converted.notna().sum()), 0

    converted_values: list[float | None] = []
    success_count = 0
    failed_count = 0

    for val in series:
        if pd.isna(val) or (isinstance(val, str) and not val.strip()):
            converted_values.append(np.nan)
            continue
        success, parsed = clean_and_parse_numeric_value(val)
        if success and parsed is not None:
            converted_values.append(parsed)
            success_count += 1
        else:
            converted_values.append(np.nan)
            failed_count += 1

    result_series = pd.Series(converted_values, index=series.index, dtype="float64")
    return result_series, success_count, failed_count


def parse_date_series(
    series: pd.Series, convention: str = "auto"
) -> tuple[pd.Series, int, int, int]:
    """
    Parse a series into datetime with convention control.
    convention options: 'auto', 'dayfirst', 'monthfirst', 'iso'
    Returns: (converted_series, success_count, failed_count, ambiguous_count)
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, int(series.notna().sum()), 0, 0

    nonempty_mask = series.notna() & series.astype(str).str.strip().ne("")
    ambiguous_count = 0

    # Detect ambiguous dates e.g. 03/04/2025 where both day & month <= 12
    for val in series[nonempty_mask]:
        val_str = str(val).strip()
        parts = re.findall(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", val_str)
        if parts:
            p1, p2, _ = parts[0]
            if 1 <= int(p1) <= 12 and 1 <= int(p2) <= 12 and int(p1) != int(p2):
                ambiguous_count += 1

    dayfirst = True if convention == "dayfirst" else False if convention == "monthfirst" else None

    try:
        if convention == "iso":
            converted = pd.to_datetime(series, errors="coerce", format="ISO8601")
        elif dayfirst is not None:
            converted = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=dayfirst)
        else:
            converted = pd.to_datetime(series, errors="coerce", format="mixed")
    except Exception:
        converted = pd.to_datetime(series, errors="coerce")

    success_count = int((nonempty_mask & converted.notna()).sum())
    failed_count = int((nonempty_mask & converted.isna()).sum())
    return converted, success_count, failed_count, ambiguous_count


def find_numeric_text_candidates(df: pd.DataFrame) -> list[str]:
    """
    Find text columns that are suitable numeric measurements.
    Strictly excludes identifier-like columns and columns with leading zeros.
    """
    candidates: list[str] = []
    for col in df.columns:
        series = df[col]
        col_str = str(col)
        if not is_text_column(series):
            continue
        if is_identifier_like(series, col_str):
            continue

        nonempty = series.dropna().astype(str).str.strip()
        nonempty = nonempty[nonempty.ne("")]
        if len(nonempty) < 2:
            continue

        _, success_count, _ = numeric_conversion_with_stats(nonempty)
        ratio = success_count / len(nonempty) if len(nonempty) else 0.0
        if ratio >= 0.75 and success_count >= 2:
            candidates.append(col_str)
    return candidates


def find_date_text_candidates(df: pd.DataFrame) -> list[str]:
    """
    Find text columns that look like dates.
    Excludes pure numbers and identifier-like columns.
    """
    candidates: list[str] = []
    for col in df.columns:
        series = df[col]
        col_str = str(col)
        if not is_text_column(series):
            continue
        if is_identifier_like(series, col_str):
            continue

        nonempty = series.dropna().astype(str).str.strip()
        nonempty = nonempty[nonempty.ne("")]
        if len(nonempty) < 2:
            continue

        # Avoid treating pure numbers as dates
        _, num_success, _ = numeric_conversion_with_stats(nonempty)
        if (num_success / len(nonempty)) > 0.80:
            continue

        try:
            parsed = pd.to_datetime(nonempty, errors="coerce", format="mixed")
        except Exception:
            parsed = pd.to_datetime(nonempty, errors="coerce")

        ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        if ratio >= 0.75 and parsed.notna().sum() >= 2:
            candidates.append(col_str)
    return candidates


def detect_exact_duplicates(df: pd.DataFrame) -> int:
    """Exact duplicate row count across all columns."""
    if len(df) <= 1 or len(df.columns) == 0:
        return 0
    return int(df.duplicated(keep="first").sum())


def detect_near_duplicates(df: pd.DataFrame) -> int:
    """
    Candidate near-duplicate rows that differ only by whitespace or letter case.
    Does NOT modify df. Returns count of near-duplicates that are not already exact duplicates.
    """
    if len(df) <= 1 or len(df.columns) == 0:
        return 0
    exact = detect_exact_duplicates(df)
    normalized = df.copy()
    for col in normalized.columns:
        if is_text_column(normalized[col]):
            normalized[col] = normalized[col].astype(str).str.strip().str.lower()
    norm_dups = int(normalized.duplicated(keep="first").sum())
    near = max(0, norm_dups - exact)
    return near


def detect_iqr_outliers(
    series: pd.Series,
) -> tuple[int, float | None, float | None, float | None, float | None]:
    """
    Detect outliers using standard IQR method:
    Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
    Lower boundary = Q1 - 1.5 * IQR, Upper boundary = Q3 + 1.5 * IQR
    Returns: (outlier_count, q1, q3, lower_boundary, upper_boundary)
    """
    if not is_numeric_column(series):
        return 0, None, None, None, None

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 4:
        return 0, None, None, None, None

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1

    if pd.isna(iqr) or iqr <= 0:
        return 0, q1, q3, q1, q3

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (values < lower) | (values > upper)
    outlier_count = int(outlier_mask.sum())
    return outlier_count, q1, q3, lower, upper


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an exhaustive and accurate column-level data quality profile.
    Maintains invariant: per-column missing count uses get_missing_mask.
    """
    rows: list[dict[str, Any]] = []
    numeric_candidates = set(find_numeric_text_candidates(df))
    date_candidates = set(find_date_text_candidates(df))

    for col in df.columns:
        series = df[col]
        col_str = str(col)
        total_rows = int(len(series))

        missing_mask = get_missing_mask(series, include_placeholders=True)
        missing_count = int(missing_mask.sum())
        missing_pct = round((missing_count / total_rows * 100) if total_rows else 0.0, 2)
        completeness_pct = round(100.0 - missing_pct, 2)

        # Breakdown of missing reasons
        raw_null_count = int(series.isna().sum())
        blank_text_count = 0
        placeholder_count = 0
        if is_text_column(series):
            str_s = series.astype(str).str.strip()
            blank_text_count = int((str_s == "").sum())
            placeholder_count = int(str_s.str.lower().isin(STANDARD_MISSING_PLACEHOLDERS).sum())

        outlier_count, _, _, _, _ = detect_iqr_outliers(series)

        # Status tags
        is_id = is_identifier_like(series, col_str)
        if col_str in numeric_candidates:
            num_status = "Yes — review"
        elif is_id:
            num_status = "Protected identifier / code"
        else:
            num_status = "No / not detected"

        date_status = "Yes — review" if col_str in date_candidates else "No / not detected"

        # Basic numeric stats for quick profile inspection
        mean_val = None
        std_val = None
        min_val = None
        max_val = None
        if is_numeric_column(series):
            clean_nums = series.dropna()
            if len(clean_nums):
                mean_val = round(float(clean_nums.mean()), 2)
                std_val = round(float(clean_nums.std()), 2) if len(clean_nums) > 1 else 0.0
                min_val = round(float(clean_nums.min()), 2)
                max_val = round(float(clean_nums.max()), 2)

        rows.append({
            "Column": col_str,
            "Data type": str(series.dtype),
            "Rows": total_rows,
            "Missing": missing_count,
            "Missing %": missing_pct,
            "Completeness %": completeness_pct,
            "Unique values": int(series.nunique(dropna=True)),
            "Explicit nulls": raw_null_count,
            "Blank text cells": blank_text_count,
            "Placeholders": placeholder_count,
            "Numbers stored as text?": num_status,
            "Date-like text?": date_status,
            "IQR outliers": outlier_count,
            "Mean": mean_val,
            "Std Dev": std_val,
            "Min": min_val,
            "Max": max_val,
        })

    return pd.DataFrame(rows)


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize dataset quality metrics.
    Guarantees strict invariants:
    - total_cells = rows * columns
    - missing_cells <= total_cells
    - completeness_percent = 100 * (1 - missing_cells / total_cells)
    - exact_duplicate_rows calculated consistently
    - sum of column missing counts == missing_cells
    """
    rows = int(df.shape[0])
    cols = int(df.shape[1])
    total_cells = rows * cols

    if total_cells == 0:
        return {
            "rows": rows,
            "columns": cols,
            "total_cells": 0,
            "missing_cells": 0,
            "missing_percent": 0.0,
            "exact_duplicate_rows": 0,
            "near_duplicate_rows": 0,
            "completeness_percent": 100.0,
            "blank_text_cells": 0,
            "placeholder_cells": 0,
            "total_outliers": 0,
        }

    # Sum per-column missing masks
    missing_cells = 0
    blank_text_cells = 0
    placeholder_cells = 0
    total_outliers = 0

    for col in df.columns:
        s = df[col]
        mask = get_missing_mask(s, include_placeholders=True)
        missing_cells += int(mask.sum())
        if is_text_column(s):
            str_s = s.astype(str).str.strip()
            blank_text_cells += int((str_s == "").sum())
            placeholder_cells += int(str_s.str.lower().isin(STANDARD_MISSING_PLACEHOLDERS).sum())
        out_cnt, _, _, _, _ = detect_iqr_outliers(s)
        total_outliers += out_cnt

    missing_percent = round((missing_cells / total_cells * 100), 2)
    completeness_percent = round(((total_cells - missing_cells) / total_cells * 100), 2)
    # Ensure bounds [0.0, 100.0]
    completeness_percent = max(0.0, min(100.0, completeness_percent))

    exact_dups = detect_exact_duplicates(df)
    near_dups = detect_near_duplicates(df)

    return {
        "rows": rows,
        "columns": cols,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percent": missing_percent,
        "exact_duplicate_rows": exact_dups,
        "near_duplicate_rows": near_dups,
        "completeness_percent": completeness_percent,
        "blank_text_cells": blank_text_cells,
        "placeholder_cells": placeholder_cells,
        "total_outliers": total_outliers,
    }


def compute_quality_issue_categories(
    df: pd.DataFrame, profile: pd.DataFrame, summary: dict[str, Any]
) -> pd.DataFrame:
    """
    Compile distinct issue categories with explicit non-overlapping metrics.
    Clearly distinguishes cell counts, row counts, and column counts.
    """
    issues: list[dict[str, Any]] = []

    # 1. Missing data cells
    if summary["missing_cells"] > 0:
        issues.append({
            "Issue Category": "Missing Cells (Explicit / Empty)",
            "Count": summary["missing_cells"],
            "Unit": "cells",
            "Description": "Cells containing null, empty whitespace, or recognized missing placeholders",
        })

    # 2. Exact Duplicate Rows
    if summary["exact_duplicate_rows"] > 0:
        issues.append({
            "Issue Category": "Exact Duplicate Rows",
            "Count": summary["exact_duplicate_rows"],
            "Unit": "rows",
            "Description": "Identical rows across all columns",
        })

    # 3. Near-Duplicate Rows
    if summary.get("near_duplicate_rows", 0) > 0:
        issues.append({
            "Issue Category": "Candidate Near-Duplicates",
            "Count": summary["near_duplicate_rows"],
            "Unit": "rows",
            "Description": "Rows differing only by casing or leading/trailing whitespace",
        })

    # 4. Outliers
    if summary.get("total_outliers", 0) > 0:
        issues.append({
            "Issue Category": "IQR Outliers Flagged",
            "Count": summary["total_outliers"],
            "Unit": "observations",
            "Description": "Values falling outside 1.5 × IQR boundaries in numeric fields",
        })

    # 5. Numbers stored as text
    num_text_cols = int((profile["Numbers stored as text?"] == "Yes — review").sum()) if not profile.empty else 0
    if num_text_cols > 0:
        issues.append({
            "Issue Category": "Numbers Stored as Text",
            "Count": num_text_cols,
            "Unit": "columns",
            "Description": "Columns containing formatted numeric strings (currency, commas, units)",
        })

    # 6. Date-like text
    date_text_cols = int((profile["Date-like text?"] == "Yes — review").sum()) if not profile.empty else 0
    if date_text_cols > 0:
        issues.append({
            "Issue Category": "Date-Like Text Columns",
            "Count": date_text_cols,
            "Unit": "columns",
            "Description": "Columns with unparsed date strings",
        })

    return pd.DataFrame(issues)


def calculate_correlation_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Calculate Pearson correlation matrix for suitable numeric columns.
    Excludes constant columns (std == 0 or nunique <= 1) and handles missing values.
    Returns: (correlation_df_or_None, list_of_eligible_columns)
    """
    eligible_cols: list[str] = []
    for col in df.columns:
        s = df[col]
        if is_numeric_column(s):
            clean = s.dropna()
            if len(clean) >= 2 and clean.nunique() > 1 and float(clean.std()) > 0:
                eligible_cols.append(str(col))

    if len(eligible_cols) < 2:
        return None, eligible_cols

    corr = df[eligible_cols].corr(method="pearson")
    return corr, eligible_cols


def sample_dataframe_for_missingness(
    df: pd.DataFrame, selected_cols: list[str] | None = None, max_rows: int = 200
) -> tuple[pd.DataFrame, bool]:
    """
    Create a deterministic binary matrix (0=Present, 1=Missing) for missingness visualization.
    Caps rows to max_rows using deterministic even sampling to ensure instantaneous rendering.
    Returns: (binary_df, was_sampled)
    """
    cols = selected_cols if selected_cols else list(df.columns)
    if not cols or df.empty:
        return pd.DataFrame(), False

    sub_df = df[cols]
    was_sampled = False
    if len(sub_df) > max_rows:
        was_sampled = True
        step = len(sub_df) / max_rows
        sample_indices = [int(i * step) for i in range(max_rows)]
        sub_df = sub_df.iloc[sample_indices]

    binary_df = pd.DataFrame(index=sub_df.index)
    for col in cols:
        binary_df[str(col)] = get_missing_mask(sub_df[col], include_placeholders=True).astype(int)

    return binary_df, was_sampled


def compare_datasets(original_df: pd.DataFrame, working_df: pd.DataFrame) -> dict[str, Any]:
    """
    Produce a rigorous before-and-after comparison between original and working datasets.
    """
    orig_sum = summarize_dataframe(original_df)
    work_sum = summarize_dataframe(working_df)

    # Column-level missing comparison
    col_comp: list[dict[str, Any]] = []
    common_cols = [c for c in original_df.columns if c in working_df.columns]

    for col in common_cols:
        orig_s = original_df[col]
        work_s = working_df[col]
        orig_miss = int(get_missing_mask(orig_s).sum())
        work_miss = int(get_missing_mask(work_s).sum())
        diff = work_miss - orig_miss
        col_comp.append({
            "Column": str(col),
            "Original Missing": orig_miss,
            "Cleaned Missing": work_miss,
            "Reduction": -diff if diff < 0 else 0,
            "Diff": diff,
        })

    return {
        "original_summary": orig_sum,
        "cleaned_summary": work_sum,
        "delta_rows": work_sum["rows"] - orig_sum["rows"],
        "delta_missing": work_sum["missing_cells"] - orig_sum["missing_cells"],
        "delta_duplicates": work_sum["exact_duplicate_rows"] - orig_sum["exact_duplicate_rows"],
        "delta_completeness": round(work_sum["completeness_percent"] - orig_sum["completeness_percent"], 2),
        "column_missing_comparison": pd.DataFrame(col_comp),
    }


def normalize_headers(columns: list[Any]) -> list[str]:
    """Convert column headers to unique snake_case names."""
    result: list[str] = []
    seen: dict[str, int] = {}
    for index, column in enumerate(columns, start=1):
        name = str(column).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name).strip("_") or f"column_{index}"
        if name[0].isdigit():
            name = "col_" + name
        seen[name] = seen.get(name, 0) + 1
        unique = name if seen[name] == 1 else f"{name}_{seen[name]}"
        while unique in result:
            seen[name] += 1
            unique = f"{name}_{seen[name]}"
        result.append(unique)
    return result


def make_excel_bytes(df: pd.DataFrame, profile: pd.DataFrame, audit: list[dict[str, Any]]) -> bytes:
    """Generate Excel workbook bytes with Cleaned Data, Profile, and Audit Log sheets."""
    output = io.BytesIO()
    audit_df = pd.DataFrame(audit) if audit else pd.DataFrame([{"Status": "No cleaning operations logged"}])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        profile.to_excel(writer, index=False, sheet_name="Data Quality Profile")
        audit_df.to_excel(writer, index=False, sheet_name="Cleaning Log")
    output.seek(0)
    return output.getvalue()


# ==============================================================================
# ADVANCED ENGINES — Added for rule-based data quality enhancement
# ==============================================================================

# ---------------------------------------------------------------------------
#  Fuzzy matching backend selection
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process

    FUZZY_BACKEND = "rapidfuzz"
except ImportError:
    FUZZY_BACKEND = "difflib"

import difflib


# ---------------------------------------------------------------------------
#  1. MISSING DATA DETECTION ENGINE
# ---------------------------------------------------------------------------

class MissingDataEngine:
    """Reusable missing-data detection with configurable placeholders.

    Detects: NaN/None/NaT, empty strings, whitespace-only strings,
    recognized placeholders, and optional user-defined placeholders.
    """

    def __init__(self, extra_placeholders: list[str] | None = None):
        self.standard = STANDARD_MISSING_PLACEHOLDERS
        self.extra: frozenset[str] = frozenset(
            p.strip().lower() for p in (extra_placeholders or []) if p.strip()
        )
        self.all_placeholders = self.standard | self.extra

    def detect(self, series: pd.Series) -> pd.DataFrame:
        """Return a DataFrame with columns: index, original_value,
        reason ('null','empty','whitespace','placeholder','extra_placeholder'),
        normalized, proposed_action.
        """
        records: list[dict[str, Any]] = []
        for idx, val in series.items():
            if pd.isna(val):
                records.append({
                    "row": idx, "original_value": repr(val),
                    "reason": "null (NaN/None/NaT)",
                    "normalized": "NaN",
                    "proposed_action": "Treat as missing",
                })
                continue
            if not isinstance(val, str):
                continue
            stripped = val.strip()
            if stripped == "":
                reason = "whitespace-only" if val != "" else "empty string"
                records.append({
                    "row": idx, "original_value": repr(val),
                    "reason": reason, "normalized": "NaN",
                    "proposed_action": "Convert to NaN",
                })
                continue
            low = stripped.lower()
            if low in self.standard:
                records.append({
                    "row": idx, "original_value": val,
                    "reason": f"recognized placeholder",
                    "normalized": "NaN",
                    "proposed_action": "Convert to NaN",
                })
            elif low in self.extra:
                records.append({
                    "row": idx, "original_value": val,
                    "reason": f"user-defined placeholder",
                    "normalized": "NaN",
                    "proposed_action": "Convert to NaN (user-configured)",
                })
        return pd.DataFrame(records)

    def summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Produce dataset-level missing-data summary."""
        total_cells = df.shape[0] * df.shape[1]
        missing_count = 0
        col_details: list[dict[str, Any]] = []
        affected_rows: set[int] = set()

        for col in df.columns:
            det = self.detect(df[col])
            n_miss = len(det)
            missing_count += n_miss
            if n_miss > 0:
                affected_rows.update(det["row"].tolist())
            col_details.append({
                "column": str(col),
                "missing_count": n_miss,
                "missing_pct": round(n_miss / len(df) * 100, 2) if len(df) else 0.0,
            })

        return {
            "total_cells": total_cells,
            "missing_cells": missing_count,
            "missing_pct_overall": round(missing_count / total_cells * 100, 2) if total_cells else 0.0,
            "affected_rows": len(affected_rows),
            "columns": col_details,
        }


# ---------------------------------------------------------------------------
#  2. MISSING-VALUE FALLBACK ENGINE
# ---------------------------------------------------------------------------

class MissingValueFallback:
    """Type-aware missing-value imputation with safety checks."""

    @staticmethod
    def suggest_strategies(series: pd.Series, col_name: str = "") -> list[str]:
        """Return list of applicable strategy names for the column."""
        strategies = ["Leave unchanged"]
        if is_numeric_column(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                strategies.extend(["Fill with mean", "Fill with median"])
            strategies.append("Fill with zero")
            strategies.append("Fill with custom numeric value")
        elif pd.api.types.is_datetime64_any_dtype(series):
            strategies.append("Fill with custom date")
            strategies.append("Forward fill (requires sorted data)")
            strategies.append("Backward fill (requires sorted data)")
        else:
            modes = series.mode(dropna=True)
            if len(modes) > 0 and len(modes) == 1:
                strategies.append(f"Fill with mode: {str(modes.iloc[0])[:40]}")
            elif len(modes) > 1:
                strategies.append(f"Fill with mode (tie: {len(modes)} modes)")
            strategies.append("Fill with custom text value")
        return strategies

    @staticmethod
    def apply(
        series: pd.Series,
        strategy: str,
        custom_value: Any = None,
        group_by_series: pd.Series | None = None,
    ) -> tuple[pd.Series, int, int, str]:
        """Apply a fill strategy.

        Returns: (filled_series, cells_filled, cells_failed, status_message)
        """
        missing_mask = series.isna()
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            return series.copy(), 0, 0, "No missing values to fill."

        result = series.copy()

        if strategy == "Leave unchanged":
            return result, 0, 0, "Values left unchanged as requested."

        # --- Numeric strategies ---
        if strategy == "Fill with mean":
            if not is_numeric_column(series):
                return result, 0, n_missing, "Cannot compute mean: column is not numeric."
            non_null = series.dropna()
            if len(non_null) == 0:
                return result, 0, n_missing, "Cannot compute mean: all values are missing."
            fill_val = float(non_null.mean())
            if pd.isna(fill_val):
                return result, 0, n_missing, "Mean computation returned NaN. Column may have no valid numbers."
            result = result.fillna(fill_val)
            return result, n_missing, 0, f"Filled {n_missing} missing values with mean ({fill_val:.4g})."

        if strategy == "Fill with median":
            if not is_numeric_column(series):
                return result, 0, n_missing, "Cannot compute median: column is not numeric."
            non_null = series.dropna()
            if len(non_null) == 0:
                return result, 0, n_missing, "Cannot compute median: all values are missing."
            fill_val = float(non_null.median())
            if pd.isna(fill_val):
                return result, 0, n_missing, "Median computation returned NaN."
            result = result.fillna(fill_val)
            return result, n_missing, 0, f"Filled {n_missing} missing values with median ({fill_val:.4g})."

        if strategy == "Fill with zero":
            if not is_numeric_column(series):
                return result, 0, n_missing, "Fill with zero is only valid for numeric columns."
            result = result.fillna(0)
            return result, n_missing, 0, f"Filled {n_missing} missing values with zero."

        if strategy == "Fill with custom numeric value":
            if custom_value is None:
                return result, 0, n_missing, "No custom value provided."
            try:
                fill_val = float(custom_value)
            except (ValueError, TypeError):
                return result, 0, n_missing, f"Cannot convert '{custom_value}' to a number."
            result = result.fillna(fill_val)
            return result, n_missing, 0, f"Filled {n_missing} values with {fill_val}."

        # --- Datetime strategies ---
        if strategy == "Fill with custom date":
            if custom_value is None:
                return result, 0, n_missing, "No custom date provided."
            try:
                fill_val = pd.Timestamp(custom_value)
            except Exception:
                return result, 0, n_missing, f"Cannot parse '{custom_value}' as a date."
            result = result.fillna(fill_val)
            return result, n_missing, 0, f"Filled {n_missing} values with {fill_val}."

        if strategy.startswith("Forward fill"):
            result = result.ffill()
            filled = n_missing - int(result.isna().sum())
            failed = n_missing - filled
            msg = f"Forward-filled {filled} values."
            if failed > 0:
                msg += f" {failed} values at the start could not be filled (no preceding value)."
            return result, filled, failed, msg

        if strategy.startswith("Backward fill"):
            result = result.bfill()
            filled = n_missing - int(result.isna().sum())
            failed = n_missing - filled
            msg = f"Backward-filled {filled} values."
            if failed > 0:
                msg += f" {failed} values at the end could not be filled (no following value)."
            return result, filled, failed, msg

        # --- Categorical / text strategies ---
        if strategy.startswith("Fill with mode"):
            modes = series.mode(dropna=True)
            if len(modes) == 0:
                return result, 0, n_missing, "No mode exists (column may be entirely missing)."
            if len(modes) > 1:
                return result, 0, n_missing, (
                    f"Mode is ambiguous: {len(modes)} values tied "
                    f"({', '.join(str(m)[:20] for m in modes[:3])}…). "
                    "Select a specific value or use custom fill."
                )
            result = result.fillna(modes.iloc[0])
            return result, n_missing, 0, f"Filled {n_missing} values with mode '{modes.iloc[0]}'."

        if strategy == "Fill with custom text value":
            if custom_value is None or str(custom_value).strip() == "":
                return result, 0, n_missing, "No custom value provided."
            result = result.fillna(str(custom_value))
            return result, n_missing, 0, f"Filled {n_missing} values with '{custom_value}'."

        # --- Group-wise strategies ---
        if strategy.startswith("Group-wise") and group_by_series is not None:
            method = "median" if "median" in strategy.lower() else "mean"
            if not is_numeric_column(series):
                return result, 0, n_missing, "Group-wise imputation requires a numeric column."
            temp_df = pd.DataFrame({"val": series, "grp": group_by_series})
            if method == "mean":
                group_fill = temp_df.groupby("grp")["val"].transform("mean")
            else:
                group_fill = temp_df.groupby("grp")["val"].transform("median")
            result = series.fillna(group_fill)
            filled = n_missing - int(result.isna().sum())
            failed = n_missing - filled
            msg = f"Group-wise {method} filled {filled} values."
            if failed > 0:
                msg += f" {failed} values could not be filled (group had no valid values)."
            return result, filled, failed, msg

        return result, 0, n_missing, f"Unknown strategy: '{strategy}'."


# ---------------------------------------------------------------------------
#  3. FUZZY TEXT MATCHING ENGINE
# ---------------------------------------------------------------------------

def _similarity_score(a: str, b: str) -> float:
    """Compute similarity between two strings as 0–100 percentage.

    Uses RapidFuzz token_sort_ratio with default_process (lowercase + strip)
    for case-insensitive, whitespace-normalizing comparison.
    Falls back to difflib.SequenceMatcher with explicit lowercasing.
    Token-aware scoring handles multi-word strings.
    """
    if FUZZY_BACKEND == "rapidfuzz":
        from rapidfuzz import utils as _rf_utils
        return float(_rf_fuzz.token_sort_ratio(a, b, processor=_rf_utils.default_process))
    else:
        # difflib SequenceMatcher — token sort + lowercase
        a_tokens = " ".join(sorted(a.lower().split()))
        b_tokens = " ".join(sorted(b.lower().split()))
        return round(difflib.SequenceMatcher(None, a_tokens, b_tokens).ratio() * 100, 2)


def _char_similarity(a: str, b: str) -> float:
    """Character-level similarity 0–100 (case-insensitive)."""
    if FUZZY_BACKEND == "rapidfuzz":
        from rapidfuzz import utils as _rf_utils
        return float(_rf_fuzz.ratio(a, b, processor=_rf_utils.default_process))
    else:
        return round(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100, 2)


class FuzzyMatcher:
    """Fuzzy text matching engine for detecting inconsistent categories.

    Uses RapidFuzz (token_sort_ratio) with difflib fallback.
    Skips identifier-like columns.
    """

    def __init__(self, threshold: float = 80.0, max_unique: int = 500):
        self.threshold = threshold
        self.max_unique = max_unique

    def find_groups(
        self, series: pd.Series, col_name: str = ""
    ) -> list[dict[str, Any]]:
        """Find groups of similar values in a text column.

        Returns list of group dicts with:
          canonical, variants: [{value, count, similarity}], total_affected
        """
        if is_identifier_like(series, col_name):
            return []

        # Get value counts for non-null values
        nonempty = series.dropna()
        if is_text_column(series):
            nonempty = nonempty[nonempty.astype(str).str.strip().ne("")]
        if len(nonempty) == 0:
            return []

        counts = nonempty.astype(str).value_counts()
        unique_vals = list(counts.index)

        if len(unique_vals) > self.max_unique:
            # Too many unique values — skip to avoid O(n²) explosion
            return []
        if len(unique_vals) < 2:
            return []

        # Build normalized lookup for comparison
        norm_map: dict[str, str] = {}  # normalized -> first original
        for v in unique_vals:
            norm_map[v.strip().lower()] = v

        # Build groups via clustering
        used: set[str] = set()
        groups: list[dict[str, Any]] = []

        for i, val_a in enumerate(unique_vals):
            if val_a in used:
                continue
            norm_a = val_a.strip().lower()
            cluster: list[dict[str, Any]] = []

            for j in range(i + 1, len(unique_vals)):
                val_b = unique_vals[j]
                if val_b in used:
                    continue
                score = _similarity_score(val_a, val_b)
                if score >= self.threshold:
                    cluster.append({
                        "value": val_b,
                        "count": int(counts[val_b]),
                        "similarity": round(score, 1),
                    })
                    used.add(val_b)

            if cluster:
                used.add(val_a)
                # Recommend most frequent as canonical
                all_in_group = [{"value": val_a, "count": int(counts[val_a]), "similarity": 100.0}] + cluster
                canonical = max(all_in_group, key=lambda x: x["count"])
                variants = [v for v in all_in_group if v["value"] != canonical["value"]]
                total_affected = sum(v["count"] for v in variants)

                groups.append({
                    "canonical": canonical["value"],
                    "canonical_count": canonical["count"],
                    "variants": variants,
                    "total_affected_rows": total_affected,
                })

        return groups

    def apply_replacements(
        self, series: pd.Series, replacements: dict[str, str]
    ) -> tuple[pd.Series, int]:
        """Apply value replacements. Returns (new_series, count_changed)."""
        result = series.copy()
        changed = 0
        for old_val, new_val in replacements.items():
            mask = result.astype(str) == old_val
            count = int(mask.sum())
            if count > 0:
                result[mask] = new_val
                changed += count
        return result, changed


# ---------------------------------------------------------------------------
#  4. REGEX VALIDATION ENGINE
# ---------------------------------------------------------------------------

# Predefined patterns with their assumptions
PREDEFINED_PATTERNS: dict[str, dict[str, str]] = {
    "Email (basic)": {
        "pattern": r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
        "description": "Basic email format check (local@domain.tld). Does not verify the address exists.",
    },
    "Phone (digits, 7–15 chars)": {
        "pattern": r"^[\+]?[\d\s\-\(\)]{7,15}$",
        "description": "Digits with optional +, spaces, dashes, parens. 7–15 characters. Does not validate country-specific formats.",
    },
    "US ZIP Code (5 or 5+4)": {
        "pattern": r"^\d{5}(-\d{4})?$",
        "description": "US ZIP code: 5 digits, optionally followed by dash and 4 more digits.",
    },
    "Indian PIN Code (6 digits)": {
        "pattern": r"^[1-9]\d{5}$",
        "description": "Indian PIN code: 6 digits, first digit not zero.",
    },
    "Date (YYYY-MM-DD)": {
        "pattern": r"^\d{4}-\d{2}-\d{2}$",
        "description": "ISO date format YYYY-MM-DD. Does not validate calendar correctness.",
    },
    "Date (DD/MM/YYYY)": {
        "pattern": r"^\d{2}/\d{2}/\d{4}$",
        "description": "Day/Month/Year format with slashes. Does not validate calendar correctness.",
    },
    "Alphanumeric ID": {
        "pattern": r"^[A-Za-z0-9\-_]+$",
        "description": "Letters, digits, dashes, underscores only.",
    },
    "Positive Integer": {
        "pattern": r"^[1-9]\d*$",
        "description": "Positive whole number (no leading zeros).",
    },
    "URL (http/https)": {
        "pattern": r"^https?://[^\s]+$",
        "description": "Basic URL starting with http:// or https://. Does not verify reachability.",
    },
}


class RegexValidator:
    """Pattern-based validation for column values."""

    def __init__(self, pattern: str, pattern_name: str = "Custom"):
        self._raw_pattern = pattern
        self.pattern_name = pattern_name
        self._compiled: re.Pattern | None = None
        self._error: str | None = None
        try:
            self._compiled = re.compile(pattern)
        except re.error as e:
            self._error = str(e)

    @property
    def is_valid_regex(self) -> bool:
        return self._compiled is not None

    @property
    def error_message(self) -> str | None:
        return self._error

    def validate(self, series: pd.Series) -> dict[str, Any]:
        """Validate series values against the pattern.

        Returns dict with valid_count, invalid_count, missing_count,
        invalid_values (sample), valid_values (sample).
        """
        if not self.is_valid_regex:
            return {
                "error": self._error,
                "valid_count": 0,
                "invalid_count": 0,
                "missing_count": 0,
                "invalid_examples": [],
                "valid_examples": [],
            }

        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())
        non_null = series[~missing_mask].astype(str)

        # Also count whitespace-only as missing if text
        if is_text_column(series):
            blank_mask = non_null.str.strip().eq("")
            missing_count += int(blank_mask.sum())
            non_null = non_null[~blank_mask]

        matches = non_null.apply(lambda v: bool(self._compiled.match(str(v).strip())))
        valid_count = int(matches.sum())
        invalid_count = len(non_null) - valid_count

        invalid_vals = non_null[~matches].head(20).tolist()
        valid_vals = non_null[matches].head(10).tolist()

        return {
            "pattern": self._raw_pattern,
            "pattern_name": self.pattern_name,
            "total_rows": len(series),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "missing_count": missing_count,
            "valid_pct": round(valid_count / len(non_null) * 100, 1) if len(non_null) else 0.0,
            "invalid_pct": round(invalid_count / len(non_null) * 100, 1) if len(non_null) else 0.0,
            "invalid_examples": invalid_vals,
            "valid_examples": valid_vals,
            "error": None,
        }


# ---------------------------------------------------------------------------
#  5. REGEX CLEANING OPERATIONS
# ---------------------------------------------------------------------------

class RegexCleaner:
    """Regex-based cleaning: find, extract, replace, remove, validate-only."""

    @staticmethod
    def compile_safe(pattern: str) -> tuple[re.Pattern | None, str | None]:
        """Compile a regex, returning (compiled, error_message)."""
        try:
            return re.compile(pattern), None
        except re.error as e:
            return None, f"Invalid regex: {e}"

    @staticmethod
    def find_matches(series: pd.Series, pattern: str) -> tuple[pd.DataFrame, str | None]:
        """Find values matching pattern. Returns (results_df, error)."""
        compiled, err = RegexCleaner.compile_safe(pattern)
        if err:
            return pd.DataFrame(), err
        non_null = series.dropna()
        matches_mask = non_null.astype(str).apply(lambda v: bool(compiled.search(v)))
        matched = non_null[matches_mask]
        df = pd.DataFrame({"row": matched.index, "value": matched.values})
        return df, None

    @staticmethod
    def preview_replace(
        series: pd.Series, pattern: str, replacement: str, max_preview: int = 30
    ) -> tuple[pd.DataFrame, str | None]:
        """Preview regex replacement without modifying data."""
        compiled, err = RegexCleaner.compile_safe(pattern)
        if err:
            return pd.DataFrame(), err
        records: list[dict[str, Any]] = []
        count = 0
        for idx, val in series.items():
            if pd.isna(val):
                continue
            s = str(val)
            new_s = compiled.sub(replacement, s)
            if new_s != s:
                records.append({"row": idx, "original": s, "result": new_s})
                count += 1
                if count >= max_preview:
                    break
        return pd.DataFrame(records), None

    @staticmethod
    def apply_replace(
        series: pd.Series, pattern: str, replacement: str
    ) -> tuple[pd.Series, int, str | None]:
        """Apply regex replacement. Returns (new_series, changed_count, error)."""
        compiled, err = RegexCleaner.compile_safe(pattern)
        if err:
            return series, 0, err
        result = series.copy()
        changed = 0
        for idx, val in series.items():
            if pd.isna(val):
                continue
            s = str(val)
            new_s = compiled.sub(replacement, s)
            if new_s != s:
                result.at[idx] = new_s
                changed += 1
        return result, changed, None

    @staticmethod
    def extract_pattern(
        series: pd.Series, pattern: str
    ) -> tuple[pd.DataFrame, str | None]:
        """Extract first regex match from each value."""
        compiled, err = RegexCleaner.compile_safe(pattern)
        if err:
            return pd.DataFrame(), err
        records: list[dict[str, Any]] = []
        for idx, val in series.items():
            if pd.isna(val):
                continue
            m = compiled.search(str(val))
            if m:
                records.append({
                    "row": idx, "original": str(val),
                    "extracted": m.group(0),
                    "groups": m.groups() if m.groups() else None,
                })
        return pd.DataFrame(records), None


# ---------------------------------------------------------------------------
#  6. DATA QUALITY RULES & TEST ENGINE
# ---------------------------------------------------------------------------

class DataQualityTestRunner:
    """Configurable data quality test engine."""

    @staticmethod
    def test_missing_required(
        series: pd.Series, col_name: str
    ) -> dict[str, Any]:
        """Test for missing required values."""
        mask = get_missing_mask(series, include_placeholders=True)
        n_missing = int(mask.sum())
        n_total = len(series)
        n_pass = n_total - n_missing
        examples = series[mask].head(5).tolist()
        return {
            "test": "Missing required values",
            "column": col_name,
            "checked": n_total,
            "passed": n_pass,
            "failed": n_missing,
            "skipped": 0,
            "failure_pct": round(n_missing / n_total * 100, 2) if n_total else 0.0,
            "examples": [repr(e) for e in examples],
            "suggestion": "Review and fill or remove missing values." if n_missing else "All values present.",
        }

    @staticmethod
    def test_invalid_numeric(
        series: pd.Series, col_name: str
    ) -> dict[str, Any]:
        """Test for non-numeric values in a column that should be numeric."""
        if is_numeric_column(series):
            n_nan = int(series.isna().sum())
            return {
                "test": "Invalid numeric values",
                "column": col_name,
                "checked": len(series),
                "passed": len(series) - n_nan,
                "failed": n_nan,
                "skipped": 0,
                "failure_pct": round(n_nan / len(series) * 100, 2) if len(series) else 0.0,
                "examples": [],
                "suggestion": "Column is numeric; NaN values may need imputation." if n_nan else "All values valid.",
            }
        # Text column — try parsing
        non_null = series.dropna()
        non_null = non_null[non_null.astype(str).str.strip().ne("")]
        n_missing = len(series) - len(non_null)
        failed = 0
        fail_examples: list[str] = []
        for val in non_null:
            success, _ = clean_and_parse_numeric_value(val)
            if not success:
                failed += 1
                if len(fail_examples) < 5:
                    fail_examples.append(str(val))
        return {
            "test": "Invalid numeric values",
            "column": col_name,
            "checked": len(non_null),
            "passed": len(non_null) - failed,
            "failed": failed,
            "skipped": n_missing,
            "failure_pct": round(failed / len(non_null) * 100, 2) if len(non_null) else 0.0,
            "examples": fail_examples,
            "suggestion": "Values that could not be parsed as numbers." if failed else "All non-missing values are numeric.",
        }

    @staticmethod
    def test_date_parsing(
        series: pd.Series, col_name: str, convention: str = "auto"
    ) -> dict[str, Any]:
        """Test date parsing."""
        if pd.api.types.is_datetime64_any_dtype(series):
            n_nat = int(series.isna().sum())
            return {
                "test": "Failed date parsing",
                "column": col_name,
                "checked": len(series),
                "passed": len(series) - n_nat,
                "failed": n_nat,
                "skipped": 0,
                "failure_pct": round(n_nat / len(series) * 100, 2) if len(series) else 0.0,
                "examples": [],
                "suggestion": "Column is datetime; NaT values may indicate parsing failures." if n_nat else "All dates valid.",
            }
        _, success, failed, _ = parse_date_series(series, convention=convention)
        n_missing = int(get_missing_mask(series).sum())
        return {
            "test": "Failed date parsing",
            "column": col_name,
            "checked": len(series) - n_missing,
            "passed": success,
            "failed": failed,
            "skipped": n_missing,
            "failure_pct": round(failed / (len(series) - n_missing) * 100, 2) if (len(series) - n_missing) else 0.0,
            "examples": [],
            "suggestion": f"{failed} values could not be parsed as dates." if failed else "All non-missing values parsed.",
        }

    @staticmethod
    def test_duplicates(df: pd.DataFrame) -> dict[str, Any]:
        """Test for exact duplicate rows."""
        n_dup = detect_exact_duplicates(df)
        return {
            "test": "Duplicate rows",
            "column": "(all columns)",
            "checked": len(df),
            "passed": len(df) - n_dup,
            "failed": n_dup,
            "skipped": 0,
            "failure_pct": round(n_dup / len(df) * 100, 2) if len(df) else 0.0,
            "examples": [],
            "suggestion": f"{n_dup} exact duplicate rows found." if n_dup else "No duplicates.",
        }

    @staticmethod
    def test_email_format(
        series: pd.Series, col_name: str
    ) -> dict[str, Any]:
        """Test email format validation."""
        validator = RegexValidator(
            PREDEFINED_PATTERNS["Email (basic)"]["pattern"], "Email"
        )
        result = validator.validate(series)
        return {
            "test": "Invalid email format",
            "column": col_name,
            "checked": result["valid_count"] + result["invalid_count"],
            "passed": result["valid_count"],
            "failed": result["invalid_count"],
            "skipped": result["missing_count"],
            "failure_pct": result["invalid_pct"],
            "examples": result["invalid_examples"][:5],
            "suggestion": "Format check only — does not verify email deliverability." if result["invalid_count"] else "All non-missing values match email format.",
        }

    @staticmethod
    def test_regex_pattern(
        series: pd.Series, col_name: str, pattern: str, test_name: str = "Regex pattern"
    ) -> dict[str, Any]:
        """Test a column against a custom regex pattern."""
        validator = RegexValidator(pattern, test_name)
        if not validator.is_valid_regex:
            return {
                "test": test_name,
                "column": col_name,
                "checked": 0, "passed": 0, "failed": 0, "skipped": len(series),
                "failure_pct": 0.0,
                "examples": [],
                "suggestion": f"Invalid regex: {validator.error_message}",
            }
        result = validator.validate(series)
        return {
            "test": test_name,
            "column": col_name,
            "checked": result["valid_count"] + result["invalid_count"],
            "passed": result["valid_count"],
            "failed": result["invalid_count"],
            "skipped": result["missing_count"],
            "failure_pct": result["invalid_pct"],
            "examples": result["invalid_examples"][:5],
            "suggestion": f"{result['invalid_count']} values failed the pattern." if result["invalid_count"] else "All values match.",
        }

    @staticmethod
    def test_inconsistent_categories(
        series: pd.Series, col_name: str, threshold: float = 85.0
    ) -> dict[str, Any]:
        """Test for inconsistent text categories using fuzzy matching."""
        matcher = FuzzyMatcher(threshold=threshold)
        groups = matcher.find_groups(series, col_name)
        n_affected = sum(g["total_affected_rows"] for g in groups)
        examples: list[str] = []
        for g in groups[:3]:
            variants_str = ", ".join(
                f"'{v['value']}' ({v['similarity']:.0f}%)"
                for v in g["variants"][:3]
            )
            examples.append(f"'{g['canonical']}' ↔ {variants_str}")
        return {
            "test": "Inconsistent text categories",
            "column": col_name,
            "checked": int(series.notna().sum()),
            "passed": int(series.notna().sum()) - n_affected,
            "failed": n_affected,
            "skipped": int(series.isna().sum()),
            "failure_pct": round(n_affected / max(1, int(series.notna().sum())) * 100, 2),
            "examples": examples,
            "suggestion": f"{len(groups)} groups of similar values found." if groups else "No inconsistent categories detected.",
        }

    @staticmethod
    def test_numeric_range(
        series: pd.Series, col_name: str,
        min_val: float | None = None, max_val: float | None = None,
    ) -> dict[str, Any]:
        """Test values are within a specified numeric range."""
        if not is_numeric_column(series):
            return {
                "test": f"Numeric range ({min_val}–{max_val})",
                "column": col_name,
                "checked": 0, "passed": 0, "failed": 0,
                "skipped": len(series),
                "failure_pct": 0.0,
                "examples": [],
                "suggestion": "Column is not numeric.",
            }
        non_null = series.dropna()
        n_missing = int(series.isna().sum())
        violations = pd.Series(False, index=non_null.index)
        if min_val is not None:
            violations = violations | (non_null < min_val)
        if max_val is not None:
            violations = violations | (non_null > max_val)
        n_fail = int(violations.sum())
        examples = non_null[violations].head(5).tolist()
        return {
            "test": f"Numeric range [{min_val}, {max_val}]",
            "column": col_name,
            "checked": len(non_null),
            "passed": len(non_null) - n_fail,
            "failed": n_fail,
            "skipped": n_missing,
            "failure_pct": round(n_fail / len(non_null) * 100, 2) if len(non_null) else 0.0,
            "examples": [str(e) for e in examples],
            "suggestion": f"{n_fail} values outside range." if n_fail else "All values in range.",
        }

    @staticmethod
    def test_no_negatives(
        series: pd.Series, col_name: str
    ) -> dict[str, Any]:
        """Test that no negative values exist."""
        return DataQualityTestRunner.test_numeric_range(
            series, col_name, min_val=0, max_val=None
        )


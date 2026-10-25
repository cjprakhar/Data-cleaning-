from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quality import (
    AMBIGUOUS_PLACEHOLDERS,
    STANDARD_MISSING_PLACEHOLDERS,
    FUZZY_BACKEND,
    PREDEFINED_PATTERNS,
    calculate_correlation_matrix,
    clean_and_parse_numeric_value,
    compare_datasets,
    compute_quality_issue_categories,
    detect_exact_duplicates,
    detect_iqr_outliers,
    detect_near_duplicates,
    find_ambiguous_placeholders,
    find_date_text_candidates,
    find_numeric_text_candidates,
    get_missing_mask,
    is_identifier_like,
    is_numeric_column,
    is_text_column,
    numeric_conversion_with_stats,
    parse_date_series,
    profile_dataframe,
    sample_dataframe_for_missingness,
    summarize_dataframe,
    MissingDataEngine,
    MissingValueFallback,
    FuzzyMatcher,
    RegexValidator,
    RegexCleaner,
    DataQualityTestRunner,
    _similarity_score,
    _char_similarity,
)

st.set_page_config(page_title="CleanSight | Data Quality Studio", page_icon="🧹", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1500px;}
      .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px; margin-bottom: 1rem;}
      .hero h1 {margin: 0 0 .35rem 0; letter-spacing: -.035em;}
      .muted {opacity: .75;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .75rem; border-radius: 12px;}
      div[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧹 CleanSight</h1>
      <div class="muted">Data Cleaning & Quality Studio · Profile, clean, validate and export spreadsheet data.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def apply_chart_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply consistent, sleek dark theme styling to all Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=25),
        font=dict(family="sans-serif", color="#E6EDF3", size=12),
        height=height,
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.08)",
        zerolinecolor="rgba(255, 255, 255, 0.15)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.08)",
        zerolinecolor="rgba(255, 255, 255, 0.15)",
    )
    return fig


def get_excel_engine(filename: str) -> str:
    return "xlrd" if filename.lower().endswith(".xls") else "openpyxl"


def get_excel_sheets(raw: bytes, filename: str) -> list[str]:
    engine = get_excel_engine(filename)
    workbook = pd.ExcelFile(io.BytesIO(raw), engine=engine)
    return workbook.sheet_names or ["Sheet1"]


def read_uploaded_data(raw: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    if not raw:
        raise ValueError("The uploaded file is empty.")
    lower = filename.lower()
    if lower.endswith(".csv"):
        errors: list[str] = []
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                frame = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=None, engine="python")
                if frame.shape[1] == 1 and b"\t" in raw and b"," not in raw:
                    frame = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep="\t")
                return frame
            except Exception as exc:
                errors.append(f"{encoding}: {exc}")
        raise ValueError("Could not read this CSV. Check its delimiter and encoding. " + " | ".join(errors[-2:]))
    if lower.endswith((".xlsx", ".xls")):
        engine = get_excel_engine(filename)
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name or 0, engine=engine)
    raise ValueError("Unsupported file type. Upload .xlsx, .xls or .csv.")


def text_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if is_text_column(df[c])]


def log_entry(operation: str, affected: int, details: str, before_rows: int, after_rows: int, **extra: Any) -> dict[str, Any]:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "affected_cells_or_rows": int(affected),
        "rows_before": int(before_rows),
        "rows_after": int(after_rows),
        "details": details,
        "status": extra.get("status", "Applied"),
    }
    for key in ("algorithm", "threshold", "canonical_value", "regex_pattern",
                "column", "values_examined", "values_changed", "values_failed"):
        if key in extra:
            entry[key] = extra[key]
    return entry



def normalize_headers(columns: list[Any]) -> list[str]:
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
    output = io.BytesIO()
    audit_df = pd.DataFrame(audit) if audit else pd.DataFrame([{"Status": "No cleaning operations logged"}])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        profile.to_excel(writer, index=False, sheet_name="Data Quality Profile")
        audit_df.to_excel(writer, index=False, sheet_name="Cleaning Log")
    output.seek(0)
    return output.getvalue()


def show_empty_state() -> None:
    st.info("Upload an Excel or CSV file to begin. Your data will be analyzed in this local app session.")
    st.markdown("**Supported inputs:** `.xlsx`, `.xls`, `.csv`  ·  **Outputs:** cleaned Excel/CSV, profile and audit report")
    st.markdown("### What CleanSight checks")
    st.write("Missing values · exact duplicate rows · candidate near-duplicates · blank strings · inferred data types · numbers stored as text · date-like text · numeric IQR outliers")


uploaded_file = st.file_uploader(
    "Upload an Excel or CSV file",
    type=["xlsx", "xls", "csv"],
    help="Excel workbooks can contain multiple worksheets. Upload files you have permission to process.",
)

if uploaded_file is not None:
    raw_bytes = uploaded_file.getvalue()
    file_fingerprint = hashlib.md5(raw_bytes).hexdigest()
    selected_sheet: str | None = None
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        try:
            sheets = get_excel_sheets(raw_bytes, uploaded_file.name)
        except Exception as exc:
            st.error(f"Could not inspect this workbook: {exc}")
            st.stop()
        selected_sheet = st.selectbox("Excel worksheet", sheets, key=f"sheet_{file_fingerprint[:10]}")
    current_source_key = f"{file_fingerprint}|{selected_sheet or ''}"
    if st.session_state.get("loaded_source_key") != current_source_key:
        try:
            loaded = read_uploaded_data(raw_bytes, uploaded_file.name, selected_sheet)
            loaded.columns = [str(col) if str(col).strip() else f"Unnamed_{i + 1}" for i, col in enumerate(loaded.columns)]
            st.session_state.original_df = loaded.copy(deep=True)
            st.session_state.working_df = loaded.copy(deep=True)
            st.session_state.audit_log = []
            st.session_state.loaded_source_key = current_source_key
            st.session_state.source_filename = uploaded_file.name
            st.session_state.source_sheet = selected_sheet or ""
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            st.stop()

if "working_df" not in st.session_state or "original_df" not in st.session_state:
    show_empty_state()
    st.stop()
else:
    original_df: pd.DataFrame = st.session_state.original_df
    working_df: pd.DataFrame = st.session_state.working_df
    source_filename: str = st.session_state.get("source_filename", "uploaded_file")
    audit_log: list[dict[str, Any]] = st.session_state.get("audit_log", [])

# Compute robust profile and summary from the active working dataset
profile = profile_dataframe(working_df)
summary = summarize_dataframe(working_df)
original_summary = summarize_dataframe(original_df)

st.caption(
    f"Source: **{source_filename}**"
    + (f" · Worksheet: **{st.session_state.get('source_sheet')}**" if st.session_state.get("source_sheet") else "")
    + f" · Current working dataset: {summary['rows']:,} rows × {summary['columns']:,} columns"
)

overview_tab, preview_tab, profile_tab, clean_tab, quality_tab, export_tab = st.tabs(
    ["Overview", "Data Preview", "Data Profile", "Cleaning Studio", "Quality Rules", "Export & Log"]
)

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with overview_tab:
    st.subheader("Current data quality")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Rows",
        f"{summary['rows']:,}",
        delta=f"{summary['rows'] - original_summary['rows']:+,} vs original" if summary["rows"] != original_summary["rows"] else None,
    )
    m2.metric("Columns", f"{summary['columns']:,}")
    m3.metric(
        "Missing cells",
        f"{summary['missing_cells']:,}",
        delta=f"{summary['missing_cells'] - original_summary['missing_cells']:+,} vs original" if summary["missing_cells"] != original_summary["missing_cells"] else None,
        delta_color="inverse",
    )
    m4.metric(
        "Duplicate rows",
        f"{summary['exact_duplicate_rows']:,}",
        delta=f"{summary['exact_duplicate_rows'] - original_summary['exact_duplicate_rows']:+,} vs original" if summary["exact_duplicate_rows"] != original_summary["exact_duplicate_rows"] else None,
        delta_color="inverse",
    )
    m5.metric("Completeness", f"{summary['completeness_percent']:.1f}%")
    st.caption("Completeness = non-missing cells ÷ all cells. It is a transparent completeness measure, not a universal data-quality score.")

    # Overview Visualizations Row 1
    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.markdown("#### Missing cells by column")
        missing_data = profile.loc[profile["Missing"] > 0, ["Column", "Missing", "Missing %"]].sort_values("Missing %", ascending=True).tail(20)
        if missing_data.empty:
            st.success("No missing values detected in the current dataset.")
        else:
            fig = px.bar(
                missing_data,
                x="Missing",
                y="Column",
                orientation="h",
                title=f"Top {len(missing_data)} columns by missing values",
                labels={"Missing": "Missing Cells", "Column": "Column Name"},
                text=missing_data.apply(lambda r: f"{int(r['Missing']):,} ({r['Missing %']:.1f}%)", axis=1),
                color="Missing %",
                color_continuous_scale="Reds",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False)
            apply_chart_theme(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

    with right_chart:
        st.markdown("#### Column data completeness")
        if profile.empty:
            st.info("No columns available to evaluate completeness.")
        else:
            # Sort ascending so the least complete columns appear at the top
            comp_data = profile[["Column", "Completeness %"]].sort_values("Completeness %", ascending=True).head(20)
            fig_comp = px.bar(
                comp_data,
                x="Completeness %",
                y="Column",
                orientation="h",
                title="Column completeness % (least complete first)",
                labels={"Completeness %": "Completeness (%)", "Column": "Column Name"},
                text=comp_data["Completeness %"].map(lambda p: f"{p:.1f}%"),
                range_x=[0, 105],
                color="Completeness %",
                color_continuous_scale="Tealgrn",
            )
            fig_comp.update_traces(textposition="outside", cliponaxis=False)
            fig_comp.update_layout(coloraxis_showscale=False)
            apply_chart_theme(fig_comp, height=420)
            st.plotly_chart(fig_comp, use_container_width=True)

    # Overview Visualizations Row 2
    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.markdown("#### Data quality issue distribution")
        issue_df = compute_quality_issue_categories(working_df, profile, summary)
        if issue_df.empty:
            st.success("No major quality issues detected in the current working dataset.")
        else:
            fig_issues = px.bar(
                issue_df,
                x="Count",
                y="Issue Category",
                orientation="h",
                title="Detected data quality findings by category",
                labels={"Count": "Quantity", "Issue Category": "Issue Type"},
                text=issue_df.apply(lambda r: f"{r['Count']:,} {r['Unit']}", axis=1),
                color="Issue Category",
                color_discrete_sequence=["#F59E0B", "#EF4444", "#EC4899", "#8B5CF6", "#3B82F6", "#06B6D4"],
            )
            fig_issues.update_traces(textposition="outside", cliponaxis=False)
            fig_issues.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            apply_chart_theme(fig_issues, height=360)
            st.plotly_chart(fig_issues, use_container_width=True)

    with row2_right:
        st.markdown("#### Column data types")
        dtype_counts = working_df.dtypes.astype(str).value_counts().rename_axis("Data type").reset_index(name="Columns")
        if not dtype_counts.empty:
            fig_dt = px.bar(
                dtype_counts,
                x="Data type",
                y="Columns",
                title="Columns by detected type",
                text="Columns",
                color="Data type",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_dt.update_traces(textposition="outside")
            fig_dt.update_layout(showlegend=False)
            apply_chart_theme(fig_dt, height=360)
            st.plotly_chart(fig_dt, use_container_width=True)

    # Findings Expander with Transparent Explanations
    warnings: list[str] = []
    duplicates = summary["exact_duplicate_rows"]
    if duplicates:
        warnings.append(f"{duplicates:,} exact duplicate rows detected across all columns.")
    near_dups = summary.get("near_duplicate_rows", 0)
    if near_dups:
        warnings.append(f"{near_dups:,} candidate near-duplicate rows detected (differing only by case or whitespace). Flagged for review.")
    blanks = summary.get("blank_text_cells", 0)
    if blanks:
        warnings.append(f"{blanks:,} whitespace-only text cells detected; counted as missing in profile.")
    placeholders = summary.get("placeholder_cells", 0)
    if placeholders:
        warnings.append(f"{placeholders:,} recognized missing-value placeholder cells detected (e.g. 'N/A', 'null', 'none').")
    outliers = summary.get("total_outliers", 0)
    if outliers:
        warnings.append(f"{outliers:,} potential numeric outliers flagged across all numeric fields using 1.5 × IQR.")
    numeric_text = profile.loc[profile["Numbers stored as text?"] == "Yes — review", "Column"].tolist()
    if numeric_text:
        warnings.append("Possible numbers stored as text (currencies/commas/units): " + ", ".join(numeric_text[:12]) + ("…" if len(numeric_text) > 12 else "."))
    date_like = profile.loc[profile["Date-like text?"] == "Yes — review", "Column"].tolist()
    if date_like:
        warnings.append("Date-like text columns: " + ", ".join(date_like[:12]) + ("…" if len(date_like) > 12 else "."))
    protected_ids = profile.loc[profile["Numbers stored as text?"] == "Protected identifier / code", "Column"].tolist()
    if protected_ids:
        warnings.append("Protected identifier/code columns (leading zeros or code pattern preserved): " + ", ".join(protected_ids[:10]))

    with st.expander("Data quality findings & audit notices", expanded=bool(warnings)):
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No common quality issues were detected by the current checks. Always validate data against your business rules.")

# ==============================================================================
# TAB 2: DATA PREVIEW
# ==============================================================================
with preview_tab:
    st.subheader("Inspect the working dataset")
    c1, c2 = st.columns([1, 2])
    with c1:
        preview_rows = st.select_slider("Rows to display", options=[10, 25, 50, 100, 250, 500], value=50)
    with c2:
        st.caption(f"Showing up to {min(preview_rows, len(working_df)):,} of {len(working_df):,} rows")
    st.dataframe(working_df.head(preview_rows), use_container_width=True, height=440)
    st.markdown("#### Column summary")
    st.dataframe(
        profile[["Column", "Data type", "Unique values", "Missing", "Missing %", "Completeness %"]],
        use_container_width=True,
        hide_index=True,
    )

# ==============================================================================
# TAB 3: DATA PROFILE
# ==============================================================================
with profile_tab:
    st.subheader("Column-level profiling")
    st.caption("Outliers are flagged using the 1.5 × IQR rule for review only; this tool never removes outliers automatically.")
    st.dataframe(profile, use_container_width=True, hide_index=True, height=380)

    if len(working_df.columns):
        st.divider()
        st.markdown("### Interactive Column Explorer")
        selected_profile_column = st.selectbox("Explore a column", list(working_df.columns), key="profile_column")
        series = working_df[selected_profile_column]

        if is_numeric_column(series):
            chart_col1, chart_col2 = st.columns(2)
            clean_series = series.dropna()
            with chart_col1:
                hist = px.histogram(
                    working_df,
                    x=selected_profile_column,
                    title=f"Distribution: {selected_profile_column}",
                    nbins=30,
                    marginal="box",
                    color_discrete_sequence=["#6366F1"],
                )
                if len(clean_series):
                    mean_val = float(clean_series.mean())
                    median_val = float(clean_series.median())
                    hist.add_vline(x=mean_val, line_dash="dash", line_color="#F59E0B", annotation_text=f"Mean: {mean_val:.2f}")
                    hist.add_vline(x=median_val, line_dash="dot", line_color="#10B981", annotation_text=f"Median: {median_val:.2f}")
                apply_chart_theme(hist, height=400)
                st.plotly_chart(hist, use_container_width=True)

            with chart_col2:
                box = px.box(
                    working_df,
                    y=selected_profile_column,
                    points="outliers",
                    title=f"Spread and potential outliers: {selected_profile_column}",
                    color_discrete_sequence=["#06B6D4"],
                )
                apply_chart_theme(box, height=400)
                st.plotly_chart(box, use_container_width=True)

            st.markdown("#### Summary statistics")
            stat_df = series.describe().to_frame("Value")
            out_cnt, q1, q3, lower_b, upper_b = detect_iqr_outliers(series)
            if q1 is not None and q3 is not None:
                stat_df.loc["IQR"] = q3 - q1
                stat_df.loc["IQR Lower Bound"] = lower_b
                stat_df.loc["IQR Upper Bound"] = upper_b
                stat_df.loc["Outlier Count"] = out_cnt
            st.dataframe(stat_df.T, use_container_width=True)

        elif pd.api.types.is_datetime64_any_dtype(series):
            date_values = series.dropna()
            if len(date_values):
                counts = (
                    date_values.dt.to_period("M")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .rename_axis("Month")
                    .reset_index(name="Records")
                )
                fig_date = px.line(
                    counts,
                    x="Month",
                    y="Records",
                    markers=True,
                    title=f"Records over time: {selected_profile_column}",
                    line_shape="linear",
                )
                fig_date.update_traces(line_color="#06B6D4", marker=dict(size=7, color="#3B82F6"))
                apply_chart_theme(fig_date, height=380)
                st.plotly_chart(fig_date, use_container_width=True)
            else:
                st.info("No valid dates in this column to plot.")

        else:
            # Categorical / Text Column Exploration
            cat_ctrl1, cat_ctrl2 = st.columns([1, 1])
            with cat_ctrl1:
                top_n = st.slider("Top N categories to display", min_value=5, max_value=50, value=10, key="cat_top_n")
            with cat_ctrl2:
                group_other = st.checkbox("Group remaining categories as 'Other'", value=False, key="cat_group_other")

            # Value counts calculation (without mutating underlying dataset)
            raw_counts = series.astype("string").fillna("(missing)").value_counts()
            total_cat_records = len(series)

            if group_other and len(raw_counts) > top_n:
                top_slice = raw_counts.head(top_n)
                other_count = int(raw_counts.iloc[top_n:].sum())
                top_slice["Other"] = other_count
                plot_counts = top_slice.rename_axis("Category").reset_index(name="Count")
            else:
                plot_counts = raw_counts.head(top_n).rename_axis("Category").reset_index(name="Count")

            plot_counts["Percentage"] = (plot_counts["Count"] / total_cat_records * 100).round(2)

            fig_cat = px.bar(
                plot_counts,
                x="Count",
                y="Category",
                orientation="h",
                title=f"Most frequent categories: {selected_profile_column} (Top {len(plot_counts)})",
                labels={"Count": "Frequency", "Category": "Category Value"},
                text=plot_counts.apply(lambda r: f"{int(r['Count']):,} ({r['Percentage']:.1f}%)", axis=1),
                color="Count",
                color_continuous_scale="Purples",
            )
            fig_cat.update_traces(textposition="outside", cliponaxis=False)
            fig_cat.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            apply_chart_theme(fig_cat, height=max(360, len(plot_counts) * 26))
            st.plotly_chart(fig_cat, use_container_width=True)

            m_u1, m_u2, m_u3 = st.columns(3)
            m_u1.metric("Unique categories", f"{series.nunique(dropna=True):,}")
            m_u2.metric("Missing values", f"{int(get_missing_mask(series).sum()):,}")
            top_val_str = str(raw_counts.index[0]) if not raw_counts.empty else "N/A"
            m_u3.metric("Most frequent value", top_val_str[:25] + ("…" if len(top_val_str) > 25 else ""))

    # Advanced Profiling Sections
    st.divider()
    corr_col, miss_col = st.columns(2)

    with corr_col:
        st.markdown("#### Numeric Correlation Heatmap")
        corr_matrix, eligible_num_cols = calculate_correlation_matrix(working_df)
        if corr_matrix is not None and len(eligible_num_cols) >= 2:
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                title=f"Pearson correlation matrix ({len(eligible_num_cols)} numeric fields)",
            )
            apply_chart_theme(fig_heatmap, height=420)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.caption("ℹ️ Pearson correlation matrix. Note: Correlation does not establish causation. Constant columns and non-numeric fields are excluded.")
        else:
            st.info("At least 2 numeric columns with variance are required to calculate a correlation heatmap.")

    with miss_col:
        st.markdown("#### Missingness Visualization Matrix")
        miss_cols_available = [c for c in working_df.columns if int(get_missing_mask(working_df[c]).sum()) > 0]
        if not miss_cols_available:
            miss_cols_available = list(working_df.columns)[:8]
        selected_miss_cols = st.multiselect(
            "Columns to display",
            options=list(working_df.columns),
            default=miss_cols_available[:8],
            key="miss_matrix_cols",
        )
        sample_rows_limit = st.slider("Display row limit", min_value=50, max_value=500, value=100, step=25, key="miss_sample_slider")

        if selected_miss_cols:
            binary_matrix, was_sampled = sample_dataframe_for_missingness(
                working_df, selected_cols=selected_miss_cols, max_rows=sample_rows_limit
            )
            if not binary_matrix.empty:
                fig_miss = px.imshow(
                    binary_matrix.T,
                    color_continuous_scale=[[0.0, "#1F2937"], [1.0, "#EF4444"]],
                    title=f"Missingness matrix (0 = Present, 1 = Missing)",
                    aspect="auto",
                )
                fig_miss.update_layout(coloraxis_showscale=False)
                apply_chart_theme(fig_miss, height=360)
                st.plotly_chart(fig_miss, use_container_width=True)
                if was_sampled:
                    st.caption("Display sampled deterministically for interactive performance. Underlying dataset calculations remain complete.")
                else:
                    st.caption("Displaying complete active dataset rows for the selected columns.")
        else:
            st.info("Select at least one column to display the missingness matrix.")

# ==============================================================================
# TAB 4: CLEANING STUDIO
# ==============================================================================
with clean_tab:
    st.subheader("Apply controlled cleaning steps")
    st.write("Choose the operations, apply them to the current working dataset, and review the change log. Your original uploaded data remains available for reset.")
    if working_df.empty or len(working_df.columns) == 0:
        st.warning("The working dataset has no rows or columns. Restore the original data or upload another file.")

    available_columns = [str(c) for c in working_df.columns]
    available_text_columns = text_columns(working_df)
    numeric_candidates = find_numeric_text_candidates(working_df)
    date_candidates = find_date_text_candidates(working_df)

    with st.form("cleaning_form", clear_on_submit=False):
        selected_columns = st.multiselect(
            "Columns for text cleaning",
            options=available_text_columns,
            default=available_text_columns,
            help="Whitespace and case operations apply only to the selected text columns.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            trim_whitespace = st.checkbox("Trim leading/trailing whitespace", value=True)
            blanks_to_missing = st.checkbox("Convert blank/whitespace-only text to missing", value=True)
            placeholders_to_missing = st.checkbox("Convert recognized placeholders ('N/A', 'null', 'none') to missing", value=True)
            standardize_headers = st.checkbox("Standardize column names to snake_case", value=False)
            remove_duplicates = st.checkbox("Remove exact duplicate rows", value=False)
        with col_b:
            text_case = st.selectbox("Text case", ["Keep as-is", "lowercase", "UPPERCASE", "Title Case"], index=0)
            numeric_to_convert = st.multiselect(
                "Convert numbers stored as text",
                options=available_text_columns,
                default=numeric_candidates,
                help="Parses currency symbols, commas, and trailing units (e.g. '5 pcs'). Preserves leading zeros on identifier codes.",
            )
            date_to_convert = st.multiselect(
                "Parse date-like text columns",
                options=available_text_columns,
                default=[],
                help="Parses date strings into timestamps with controlled ambiguity handling.",
            )
            date_convention = st.selectbox(
                "Date format convention",
                options=["Auto-detect (mixed)", "Day first: DD/MM/YYYY", "Month first: MM/DD/YYYY", "ISO format: YYYY-MM-DD"],
                index=0,
                help="Disambiguates dates like 03/04/2025.",
            )

        st.markdown("#### Missing-value strategy")
        missing_strategy = st.selectbox(
            "Choose how to handle missing values",
            [
                "Leave missing values unchanged",
                "Drop rows containing any missing value",
                "Fill numeric missing values with mean",
                "Fill numeric missing values with median",
                "Fill numeric missing values with 0",
                "Fill missing values with each column's mode",
                "Fill all missing values with a custom value",
            ],
        )
        custom_missing_value = st.text_input("Custom fill value (used only for the custom-value option)", value="Unknown")
        confirm_apply = st.form_submit_button("Apply selected cleaning steps", type="primary", use_container_width=True)

    if confirm_apply:
        cleaned = working_df.copy(deep=True)
        new_entries: list[dict[str, Any]] = []
        rows_start = len(cleaned)

        # 1. Text transformations
        for column in selected_columns:
            if column not in cleaned.columns or not is_text_column(cleaned[column]):
                continue
            if trim_whitespace:
                old = cleaned[column].copy()
                cleaned[column] = cleaned[column].map(lambda value: value.strip() if isinstance(value, str) else value)
                changed = int((old.astype("string").fillna("<NA>") != cleaned[column].astype("string").fillna("<NA>")).sum())
                if changed:
                    new_entries.append(log_entry("Trim whitespace", changed, f"Column: {column}", len(cleaned), len(cleaned)))

            if blanks_to_missing:
                blank_mask = cleaned[column].astype("string").str.strip().eq("") & cleaned[column].notna()
                changed = int(blank_mask.sum())
                if changed:
                    cleaned.loc[blank_mask, column] = np.nan
                    new_entries.append(log_entry("Blank text → missing", changed, f"Column: {column}", len(cleaned), len(cleaned)))

            if placeholders_to_missing:
                placeholder_mask = cleaned[column].astype("string").str.strip().str.lower().isin(STANDARD_MISSING_PLACEHOLDERS)
                changed = int(placeholder_mask.sum())
                if changed:
                    cleaned.loc[placeholder_mask, column] = np.nan
                    new_entries.append(log_entry("Placeholders → missing", changed, f"Column: {column}; recognized placeholders converted to NaN", len(cleaned), len(cleaned)))

            if text_case != "Keep as-is":
                old = cleaned[column].copy()
                if text_case == "lowercase":
                    cleaned[column] = cleaned[column].map(lambda value: value.lower() if isinstance(value, str) else value)
                elif text_case == "UPPERCASE":
                    cleaned[column] = cleaned[column].map(lambda value: value.upper() if isinstance(value, str) else value)
                elif text_case == "Title Case":
                    cleaned[column] = cleaned[column].map(lambda value: value.title() if isinstance(value, str) else value)
                changed = int((old.astype("string").fillna("<NA>") != cleaned[column].astype("string").fillna("<NA>")).sum())
                if changed:
                    new_entries.append(log_entry(f"Text case: {text_case}", changed, f"Column: {column}", len(cleaned), len(cleaned)))

        # 2. Robust Numeric conversions
        for column in numeric_to_convert:
            if column not in cleaned.columns:
                continue
            old = cleaned[column]
            converted, success_count, failed_count = numeric_conversion_with_stats(old)
            cleaned[column] = converted
            new_entries.append(
                log_entry(
                    "Convert text to numeric",
                    success_count,
                    f"Column: {column}; parsed: {success_count}; unparsed/invalid now missing: {failed_count}",
                    len(cleaned),
                    len(cleaned),
                )
            )

        # 3. Robust Date conversions
        convention_key = "dayfirst" if "Day first" in date_convention else "monthfirst" if "Month first" in date_convention else "iso" if "ISO" in date_convention else "auto"
        for column in date_to_convert:
            if column not in cleaned.columns:
                continue
            old = cleaned[column]
            converted, parsed_count, invalid_count, amb_count = parse_date_series(old, convention=convention_key)
            cleaned[column] = converted
            new_entries.append(
                log_entry(
                    "Parse date column",
                    parsed_count,
                    f"Column: {column}; parsed: {parsed_count}; invalid now missing: {invalid_count}; ambiguous detected: {amb_count}",
                    len(cleaned),
                    len(cleaned),
                )
            )

        # 4. Standardize headers
        if standardize_headers:
            before_columns = [str(c) for c in cleaned.columns]
            after_columns = normalize_headers(before_columns)
            if before_columns != after_columns:
                cleaned.columns = after_columns
                new_entries.append(log_entry("Standardize column names", len(before_columns), "Converted column names to unique snake_case names.", len(cleaned), len(cleaned)))

        # 5. Exact duplicate rows removal
        if remove_duplicates:
            before_rows = len(cleaned)
            duplicated_count = detect_exact_duplicates(cleaned)
            cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
            if duplicated_count:
                new_entries.append(log_entry("Remove exact duplicate rows", duplicated_count, "Removed rows identical across all columns.", before_rows, len(cleaned)))

        # 6. Missing value strategy
        before_missing_rows = len(cleaned)
        if missing_strategy == "Drop rows containing any missing value":
            cleaned = cleaned.dropna(axis=0, how="any").reset_index(drop=True)
            removed = before_missing_rows - len(cleaned)
            if removed:
                new_entries.append(log_entry("Drop rows with missing values", removed, "Removed a row if any column was missing.", before_missing_rows, len(cleaned)))
        elif missing_strategy in ("Fill numeric missing values with mean", "Fill numeric missing values with median", "Fill numeric missing values with 0"):
            method = "mean" if "mean" in missing_strategy else "median" if "median" in missing_strategy else "zero"
            filled_total = 0
            for column in cleaned.columns:
                if not is_numeric_column(cleaned[column]):
                    continue
                missing_mask = cleaned[column].isna()
                count_missing = int(missing_mask.sum())
                if count_missing == 0:
                    continue
                if method == "mean":
                    fill_val = cleaned[column].mean()
                elif method == "median":
                    fill_val = cleaned[column].median()
                else:
                    fill_val = 0
                if pd.notna(fill_val):
                    cleaned[column] = cleaned[column].fillna(fill_val)
                    filled_total += count_missing
            if filled_total:
                new_entries.append(log_entry(f"Fill numeric missing values: {method}", filled_total, "Applied to numeric columns only; non-numeric columns were left unchanged.", len(cleaned), len(cleaned)))
        elif missing_strategy == "Fill missing values with each column's mode":
            filled_total = 0
            for column in cleaned.columns:
                count_missing = int(cleaned[column].isna().sum())
                if not count_missing:
                    continue
                modes = cleaned[column].mode(dropna=True)
                if not modes.empty:
                    cleaned[column] = cleaned[column].fillna(modes.iloc[0])
                    filled_total += count_missing
            if filled_total:
                new_entries.append(log_entry("Fill missing values with mode", filled_total, "Filled each column with its most frequent non-missing value where a mode existed.", len(cleaned), len(cleaned)))
        elif missing_strategy == "Fill all missing values with a custom value":
            filled_total = int(cleaned.isna().sum().sum())
            if filled_total:
                cleaned = cleaned.fillna(custom_missing_value)
                new_entries.append(log_entry("Fill missing values with custom value", filled_total, f"Fill value: {custom_missing_value!r}.", len(cleaned), len(cleaned)))

        if not new_entries:
            new_entries.append(log_entry("No changes", 0, "No selected operation changed the current dataset.", rows_start, len(cleaned)))

        st.session_state.working_df = cleaned
        st.session_state.audit_log = audit_log + new_entries
        st.success(f"Cleaning steps applied. {len(new_entries)} log entr{'y' if len(new_entries) == 1 else 'ies'} added.")
        st.rerun()

    # Reset & Audit Controls
    reset_col, note_col = st.columns([1, 3])
    with reset_col:
        if st.button("Restore original uploaded data", use_container_width=True):
            st.session_state.working_df = original_df.copy(deep=True)
            st.session_state.audit_log = []
            st.rerun()
    with note_col:
        st.caption("Reset discards cleaning changes in this session and clears the log. It does not alter your source file.")

    # Before-and-After Cleaning Comparison Charts
    if len(audit_log) > 0 or working_df.shape != original_df.shape or summary["missing_cells"] != original_summary["missing_cells"]:
        st.divider()
        st.markdown("### Cleaning Impact: Before vs. After")
        comparison = compare_datasets(original_df, working_df)

        diff1, diff2, diff3, diff4 = st.columns(4)
        diff1.metric(
            "Rows",
            f"{summary['rows']:,}",
            delta=f"{comparison['delta_rows']:+,} rows" if comparison["delta_rows"] != 0 else "0",
        )
        diff2.metric(
            "Missing cells",
            f"{summary['missing_cells']:,}",
            delta=f"{comparison['delta_missing']:+,} cells" if comparison["delta_missing"] != 0 else "0",
            delta_color="inverse",
        )
        diff3.metric(
            "Duplicate rows",
            f"{summary['exact_duplicate_rows']:,}",
            delta=f"{comparison['delta_duplicates']:+,} rows" if comparison["delta_duplicates"] != 0 else "0",
            delta_color="inverse",
        )
        diff4.metric(
            "Completeness",
            f"{summary['completeness_percent']:.1f}%",
            delta=f"{comparison['delta_completeness']:+.1f}%" if comparison["delta_completeness"] != 0 else "0.0%",
        )

        # Before vs After Missing Values Chart
        col_comp_df = comparison["column_missing_comparison"]
        active_missing_cols = col_comp_df[(col_comp_df["Original Missing"] > 0) | (col_comp_df["Cleaned Missing"] > 0)].sort_values("Original Missing", ascending=False).head(15)

        if not active_missing_cols.empty:
            melted_comp = pd.melt(
                active_missing_cols,
                id_vars=["Column"],
                value_vars=["Original Missing", "Cleaned Missing"],
                var_name="Dataset",
                value_name="Missing Cells",
            )
            fig_before_after = px.bar(
                melted_comp,
                x="Column",
                y="Missing Cells",
                color="Dataset",
                barmode="group",
                title="Missing cells before vs. after cleaning (Top affected columns)",
                color_discrete_map={"Original Missing": "#EF4444", "Cleaned Missing": "#10B981"},
            )
            apply_chart_theme(fig_before_after, height=360)
            st.plotly_chart(fig_before_after, use_container_width=True)

    st.markdown("#### Current working preview")
    st.dataframe(working_df.head(50), use_container_width=True, height=320)
    st.markdown("#### Recent cleaning operations")
    if audit_log:
        st.dataframe(pd.DataFrame(audit_log).tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("No cleaning operations have been applied yet.")

    # ------------------------------------------------------------------
    # ADVANCED MISSING-VALUE FALLBACK (inside Cleaning Studio)
    # ------------------------------------------------------------------
    st.divider()
    with st.expander("🔧 Advanced Missing-Value Fallback (per-column)", expanded=False):
        st.caption("Select a column and a type-appropriate strategy. The fallback engine will not silently substitute zero or drop rows.")
        missing_cols = [str(c) for c in working_df.columns if int(working_df[c].isna().sum()) > 0]
        if not missing_cols:
            st.success("No columns with missing values in the current working dataset.")
        else:
            fb_col = st.selectbox("Column with missing values", missing_cols, key="fb_col_select")
            fb_series = working_df[fb_col]
            n_miss = int(fb_series.isna().sum())
            st.info(f"**{fb_col}**: {n_miss:,} missing value{'s' if n_miss != 1 else ''} ({n_miss / len(fb_series) * 100:.1f}%)")

            strategies = MissingValueFallback.suggest_strategies(fb_series, fb_col)
            fb_strategy = st.selectbox("Strategy", strategies, key="fb_strategy_select")

            fb_custom = None
            if "custom" in fb_strategy.lower():
                fb_custom = st.text_input("Custom fill value", key="fb_custom_val")

            if st.button("Preview & Apply", key="fb_apply_btn"):
                filled, cells_filled, cells_failed, msg = MissingValueFallback.apply(
                    fb_series, fb_strategy, custom_value=fb_custom
                )
                st.info(msg)
                if cells_filled > 0:
                    # Show preview
                    preview_df = pd.DataFrame({
                        "Before": fb_series.head(20),
                        "After": filled.head(20),
                    })
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    working_df[fb_col] = filled
                    st.session_state.working_df = working_df
                    new_log = log_entry(
                        f"Fallback fill: {fb_strategy}", cells_filled,
                        f"Column: {fb_col}; {msg}",
                        len(working_df), len(working_df),
                        column=fb_col, values_changed=cells_filled, values_failed=cells_failed,
                        status="Applied" if cells_failed == 0 else "Partially applied",
                    )
                    st.session_state.audit_log = audit_log + [new_log]
                    st.success(f"✅ Applied. {cells_filled} values filled, {cells_failed} failed.")
                    st.rerun()
                elif cells_failed > 0:
                    st.warning(f"⚠️ Strategy could not be applied: {msg}")

    # ------------------------------------------------------------------
    # FUZZY TEXT MATCHING (inside Cleaning Studio)
    # ------------------------------------------------------------------
    with st.expander("🔍 Fuzzy Text Matching — Inconsistent Categories", expanded=False):
        st.caption(
            f"Detect potential spelling variations in text columns using "
            f"**{FUZZY_BACKEND}** (token_sort_ratio). "
            "Similarity is a textual resemblance score, not a probability of correctness. "
            "Identifier-like columns are automatically skipped."
        )
        fuzz_text_cols = [c for c in text_columns(working_df) if not is_identifier_like(working_df[c], c)]
        if not fuzz_text_cols:
            st.info("No eligible text columns for fuzzy matching.")
        else:
            fc1, fc2 = st.columns([2, 1])
            with fc1:
                fuzz_col = st.selectbox("Column to analyze", fuzz_text_cols, key="fuzz_col")
            with fc2:
                fuzz_threshold = st.slider("Similarity threshold (%)", 60, 100, 80, key="fuzz_thresh")

            if st.button("Find similar groups", key="fuzz_find_btn"):
                matcher = FuzzyMatcher(threshold=fuzz_threshold)
                groups = matcher.find_groups(working_df[fuzz_col], fuzz_col)
                st.session_state["_fuzz_groups"] = groups
                st.session_state["_fuzz_col"] = fuzz_col

            groups = st.session_state.get("_fuzz_groups", [])
            fuzz_active_col = st.session_state.get("_fuzz_col", "")

            if groups and fuzz_active_col == (fuzz_col if fuzz_text_cols else ""):
                st.markdown(f"**{len(groups)} group(s) of similar values found** (threshold ≥ {fuzz_threshold}%)")
                for gi, g in enumerate(groups):
                    with st.container():
                        st.markdown(
                            f"**Group {gi + 1}**: Canonical suggestion: `{g['canonical']}` "
                            f"({g['canonical_count']} occurrences) — "
                            f"{g['total_affected_rows']} rows with variants"
                        )
                        var_df = pd.DataFrame(g["variants"])
                        st.dataframe(var_df, use_container_width=True, hide_index=True)

                        # User can pick canonical or type their own
                        all_vals = [g["canonical"]] + [v["value"] for v in g["variants"]]
                        canonical_choice = st.selectbox(
                            "Canonical value",
                            all_vals,
                            index=0,
                            key=f"fuzz_canonical_{gi}",
                        )
                        custom_canonical = st.text_input(
                            "Or enter custom canonical",
                            value="",
                            key=f"fuzz_custom_canon_{gi}",
                        )
                        final_canonical = custom_canonical.strip() if custom_canonical.strip() else canonical_choice

                        if st.button(f"Apply replacements → '{final_canonical}'", key=f"fuzz_apply_{gi}"):
                            replacements = {}
                            for v in g["variants"]:
                                if v["value"] != final_canonical:
                                    replacements[v["value"]] = final_canonical
                            if g["canonical"] != final_canonical:
                                replacements[g["canonical"]] = final_canonical

                            if replacements:
                                fm = FuzzyMatcher()
                                new_series, changed = fm.apply_replacements(working_df[fuzz_col], replacements)
                                working_df[fuzz_col] = new_series
                                st.session_state.working_df = working_df
                                log_details = "; ".join(f"'{k}' → '{v}'" for k, v in list(replacements.items())[:5])
                                new_log = log_entry(
                                    "Fuzzy match replacement", changed,
                                    f"Column: {fuzz_col}; {log_details}",
                                    len(working_df), len(working_df),
                                    column=fuzz_col, canonical_value=final_canonical,
                                    algorithm=FUZZY_BACKEND, threshold=fuzz_threshold,
                                    values_changed=changed,
                                )
                                st.session_state.audit_log = audit_log + [new_log]
                                st.session_state["_fuzz_groups"] = []
                                st.success(f"✅ {changed} values replaced with '{final_canonical}'.")
                                st.rerun()
                        st.divider()
            elif not groups and fuzz_active_col:
                st.info("No similar value groups found at this threshold. Try lowering the threshold.")

    # ------------------------------------------------------------------
    # REGEX CLEANING OPERATIONS (inside Cleaning Studio)
    # ------------------------------------------------------------------
    with st.expander("📐 Regex Pattern Cleaning", expanded=False):
        st.caption(
            "Apply regex-based find, extract, or replace operations on a column. "
            "A preview is shown before any modification. Invalid regex expressions are caught gracefully."
        )
        rx_col = st.selectbox("Column", list(working_df.columns), key="rx_col")
        rx_op = st.selectbox("Operation", ["Find matches", "Preview replacement", "Apply replacement", "Extract pattern"], key="rx_op")
        rx_pattern = st.text_input("Regex pattern", value="", key="rx_pattern", help="Python re syntax. Example: `\\d+` to find digits.")
        rx_replacement = ""
        if "replace" in rx_op.lower():
            rx_replacement = st.text_input("Replacement string", value="", key="rx_repl")

        if st.button("Run", key="rx_run_btn") and rx_pattern:
            series = working_df[rx_col]
            if rx_op == "Find matches":
                result_df, err = RegexCleaner.find_matches(series, rx_pattern)
                if err:
                    st.error(err)
                elif result_df.empty:
                    st.info("No matches found.")
                else:
                    st.dataframe(result_df.head(50), use_container_width=True, hide_index=True)
                    st.caption(f"{len(result_df)} value(s) matched.")
            elif rx_op == "Preview replacement":
                result_df, err = RegexCleaner.preview_replace(series, rx_pattern, rx_replacement)
                if err:
                    st.error(err)
                elif result_df.empty:
                    st.info("No values would change.")
                else:
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.caption(f"{len(result_df)} value(s) would change.")
            elif rx_op == "Apply replacement":
                preview_df, err = RegexCleaner.preview_replace(series, rx_pattern, rx_replacement)
                if err:
                    st.error(err)
                elif preview_df.empty:
                    st.info("No values would change.")
                else:
                    st.dataframe(preview_df.head(20), use_container_width=True, hide_index=True)
                    new_series, changed, err2 = RegexCleaner.apply_replace(series, rx_pattern, rx_replacement)
                    if err2:
                        st.error(err2)
                    else:
                        working_df[rx_col] = new_series
                        st.session_state.working_df = working_df
                        new_log = log_entry(
                            "Regex replacement", changed,
                            f"Column: {rx_col}; pattern: {rx_pattern!r} → {rx_replacement!r}",
                            len(working_df), len(working_df),
                            column=rx_col, regex_pattern=rx_pattern,
                            values_changed=changed,
                        )
                        st.session_state.audit_log = audit_log + [new_log]
                        st.success(f"✅ {changed} values replaced.")
                        st.rerun()
            elif rx_op == "Extract pattern":
                result_df, err = RegexCleaner.extract_pattern(series, rx_pattern)
                if err:
                    st.error(err)
                elif result_df.empty:
                    st.info("No matches extracted.")
                else:
                    st.dataframe(result_df.head(50), use_container_width=True, hide_index=True)
                    st.caption(f"{len(result_df)} value(s) matched.")

    # ------------------------------------------------------------------
    # MISSING DATA DETECTION (inside Cleaning Studio)
    # ------------------------------------------------------------------
    with st.expander("🔎 Missing Data Detection Engine", expanded=False):
        st.caption(
            "Advanced detection of NaN, empty strings, whitespace, recognized placeholders, "
            "and optional user-defined placeholders. "
            "Original values are preserved until you approve normalization above."
        )
        extra_ph = st.text_input(
            "Additional placeholders to detect (comma-separated)",
            value="", key="miss_extra_ph",
            help="E.g.: 'TBD, pending, -'. These will be checked case-insensitively.",
        )
        extra_list = [p.strip() for p in extra_ph.split(",") if p.strip()] if extra_ph else []
        engine = MissingDataEngine(extra_placeholders=extra_list)
        miss_summary = engine.summary(working_df)

        ms1, ms2, ms3, ms4 = st.columns(4)
        ms1.metric("Missing cells", f"{miss_summary['missing_cells']:,}")
        ms2.metric("Missing %", f"{miss_summary['missing_pct_overall']:.1f}%")
        ms3.metric("Affected rows", f"{miss_summary['affected_rows']:,}")
        ms4.metric("Total cells", f"{miss_summary['total_cells']:,}")

        miss_col_df = pd.DataFrame(miss_summary["columns"])
        miss_col_df = miss_col_df[miss_col_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
        if not miss_col_df.empty:
            st.dataframe(miss_col_df, use_container_width=True, hide_index=True)

        det_col = st.selectbox("Inspect column details", list(working_df.columns), key="miss_det_col")
        det_df = engine.detect(working_df[det_col])
        if det_df.empty:
            st.success(f"No missing values detected in '{det_col}'.")
        else:
            st.dataframe(det_df.head(50), use_container_width=True, hide_index=True)
            st.caption(f"{len(det_df)} missing value(s) detected. Original values shown — use cleaning controls above to normalize.")

# ==============================================================================
# TAB 5: DATA QUALITY RULES
# ==============================================================================
with quality_tab:
    st.subheader("Data Quality Rules & Test Cases")
    st.caption("Run configurable tests against the working dataset. Results show pass/fail/skip counts with failure percentages computed on applicable (non-skipped) records.")

    # ---------- Built-in tests ----------
    with st.expander("📋 Run built-in quality tests", expanded=True):
        run_builtin = st.button("Run all built-in tests", key="dq_run_builtin", type="primary", use_container_width=True)
        if run_builtin:
            results: list[dict[str, Any]] = []
            # Duplicates
            results.append(DataQualityTestRunner.test_duplicates(working_df))
            # Per-column tests
            for col_name in working_df.columns:
                col_name_str = str(col_name)
                series = working_df[col_name]
                # Missing required
                results.append(DataQualityTestRunner.test_missing_required(series, col_name_str))
                # Numeric validation
                if is_numeric_column(series) or col_name_str in find_numeric_text_candidates(working_df):
                    results.append(DataQualityTestRunner.test_invalid_numeric(series, col_name_str))
                # Date validation
                if pd.api.types.is_datetime64_any_dtype(series) or col_name_str in find_date_text_candidates(working_df):
                    results.append(DataQualityTestRunner.test_date_parsing(series, col_name_str))
                # Inconsistent categories
                if is_text_column(series) and not is_identifier_like(series, col_name_str):
                    results.append(DataQualityTestRunner.test_inconsistent_categories(series, col_name_str))

            st.session_state["_dq_results"] = results

        dq_results = st.session_state.get("_dq_results", [])
        if dq_results:
            res_df = pd.DataFrame(dq_results)
            # Color-code failures
            failed_tests = res_df[res_df["failed"] > 0]
            passed_tests = res_df[res_df["failed"] == 0]

            t1, t2, t3 = st.columns(3)
            t1.metric("Total tests", len(res_df))
            t2.metric("Passed", len(passed_tests))
            t3.metric("With failures", len(failed_tests))

            if not failed_tests.empty:
                st.markdown("##### ❌ Tests with failures")
                st.dataframe(
                    failed_tests[["test", "column", "checked", "passed", "failed", "skipped", "failure_pct", "suggestion"]],
                    use_container_width=True, hide_index=True,
                )
            if not passed_tests.empty:
                st.markdown("##### ✅ Passing tests")
                st.dataframe(
                    passed_tests[["test", "column", "checked", "passed", "failed", "skipped", "suggestion"]].head(20),
                    use_container_width=True, hide_index=True,
                )

    # ---------- Regex pattern validation ----------
    with st.expander("📐 Regex Pattern Validation", expanded=False):
        st.caption(
            "Validate a column against a predefined or custom regex pattern. "
            "Regex validates format, not real-world authenticity (e.g., a valid email format doesn't mean the address exists)."
        )
        pv_col = st.selectbox("Column to validate", list(working_df.columns), key="pv_col")
        pattern_names = list(PREDEFINED_PATTERNS.keys()) + ["Custom pattern"]
        pv_pattern_choice = st.selectbox("Pattern", pattern_names, key="pv_pattern_choice")

        if pv_pattern_choice == "Custom pattern":
            pv_regex = st.text_input("Enter regex", value="", key="pv_custom_regex")
            pv_desc = "User-defined pattern."
        else:
            pv_info = PREDEFINED_PATTERNS[pv_pattern_choice]
            pv_regex = pv_info["pattern"]
            pv_desc = pv_info["description"]
            st.caption(f"**Pattern**: `{pv_regex}`  \n{pv_desc}")

        if st.button("Validate column", key="pv_validate_btn") and pv_regex:
            validator = RegexValidator(pv_regex, pv_pattern_choice)
            if not validator.is_valid_regex:
                st.error(f"Invalid regex: {validator.error_message}")
            else:
                vr = validator.validate(working_df[pv_col])
                v1, v2, v3 = st.columns(3)
                v1.metric("Valid", f"{vr['valid_count']:,} ({vr['valid_pct']:.1f}%)")
                v2.metric("Invalid", f"{vr['invalid_count']:,} ({vr['invalid_pct']:.1f}%)")
                v3.metric("Missing/blank", f"{vr['missing_count']:,}")

                if vr["invalid_examples"]:
                    st.markdown("**Sample invalid values:**")
                    st.dataframe(
                        pd.DataFrame({"Invalid values": vr["invalid_examples"]}),
                        use_container_width=True, hide_index=True,
                    )
                if vr["valid_examples"]:
                    st.markdown("**Sample valid values:**")
                    st.dataframe(
                        pd.DataFrame({"Valid values": vr["valid_examples"][:10]}),
                        use_container_width=True, hide_index=True,
                    )

    # ---------- Custom range / no-negative tests ----------
    with st.expander("📏 Numeric Range & Business Rules", expanded=False):
        st.caption("Test numeric columns against explicit range constraints or business rules like 'no negatives'.")
        nr_col = st.selectbox("Numeric column", [str(c) for c in working_df.columns if is_numeric_column(working_df[c])], key="nr_col")
        if nr_col:
            nr1, nr2 = st.columns(2)
            with nr1:
                nr_min = st.number_input("Minimum (leave 0 for no lower bound)", value=0.0, key="nr_min")
            with nr2:
                nr_max = st.number_input("Maximum (leave 0 for no upper bound)", value=0.0, key="nr_max")
            nr_no_neg = st.checkbox("Disallow negative values", value=False, key="nr_no_neg")

            if st.button("Run range test", key="nr_run_btn"):
                min_v = nr_min if nr_min != 0 else None
                max_v = nr_max if nr_max != 0 else None
                if nr_no_neg:
                    result = DataQualityTestRunner.test_no_negatives(working_df[nr_col], nr_col)
                else:
                    result = DataQualityTestRunner.test_numeric_range(working_df[nr_col], nr_col, min_v, max_v)
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Checked", result["checked"])
                rc2.metric("Failed", result["failed"])
                rc3.metric("Failure %", f"{result['failure_pct']:.1f}%")
                if result["examples"]:
                    st.markdown("**Failing values:**")
                    st.write(result["examples"][:10])
                st.info(result["suggestion"])

# ==============================================================================
# TAB 6: EXPORT & LOG
# ==============================================================================
with export_tab:
    st.subheader("Export cleaned data and the audit trail")
    final_profile = profile_dataframe(working_df)
    final_summary = summarize_dataframe(working_df)

    report = {
        "application": "CleanSight - Data Cleaning & Quality Studio",
        "source_file": source_filename,
        "worksheet": st.session_state.get("source_sheet", ""),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "original_summary": original_summary,
        "cleaned_summary": final_summary,
        "cleaning_log": audit_log,
        "notes": [
            "Completeness is calculated as 100 * (1 - missing_cells / total_cells) with strict invariant integrity.",
            "Outliers are flagged using the standard 1.5 × IQR rule for review and are never automatically removed.",
            "Ambiguous formats and leading zeros on identifiers are preserved unless explicitly overridden by user request.",
        ],
    }

    st.write("Review the current row count, missing cells and cleaning history before exporting.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in export", f"{final_summary['rows']:,}")
    c2.metric("Missing cells", f"{final_summary['missing_cells']:,}")
    c3.metric("Completeness", f"{final_summary['completeness_percent']:.1f}%")
    c4.metric("Logged operations", f"{len(audit_log):,}")

    # Visual Summary of Audit Trail
    if audit_log:
        audit_chart_df = pd.DataFrame(audit_log)
        op_counts = audit_chart_df["operation"].value_counts().reset_index()
        op_counts.columns = ["Operation", "Count"]

        fig_audit = px.bar(
            op_counts,
            x="Count",
            y="Operation",
            orientation="h",
            title="Summary of applied transformations",
            text="Count",
            color="Operation",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_audit.update_traces(textposition="outside")
        fig_audit.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        apply_chart_theme(fig_audit, height=280)
        st.plotly_chart(fig_audit, use_container_width=True)

    base_name = re.sub(r"[^A-Za-z0-9_-]+", "_", source_filename.rsplit(".", 1)[0]).strip("_") or "dataset"
    csv_bytes = working_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Download cleaned CSV",
        data=csv_bytes,
        file_name=f"{base_name}_cleaned.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
    try:
        excel_bytes = make_excel_bytes(working_df, final_profile, audit_log)
        st.download_button(
            "Download Excel workbook (data + profile + log)",
            data=excel_bytes,
            file_name=f"{base_name}_cleaned_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Could not prepare the Excel workbook: {exc}")

    st.download_button(
        "Download column profile CSV",
        data=final_profile.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{base_name}_column_profile.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download JSON cleaning report",
        data=json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=f"{base_name}_cleaning_report.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("#### Full audit log")
    if audit_log:
        st.dataframe(pd.DataFrame(audit_log), use_container_width=True, hide_index=True)
    else:
        st.info("The audit log is empty until you apply one or more cleaning operations.")

st.divider()
st.caption("CleanSight is a portfolio studio. Validate transformations against your source data and business rules before relying on the exported result.")

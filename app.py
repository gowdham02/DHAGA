"""DHAGA FITLENS — internal return-reason intelligence MVP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from chains.config import load_model_config
from data.generate_demo_data import generate_records
from services.analytics import actionable_insights, calculate_analytics, estimate_cost
from services.data_processor import CsvValidationError, analyze_returns, read_and_validate_csv


st.set_page_config(page_title="DHAGA FITLENS", page_icon="🧵", layout="wide")


def init_state() -> None:
    for key, value in {"source_df": None, "cleaning_report": None, "results_df": None, "summary": None}.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_demo() -> None:
    demo = generate_records()
    demo["original_return_text"] = demo.apply(
        lambda row: " | ".join(part for part in [str(row["return_reason"]).strip(), str(row["return_reason_detail"]).strip()] if part),
        axis=1,
    )
    # Generator already produces valid strings, so no upload round-trip is required.
    st.session_state.source_df = demo
    st.session_state.cleaning_report = type("Report", (), {"input_rows": len(demo), "duplicate_rows_removed": 0, "blank_text_rows": int(demo["return_reason_detail"].eq("").sum())})()
    st.session_state.results_df = None
    st.session_state.summary = None


def source_is_synthetic(df: pd.DataFrame | None) -> bool:
    return bool(df is not None and "data_source" in df.columns and df["data_source"].astype(str).str.contains("SYNTHETIC", case=False).any())


def show_header() -> None:
    st.title("DHAGA FITLENS")
    st.subheader("AI-powered Return Reason Intelligence")
    st.caption("Turn unstructured return comments into actionable fit-related product insights.")
    if source_is_synthetic(st.session_state.source_df):
        st.info("SYNTHETIC DEMO DATA — this dashboard contains generated records, not Dhaga production data.")


def show_upload() -> None:
    show_header()
    st.markdown("Upload a return-record CSV or load a ready-to-run synthetic demo. Required columns: `order_id`, `sku`, `product_name`, `vendor`, `size`, `return_reason`, `return_reason_detail`.")
    left, right = st.columns([2, 1])
    with left:
        upload = st.file_uploader("Return records CSV", type=["csv"])
    with right:
        st.write("")
        if st.button("Load synthetic demo data", width="stretch"):
            load_demo()
            st.rerun()
    if upload is not None:
        try:
            df, report = read_and_validate_csv(upload)
            st.session_state.source_df, st.session_state.cleaning_report = df, report
            st.session_state.results_df, st.session_state.summary = None, None
            st.success(f"Valid CSV loaded: {len(df):,} unique records ready for analysis.")
        except CsvValidationError as error:
            st.error(str(error))
            return
    df = st.session_state.source_df
    if df is None:
        st.warning("No data is loaded yet. Load the synthetic dataset or upload a valid CSV.")
        return
    report = st.session_state.cleaning_report
    a, b, c = st.columns(3)
    a.metric("Input records", f"{report.input_rows:,}")
    b.metric("Duplicate rows removed", f"{report.duplicate_rows_removed:,}")
    c.metric("Records with no text", f"{report.blank_text_rows:,}")
    st.dataframe(df.head(20), width="stretch", hide_index=True)
    config = load_model_config()
    if config.has_live_llm:
        st.success(f"Live LangChain mode: classifier `{config.classifier_model}` → evaluator `{config.evaluator_model}`.")
    else:
        st.warning("Demo mode: no `LLM_API_KEY` is configured. The walkthrough uses transparent deterministic rules, not an LLM. Add a key in `.env` to run the real LangChain two-model workflow.")
    if st.button("Analyze Returns", type="primary", width="stretch"):
        run_analysis(df, config)


def run_analysis(df: pd.DataFrame, config) -> None:
    try:
        if config.has_live_llm and config.classifier_model == config.evaluator_model:
            st.error("Live analysis requires two different model names. Update CLASSIFIER_MODEL and EVALUATOR_MODEL.")
            return
        progress = st.progress(0, text="1/6 Loading data")
        stage = st.empty()

        def update(fraction: float, done: int, total: int) -> None:
            if fraction < 0.1:
                label = "2/6 Classifying returns"
            elif fraction < 0.7:
                label = "3/6 Routing FIT records and analyzing fit issues"
            elif fraction < 0.95:
                label = "4/6 Evaluating classifications"
            else:
                label = "5/6 Aggregating results"
            progress.progress(fraction, text=f"{label} — {done:,}/{total:,}")

        results, summary = analyze_returns(df, config, update)
        stage.write("6/6 Preparing dashboard")
        st.session_state.results_df, st.session_state.summary = results, summary
        progress.progress(1.0, text="Analysis complete")
        if summary.failed:
            st.warning(f"{summary.failed} record(s) failed structured processing and were sent to Human Review with the visible error.")
        st.success(f"Analysis complete: {summary.processed:,} processed, {summary.failed:,} failed.")
    except Exception as error:
        st.error(f"Analysis could not be completed: {error}")


def require_results() -> bool:
    if st.session_state.results_df is None:
        st.info("Load data and select Analyze Returns on the Upload Data page first.")
        return False
    return True


def show_dashboard() -> None:
    show_header()
    if not require_results():
        return
    results, summary = st.session_state.results_df, st.session_state.summary
    analytics = calculate_analytics(results)
    a, b, c, d = st.columns(4)
    a.metric("Total Returns", f"{analytics['total_returns']:,}")
    b.metric("FIT Returns", f"{analytics['fit_returns']:,}")
    c.metric("FIT Return Rate", f"{analytics['fit_return_rate']:.1%}")
    d.metric("Human Review Required", f"{analytics['human_review_count']:,}")
    st.caption("Fit-return rate = classified FIT returns ÷ analyzed return records. It is an insight signal, not a measure of overall company return-rate reduction.")
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown("#### Return category distribution")
        st.bar_chart(analytics["category_distribution"], x="category", y="returns")
    with chart_b:
        st.markdown("#### Fit issue distribution")
        st.bar_chart(analytics["fit_distribution"], x="fit_issue", y="returns")
    tables_a, tables_b = st.columns(2)
    with tables_a:
        st.markdown("#### Top problematic SKUs")
        st.dataframe(analytics["sku_table"].head(10), width="stretch", hide_index=True,
                     column_config={"fit_return_rate": st.column_config.ProgressColumn("Fit return rate", min_value=0, max_value=1, format="%.1%%")})
    with tables_b:
        st.markdown("#### Top problematic vendors")
        st.dataframe(analytics["vendor_table"].head(10), width="stretch", hide_index=True,
                     column_config={"fit_return_rate": st.column_config.ProgressColumn("Fit return rate", min_value=0, max_value=1, format="%.1%%")})
    st.markdown("#### Actionable Insights")
    for insight in actionable_insights(analytics):
        st.write(f"• {insight}")
    st.markdown("#### Cost and processing")
    st.write(f"Mode: **{summary.mode}**. Calls — classifier: {summary.classifier_calls:,}; FIT detail: {summary.fit_calls:,}; evaluator: {summary.evaluator_calls:,}; total: {summary.total_llm_calls:,}.")
    cost = estimate_cost(summary, load_model_config())
    st.caption(cost or "Cost estimate unavailable until model pricing is configured.")


def show_return_analysis() -> None:
    show_header()
    if not require_results():
        return
    df = st.session_state.results_df.copy()
    st.markdown("#### Return Analysis")
    query = st.text_input("Search order, SKU, product, vendor, or return text")
    categories = st.multiselect("Category", options=sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    if query:
        fields = ["order_id", "sku", "product_name", "vendor", "original_return_text"]
        mask = df[fields].astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        df = df[mask]
    df = df[df["category"].isin(categories)]
    columns = ["order_id", "sku", "product_name", "vendor", "size", "original_return_text", "category", "subcategory", "classification_confidence", "evaluation_status"]
    st.dataframe(df[columns], width="stretch", hide_index=True, height=520)
    st.caption(f"Showing {len(df):,} records.")


def show_human_review() -> None:
    show_header()
    if not require_results():
        return
    queue = calculate_analytics(st.session_state.results_df)["review_queue"]
    st.markdown("#### HUMAN REVIEW REQUIRED")
    st.caption("These cases were low-confidence, rejected by the evaluator, missing text, or failed structured processing. They are never silently accepted.")
    search = st.text_input("Search the review queue")
    review_categories = st.multiselect(
        "Filter by predicted category",
        options=sorted(queue["category"].dropna().unique()),
        default=sorted(queue["category"].dropna().unique()),
    )
    if search:
        queue = queue[queue.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)]
    queue = queue[queue["category"].isin(review_categories)]
    columns = ["order_id", "sku", "product_name", "vendor", "size", "original_return_text", "category", "subcategory", "classification_confidence", "evaluation_status", "evaluator_reason", "processing_error"]
    st.dataframe(queue[columns], width="stretch", hide_index=True, height=560)
    st.caption(f"{len(queue):,} cases require a Category-team decision.")


def main() -> None:
    init_state()
    with st.sidebar:
        st.markdown("## 🧵 FITLENS")
        st.markdown("### Navigation")
        page = st.radio("Go to", ["Upload Data", "Dashboard", "Return Analysis", "Human Review"], label_visibility="collapsed")
        st.divider()
        st.caption("Internal MVP · Dhaga & Co.")
        st.caption("Synthetic records are prominently labelled.")
    {"Upload Data": show_upload, "Dashboard": show_dashboard, "Return Analysis": show_return_analysis, "Human Review": show_human_review}[page]()


if __name__ == "__main__":
    main()

"""Public Crossref Dashboard (Streamlit)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from crossref_client import build_month_windows, run_serial_crossref_search
from defaults import (
    DEFAULT_CROSSREF_ROWS,
    DEFAULT_CROSSREF_TIMEOUT_SECONDS,
    DEFAULT_ISSNS,
    DEFAULT_OPENAI_MODELS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TARGET_QUERY_COUNT,
)
from query_generator import fetch_available_models_openai, generate_queries_openai


st.set_page_config(page_title="Public Crossref Dashboard", layout="wide")


def _normalize_query_lines(text_block: str) -> List[str]:
    lines = [line.strip() for line in text_block.splitlines()]
    lines = [line for line in lines if line]
    deduped = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    return deduped


def _parse_issn_text(text: str) -> List[str]:
    if not text.strip():
        return []
    separators_replaced = text.replace(",", "\n").replace(";", "\n")
    items = [item.strip() for item in separators_replaced.splitlines() if item.strip()]
    deduped = []
    seen = set()
    for item in items:
        key = item.upper()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def main() -> None:
    st.title("Public Crossref Dashboard")
    st.caption("Generate search queries with OpenAI, edit them, and run a serial Crossref search.")

    with st.sidebar:
        st.header("OpenAI Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        model_options = DEFAULT_OPENAI_MODELS
        model_fetch_error = None
        if api_key.strip():
            cached_key = st.session_state.get("model_cache_api_key")
            cached_models = st.session_state.get("model_cache_models")
            if cached_key != api_key.strip() or not cached_models:
                try:
                    fetched_models = fetch_available_models_openai(api_key=api_key.strip())
                    if fetched_models:
                        st.session_state["model_cache_api_key"] = api_key.strip()
                        st.session_state["model_cache_models"] = fetched_models
                except Exception as exc:
                    model_fetch_error = str(exc)
            model_options = st.session_state.get("model_cache_models") or DEFAULT_OPENAI_MODELS

        selected_model = st.selectbox("Model", options=model_options, index=0)
        custom_model = st.text_input("Custom model (optional)")
        model = custom_model.strip() or selected_model
        system_prompt = st.text_area(
            "System prompt (editable)",
            value=DEFAULT_SYSTEM_PROMPT,
            height=160,
        )
        if model_fetch_error:
            st.warning(f"Could not fetch models from API key. Using defaults. Details: {model_fetch_error}")

        st.header("Crossref Settings")
        mailto = st.text_input("Contact email for Crossref (recommended)")
        inter_task_sleep = st.number_input(
            "Pause between tasks (seconds)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=0.5,
            help="Sleep after each (query, journal, month) task to avoid 429 errors.",
        )
        inter_request_sleep = st.number_input(
            "Pause between paginated requests (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=1.2,
            step=0.1,
            help="Sleep between cursor-paginated requests inside a single task.",
        )
        st.caption("Search runs serially (one task at a time) to match the proven notebook loop.")
        st.caption("Rows per request is fixed to 1000 for maximum retrieval.")

    st.subheader("1) Research Input")
    research_query = st.text_area(
        "Natural language research query",
        placeholder="e.g. I am researching carbide titanium niobium chromium ceramics and high-temperature behavior.",
        height=120,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start month/year", value=date(2020, 1, 1))
    with col_b:
        end_date = st.date_input("End month/year", value=date.today())

    if start_date > end_date:
        st.error("Start month/year must be before end month/year.")
        return

    start_month = start_date.strftime("%Y-%m")
    end_month = end_date.strftime("%Y-%m")

    st.subheader("2) Generate and Edit Queries")
    if st.button(f"Generate {DEFAULT_TARGET_QUERY_COUNT} Queries", type="primary"):
        if not api_key.strip():
            st.error("Please enter an OpenAI API key in the sidebar.")
        elif not research_query.strip():
            st.error("Please enter a research query.")
        else:
            with st.spinner("Generating queries with OpenAI..."):
                try:
                    generated = generate_queries_openai(
                        api_key=api_key.strip(),
                        model=model,
                        research_query=research_query.strip(),
                        system_prompt=system_prompt.strip(),
                        target_count=DEFAULT_TARGET_QUERY_COUNT,
                    )
                except Exception as exc:
                    st.error(f"Failed to generate queries: {exc}")
                    generated = []

            if generated:
                st.session_state["editable_queries_text"] = "\n".join(generated)
                st.success(f"Generated {len(generated)} queries. You can edit them below.")

    editable_text = st.text_area(
        "Editable query list (one per line)",
        value=st.session_state.get("editable_queries_text", ""),
        height=220,
        key="editable_queries_text",
    )
    edited_queries = _normalize_query_lines(editable_text)
    st.caption(f"Final query count: {len(edited_queries)}")

    st.subheader("3) ISSN Input (Optional)")
    issn_text = st.text_area(
        "ISSNs (optional). One per line or comma-separated. If blank, default ISSN list is used.",
        height=160,
        placeholder="0272-8842\n0925-8388",
    )
    user_issns = _parse_issn_text(issn_text)
    final_issns = user_issns if user_issns else DEFAULT_ISSNS
    st.caption(f"Using {len(final_issns)} ISSN(s).")

    st.subheader("4) Run Serial Crossref Search")
    run_button = st.button("Run Search", type="primary")

    if run_button:
        if not edited_queries:
            st.error("Please generate or enter at least one query before running search.")
            return

        month_windows = build_month_windows(start_month, end_month)
        total_tasks = len(edited_queries) * len(final_issns) * len(month_windows)
        progress_text = st.empty()
        progress_bar = st.progress(0)
        log_area = st.empty()
        st.session_state["progress_log"] = []
        output_root = Path("public_crossref_dashboard") / "run_outputs"
        run_stamp = date.today().strftime("%Y%m%d")
        run_index = st.session_state.get("run_index", 0) + 1
        st.session_state["run_index"] = run_index
        run_dir = output_root / f"{run_stamp}_run_{run_index}"
        task_csv_dir = run_dir / "task_csvs"
        consolidated_csv_path = run_dir / "crossref_unified.csv"

        def _on_task_complete(
            completed: int,
            total: int,
            query: str,
            issn: str,
            month_window: str,
            doi_count: int,
            error: str | None,
            task_csv_path: str | None,
        ) -> None:
            pct = int((completed / max(total, 1)) * 100)
            progress_bar.progress(min(pct, 100))

            if error:
                line = (
                    f"[{completed}/{total} | {pct}%] FAILED  "
                    f"query='{query}' issn='{issn}' month='{month_window}' "
                    f"error='{error}'"
                )
                progress_text.warning(line)
            else:
                line = (
                    f"[{completed}/{total} | {pct}%] {doi_count:>4d} DOIs  "
                    f"query='{query}' issn='{issn}' month='{month_window}'"
                )
                if task_csv_path:
                    line += f"  csv={task_csv_path}"
                progress_text.info(line)

            log_lines = st.session_state.get("progress_log", [])
            log_lines.append(line)
            st.session_state["progress_log"] = log_lines[-200:]
            log_area.code("\n".join(st.session_state["progress_log"]))

        with st.spinner("Running serial Crossref search..."):
            try:
                result_df = run_serial_crossref_search(
                    queries=edited_queries,
                    issns=final_issns,
                    start_month=start_month,
                    end_month=end_month,
                    mailto=mailto.strip(),
                    rows=DEFAULT_CROSSREF_ROWS,
                    timeout_seconds=DEFAULT_CROSSREF_TIMEOUT_SECONDS,
                    inter_task_sleep=float(inter_task_sleep),
                    inter_request_sleep=float(inter_request_sleep),
                    task_csv_dir=str(task_csv_dir),
                    consolidated_csv_path=str(consolidated_csv_path),
                    progress_callback=_on_task_complete,
                )
            except Exception as exc:
                st.error(f"Crossref search failed: {exc}")
                return

        progress_bar.progress(100)
        progress_text.success(f"All serial tasks completed ({total_tasks}/{total_tasks}).")

        st.session_state["result_df"] = result_df
        st.session_state["consolidated_csv_path"] = str(consolidated_csv_path)
        st.success(f"Search complete. Found {len(result_df)} row(s).")
        st.caption(f"Single consolidated file saved at: {consolidated_csv_path}")

    if "result_df" in st.session_state:
        st.subheader("Results")
        result_df = st.session_state["result_df"]
        st.dataframe(result_df, use_container_width=True, height=500)

        csv_data = result_df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv_data,
            file_name=f"crossref_results_{start_month}_to_{end_month}.csv",
            mime="text/csv",
        )
        consolidated_path = st.session_state.get("consolidated_csv_path")
        if consolidated_path:
            st.caption(f"Consolidated file on disk: {consolidated_path}")

        with st.expander("Run Summary", expanded=False):
            st.write(f"Date range: {start_month} to {end_month}")
            st.write(f"Month windows: {len(build_month_windows(start_month, end_month))}")
            st.write(f"Queries used: {len(edited_queries)}")
            st.write(f"ISSNs used: {len(final_issns)}")
            st.write(f"Total serial tasks: {len(edited_queries) * len(final_issns) * len(build_month_windows(start_month, end_month))}")
            st.write(f"Unique DOIs: {result_df['doi'].nunique() if not result_df.empty else 0}")


if __name__ == "__main__":
    main()

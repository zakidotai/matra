"""Standalone test runner for the serial Crossref search.

(File name kept for compatibility with prior conversations; the search is now serial.)
"""

from __future__ import annotations

import os
from pathlib import Path

from crossref_client import run_serial_crossref_search
from defaults import DEFAULT_CROSSREF_ROWS, DEFAULT_CROSSREF_TIMEOUT_SECONDS, DEFAULT_ISSNS


def main() -> None:
    queries = ["carbide materials", "high entropy carbides"]
    issns = DEFAULT_ISSNS
    start_month = "2026-04"
    end_month = "2026-05"

    mailto = os.getenv("CROSSREF_EMAIL", "").strip()
    output_dir = Path("public_crossref_dashboard") / "test_outputs"
    task_csv_dir = output_dir / "task_csvs"
    consolidated_csv = output_dir / "crossref_test_unified.csv"

    print("Starting serial Crossref test run")
    print(f"Queries: {queries}")
    print(f"ISSN count: {len(issns)}")
    print(f"Date range: {start_month} to {end_month}")
    print(f"Rows/request: {DEFAULT_CROSSREF_ROWS}")
    print("Inter-task sleep: 5.0s | Inter-request sleep: 1.2s")

    def on_progress(
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
        if error:
            print(
                f"[{completed}/{total} | {pct}%] FAILED  "
                f"query='{query}' issn='{issn}' month='{month_window}' error='{error}'"
            )
        else:
            msg = (
                f"[{completed}/{total} | {pct}%] {doi_count:>4d} DOIs  "
                f"query='{query}' issn='{issn}' month='{month_window}'"
            )
            if task_csv_path:
                msg += f"  csv={task_csv_path}"
            print(msg)

    result_df = run_serial_crossref_search(
        queries=queries,
        issns=issns,
        start_month=start_month,
        end_month=end_month,
        mailto=mailto,
        rows=DEFAULT_CROSSREF_ROWS,
        timeout_seconds=DEFAULT_CROSSREF_TIMEOUT_SECONDS,
        inter_task_sleep=5.0,
        inter_request_sleep=1.2,
        task_csv_dir=str(task_csv_dir),
        consolidated_csv_path=str(consolidated_csv),
        progress_callback=on_progress,
    )

    print("\nRun completed.")
    print(f"Total rows: {len(result_df)}")
    print(f"Unique DOIs: {result_df['doi'].nunique() if not result_df.empty else 0}")
    print(f"Consolidated CSV: {consolidated_csv.resolve()}")


if __name__ == "__main__":
    main()

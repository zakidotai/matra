"""Crossref search client with serial query/ISSN/month execution.

Mirrors the proven loop from example/example_notebook_share.ipynb:
- cursor = "*"
- keep_paging = True
- max_rows = 1000
- swallow all per-request exceptions and stop paging (no raise_for_status)
- sleep 1s every 500 collected DOIs (polite pacing)
- sleep 5s between (query, issn, month) tasks
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests


BASE_URL = "https://api.crossref.org/works?query="


def _search_single_query_issn(
    *,
    query: str,
    issn: str,
    month_window: str,
    mailto: str,
    max_rows: int,
    timeout_seconds: int,
    inter_request_sleep: float,
) -> Tuple[List[Dict], int]:
    """Search Crossref for one (query, issn, month) task and return normalized rows.

    Implementation intentionally matches the notebook's loop:
    a try/except wraps the entire request+parse so any failure ends paging
    for this task cleanly without raising upward.
    """
    cursor = "*"
    keep_paging = True
    pages_fetched = 0
    out_rows: List[Dict] = []

    headers = {
        "Accept": "application/json",
    }
    if mailto:
        headers["mailto"] = mailto

    exact_query = f'"{query}"'

    params = {
        "filter": f"from-pub-date:{month_window},until-pub-date:{month_window},issn:{issn}",
    }

    while keep_paging:
        try:
            url = (
                BASE_URL
                + exact_query
                + "&rows="
                + str(max_rows)
                + "&cursor="
                + cursor
            )
            r = requests.get(url, headers=headers, timeout=timeout_seconds, params=params)
            payload = r.json().get("message", {})

            next_cursor_raw = payload.get("next-cursor")
            if next_cursor_raw:
                cursor = quote(next_cursor_raw, safe="")

            items = payload.get("items", []) or []

            if len(items) == 0:
                keep_paging = False

            for item in items:
                try:
                    container_titles = item.get("container-title") or []
                    journal_title = container_titles[0] if container_titles else "None"
                except Exception:
                    journal_title = "None"

                doi = (item.get("DOI") or "").strip()
                if not doi:
                    continue
                titles = item.get("title") or []
                article_title = titles[0] if titles else "None"

                out_rows.append(
                    {
                        "doi": doi,
                        "journal_title": journal_title,
                        "query_used": query,
                        "issn": issn,
                        "month_window": month_window,
                        "title": article_title,
                    }
                )

                if len(out_rows) % 500 == 0:
                    time.sleep(1)

            pages_fetched += 1

            # Polite pacing between paginated requests inside a task.
            if keep_paging and inter_request_sleep > 0:
                time.sleep(inter_request_sleep)

        except Exception:
            # Match notebook behavior: any error ends paging for this task.
            keep_paging = False

    return out_rows, pages_fetched


def build_month_windows(start_month: str, end_month: str) -> List[str]:
    """Build inclusive YYYY-MM month windows between start and end."""
    start_dt = datetime.strptime(start_month, "%Y-%m")
    end_dt = datetime.strptime(end_month, "%Y-%m")
    if start_dt > end_dt:
        return []

    months: List[str] = []
    year, month = start_dt.year, start_dt.month
    while (year < end_dt.year) or (year == end_dt.year and month <= end_dt.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def run_serial_crossref_search(
    *,
    queries: List[str],
    issns: List[str],
    start_month: str,
    end_month: str,
    mailto: str = "",
    rows: int = 1000,
    timeout_seconds: int = 100,
    inter_task_sleep: float = 5.0,
    inter_request_sleep: float = 1.2,
    task_csv_dir: Optional[str] = None,
    consolidated_csv_path: Optional[str] = None,
    progress_callback: Optional[
        Callable[[int, int, str, str, str, int, Optional[str], Optional[str]], None]
    ] = None,
) -> pd.DataFrame:
    """Run serial Crossref search across all query/ISSN/month combinations.

    progress_callback signature:
        (completed, total, query, issn, month_window, doi_count, error, task_csv_path)
    """
    month_windows = build_month_windows(start_month, end_month)

    tasks: List[Tuple[str, str, str]] = []
    for query in queries:
        for issn in issns:
            for month_window in month_windows:
                tasks.append((query, issn, month_window))

    columns = ["doi", "journal_title", "query_used", "issn", "month_window", "title"]

    if not tasks:
        return pd.DataFrame(columns=columns)

    csv_dir_path: Optional[Path] = None
    if task_csv_dir:
        csv_dir_path = Path(task_csv_dir)
        csv_dir_path.mkdir(parents=True, exist_ok=True)

    task_csv_paths: List[Path] = []
    all_rows: List[Dict] = []
    total_tasks = len(tasks)

    for idx, (query, issn, month_window) in enumerate(tasks, start=1):
        task_error: Optional[str] = None
        task_csv_path_str: Optional[str] = None
        rows_out: List[Dict] = []

        try:
            rows_out, _pages = _search_single_query_issn(
                query=query,
                issn=issn,
                month_window=month_window,
                mailto=mailto,
                max_rows=rows,
                timeout_seconds=timeout_seconds,
                inter_request_sleep=inter_request_sleep,
            )
            all_rows.extend(rows_out)

            if csv_dir_path is not None:
                task_df = pd.DataFrame(rows_out, columns=columns)
                safe_query = "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in query
                )[:80]
                safe_issn = "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in issn
                )
                safe_month = month_window.replace("-", "_")
                task_csv_path = csv_dir_path / f"{safe_query}__{safe_issn}__{safe_month}.csv"
                task_df.to_csv(task_csv_path, index=False)
                task_csv_paths.append(task_csv_path)
                task_csv_path_str = str(task_csv_path)

        except Exception as exc:
            task_error = str(exc)

        if progress_callback:
            progress_callback(
                idx,
                total_tasks,
                query,
                issn,
                month_window,
                len(rows_out),
                task_error,
                task_csv_path_str,
            )

        # Pause between serial tasks to be polite to Crossref.
        if idx < total_tasks and inter_task_sleep > 0:
            time.sleep(inter_task_sleep)

    if not all_rows:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset=["doi", "query_used"]).reset_index(drop=True)

    # Save single consolidated file and cleanup task-level files.
    if consolidated_csv_path:
        consolidated_path = Path(consolidated_csv_path)
        consolidated_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(consolidated_path, index=False)

        for task_csv in task_csv_paths:
            try:
                task_csv.unlink(missing_ok=True)
            except Exception:
                pass

        if csv_dir_path is not None:
            try:
                csv_dir_path.rmdir()
            except Exception:
                pass

    return df


# Backwards-compatible alias so existing callers keep working.
run_parallel_crossref_search = run_serial_crossref_search

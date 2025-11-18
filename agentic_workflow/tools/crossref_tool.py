"""
Crossref search tool for finding research papers
"""

import os
import time
import logging
import pandas as pd
import requests
from typing import List, Tuple, Optional
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..tool_registry import get_registry

logger = logging.getLogger(__name__)


def crossref_search_single(
    query: str,
    start_date: str,
    end_date: str,
    journal_issn: Optional[str],
    email: str,
    max_rows: int = 1000,
    base_url: str = "https://api.crossref.org/works?query="
) -> pd.DataFrame:
    """
    Search CrossRef API for a single query/journal/date combination
    
    Args:
        query: Search query string
        start_date: Start date in format 'YYYY-MM'
        end_date: End date in format 'YYYY-MM'
        journal_issn: Journal ISSN code (optional)
        email: Contact email for API
        max_rows: Maximum rows per page
        base_url: Base URL for CrossRef API
        
    Returns:
        DataFrame with columns: DOI, Query, PII, Title, Journal
    """
    results_df = pd.DataFrame(columns=['Query', 'PII', 'DOI', 'Title', 'Journal'])
    results_df = results_df.set_index('DOI')
    
    cursor = "*"
    keep_paging = True
    
    headers = {
        'Accept': 'application/json',
        'mailto': email
    }
    
    # Build filter
    filters = [f'from-pub-date:{start_date}', f'until-pub-date:{end_date}']
    if journal_issn:
        filters.append(f'issn:{journal_issn}')
    
    params = {
        'filter': ','.join(filters)
    }
    
    while keep_paging:
        try:
            url = base_url + query + "&rows=" + str(max_rows) + "&cursor=" + cursor
            r = requests.get(url, headers=headers, timeout=100, params=params)
            r.raise_for_status()
            
            data = r.json()
            cursor = quote(data['message']['next-cursor'], safe='')
            items = data['message']['items']
            
            if len(items) == 0:
                keep_paging = False
                break
            
            for item in items:
                try:
                    journal = item.get('container-title', ['None'])[0]
                except (KeyError, IndexError):
                    journal = 'None'
                
                try:
                    doi = item['DOI']
                    title = item.get('title', ['None'])[0] if item.get('title') else 'None'
                    results_df.loc[doi] = (query, 'None', title, journal)
                    
                    if results_df.shape[0] % 500 == 0:
                        time.sleep(1)
                except (KeyError, IndexError) as e:
                    logger.warning(f"Error processing item: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error during CrossRef search: {e}")
            keep_paging = False
    
    return results_df


def crossref_search_worker(args: Tuple) -> Tuple[str, pd.DataFrame]:
    """Worker function for parallel search"""
    query, start_date, end_date, journal_issn, email, max_rows, base_url = args
    df = crossref_search_single(query, start_date, end_date, journal_issn, email, max_rows, base_url)
    return f"{query}_{journal_issn}_{start_date}_{end_date}", df


def crossref_search_tool(
    queries: List[str],
    journal_issns: List[str],
    date_ranges: List[Tuple[str, str]],
    email: str,
    output_dir: str,
    n_workers: int = 4,
    max_rows: int = 1000,
    base_url: str = "https://api.crossref.org/works?query="
) -> dict:
    """
    Search CrossRef API for multiple queries, journals, and date ranges in parallel
    
    Args:
        queries: List of search query strings
        journal_issns: List of journal ISSN codes
        date_ranges: List of (start_date, end_date) tuples
        email: Contact email for API
        output_dir: Output directory for CSV files
        n_workers: Number of parallel workers
        max_rows: Maximum rows per page
        base_url: Base URL for CrossRef API
        
    Returns:
        Dictionary with 'success', 'csv_files', 'total_results', and 'error' keys
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare all search combinations
    search_tasks = []
    for query in queries:
        exact_query = f'"{query}"'
        for journal_issn in journal_issns:
            for start_date, end_date in date_ranges:
                search_tasks.append((
                    exact_query, start_date, end_date, journal_issn,
                    email, max_rows, base_url
                ))
    
    logger.info(f"Executing {len(search_tasks)} crossref searches with {n_workers} workers")
    
    csv_files = []
    all_results = []
    
    # Execute searches in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(crossref_search_worker, task): task for task in search_tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Searching CrossRef"):
            task = futures[future]
            try:
                key, df = future.result()
                if len(df) > 0:
                    # Save individual result
                    query, journal_issn, start_date, end_date = task[0], task[3], task[1], task[2]
                    s1 = start_date.split('-')[0]
                    s2 = end_date.split('-')[0]
                    
                    try:
                        journal_name = df['Journal'].iloc[0].strip()
                        jname = journal_name.replace(' ', '_')
                        journal_dir = os.path.join(output_dir, jname)
                        os.makedirs(journal_dir, exist_ok=True)
                        
                        csv_file = os.path.join(journal_dir, f'{journal_issn}_{s1}_{s2}.csv')
                        df.to_csv(csv_file)
                        csv_files.append(csv_file)
                        all_results.append(df)
                        logger.info(f"Saved {len(df)} results to {csv_file}")
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Could not save results for {key}: {e}")
                
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error in search task {task}: {e}")
    
    total_results = sum(len(df) for df in all_results)
    
    return {
        "success": True,
        "csv_files": csv_files,
        "total_results": total_results,
        "output_dir": output_dir,
        "error": None
    }


# Register the tool
registry = get_registry()
registry.register(
    name="crossref_search",
    func=crossref_search_tool,
    description="Search CrossRef API for research papers matching queries, journals, and date ranges. Returns paths to CSV files with search results.",
    parameters={
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search query strings"
            },
            "journal_issns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of journal ISSN codes"
            },
            "date_ranges": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "List of [start_date, end_date] pairs in format ['YYYY-MM', 'YYYY-MM']"
            },
            "email": {
                "type": "string",
                "description": "Email address for CrossRef API"
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory for CSV files"
            },
            "n_workers": {
                "type": "integer",
                "description": "Number of parallel workers",
                "default": 4
            }
        },
        "required": ["queries", "journal_issns", "date_ranges", "email", "output_dir"]
    }
)


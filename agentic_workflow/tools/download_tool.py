"""
Tool for downloading article XMLs from DOIs
"""

import os
import random
import time
import logging
import pandas as pd
import requests
from typing import List, Optional, Set
from bs4 import BeautifulSoup
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

from ..tool_registry import get_registry

logger = logging.getLogger(__name__)

# Master database file path (in output directory)
MASTER_DB_FILENAME = "master_downloaded_dois.csv"


def get_master_db_path(output_dir: str) -> str:
    """Get path to master database file"""
    return os.path.join(output_dir, MASTER_DB_FILENAME)


def load_downloaded_dois(output_dir: str) -> Set[str]:
    """
    Load set of successfully downloaded DOIs from master database.
    Only returns DOIs where Downloaded == True.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Set of DOI strings (normalized, lowercase)
    """
    master_db_path = get_master_db_path(output_dir)
    
    if not os.path.exists(master_db_path):
        return set()
    
    try:
        df = pd.read_csv(master_db_path)
        if 'DOI' not in df.columns:
            return set()
        # Normalize DOIs (lowercase, strip whitespace, remove URL prefixes)
        def normalize_doi(doi_str):
            doi_str = str(doi_str).lower().strip()
            # Remove common DOI URL prefixes
            for prefix in ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi.org/']:
                if doi_str.startswith(prefix):
                    doi_str = doi_str[len(prefix):]
            return doi_str.strip()
        
        # Only consider DOIs that were actually downloaded successfully
        if 'Downloaded' in df.columns:
            df = df[df['Downloaded'] == True]
        
        dois = set(df['DOI'].astype(str).apply(normalize_doi))
        dois.discard('nan')  # Remove any NaN values
        dois.discard('')  # Remove empty strings
        logger.info(f"Loaded {len(dois)} successfully downloaded DOIs from master database at {master_db_path}")
        return dois
    except Exception as e:
        logger.warning(f"Error loading master database: {e}")
        return set()


def update_master_db(output_dir: str, downloaded_dois: List[str], journal_names: List[str] = None):
    """
    Update master database with newly downloaded DOIs (marks them as Downloaded=True)
    
    Args:
        output_dir: Base output directory
        downloaded_dois: List of DOIs that were successfully downloaded
        journal_names: Optional list of journal names (same length as downloaded_dois)
    """
    if not downloaded_dois:
        return
    
    master_db_path = get_master_db_path(output_dir)
    
    # Normalize DOI function
    def normalize_doi(doi_str):
        doi_str = str(doi_str).lower().strip()
        for prefix in ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi.org/']:
            if doi_str.startswith(prefix):
                doi_str = doi_str[len(prefix):]
        return doi_str.strip()
    
    # Load existing database or create new
    if os.path.exists(master_db_path):
        try:
            df_existing = pd.read_csv(master_db_path)
        except:
            df_existing = pd.DataFrame(columns=['DOI', 'Journal', 'Downloaded'])
    else:
        df_existing = pd.DataFrame(columns=['DOI', 'Journal', 'Downloaded'])
    
    # Ensure Downloaded column exists (backward compatibility with old master DBs)
    if 'Downloaded' not in df_existing.columns:
        df_existing['Downloaded'] = True  # Assume old entries were downloaded
    
    # Get existing normalized DOIs
    if 'DOI' in df_existing.columns and len(df_existing) > 0:
        existing_normalized = set(df_existing['DOI'].astype(str).apply(normalize_doi))
    else:
        existing_normalized = set()
    
    # For DOIs already in DB, update their Downloaded flag to True
    dois_to_update = set()
    new_entries = []
    for i, doi in enumerate(downloaded_dois):
        doi_normalized = normalize_doi(doi)
        if not doi_normalized or doi_normalized == 'nan' or doi_normalized == '':
            continue
        if doi_normalized in existing_normalized:
            dois_to_update.add(doi_normalized)
        else:
            entry = {'DOI': doi, 'Downloaded': True}
            if journal_names and i < len(journal_names):
                entry['Journal'] = journal_names[i]
            new_entries.append(entry)
            existing_normalized.add(doi_normalized)
    
    # Update existing entries to Downloaded=True
    if dois_to_update and len(df_existing) > 0:
        df_existing['_norm'] = df_existing['DOI'].astype(str).apply(normalize_doi)
        df_existing.loc[df_existing['_norm'].isin(dois_to_update), 'Downloaded'] = True
        df_existing = df_existing.drop(columns=['_norm'])
    
    # Add new entries
    if new_entries:
        df_new = pd.DataFrame(new_entries)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_existing
    
    # Remove duplicates (keep first occurrence)
    df_combined = df_combined.drop_duplicates(subset=['DOI'], keep='first')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    df_combined.to_csv(master_db_path, index=False)
    updated_count = len(dois_to_update) + len(new_entries)
    logger.info(f"Updated master database: {len(new_entries)} new + {len(dois_to_update)} marked downloaded. Total: {len(df_combined)} DOIs")


def download_article_xml(
    doi: str,
    journal_name: str,
    output_base: str,
    api_keys: List[str],
    elsevier_api_base: str = "https://api.elsevier.com/content/article/doi/"
) -> dict:
    """
    Download XML for a single article from Elsevier API
    
    Args:
        doi: Article DOI
        journal_name: Journal name for organizing files
        output_base: Base output directory
        api_keys: List of Elsevier API keys
        elsevier_api_base: Base URL for Elsevier API
        
    Returns:
        Dictionary with 'success', 'pii', 'xml_path', and 'error' keys
    """
    jdir = '_'.join(journal_name.split())
    output_dir = os.path.join(output_base, jdir)
    os.makedirs(output_dir, exist_ok=True)
    
    api_id = random.randint(0, len(api_keys) - 1)
    xml_url = f"{elsevier_api_base}{doi}?APIKey={api_keys[api_id]}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(xml_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')

        # example for pii : <prism:url>https://api.elsevier.com/content/article/pii/S0272884225027476</prism:url>
        # pii is last part of the url
        pii_element = soup.find('prism:url')
        pii = pii_element.text.split('/')[-1]
        
        if pii_element is None:
            return {
                "success": False,
                "pii": None,
                "xml_path": None,
                "error": "PII not found in XML"
            }
        
        pii = pii_element.text
        pii_path = os.path.join(output_dir, str(pii))
        xml_path = os.path.join(pii_path, f"{pii}.xml")
        
        os.makedirs(pii_path, exist_ok=True)
        
        with open(xml_path, 'w', encoding='utf-8') as file:
            file.write(str(soup))
        
        return {
            "success": True,
            "pii": pii,
            "xml_path": xml_path,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "pii": None,
            "xml_path": None,
            "error": str(e)
        }


def download_wrapper(args: tuple) -> dict:
    """Wrapper for parallel download"""
    idx, doi, journal, output_base, api_keys, elsevier_api_base = args
    
    # Validate DOI
    if pd.isna(doi) or doi == '' or str(doi).strip() == '':
        return {
            "success": False,
            "idx": idx,
            "doi": str(doi) if not pd.isna(doi) else "nan",
            "pii": None,
            "xml_path": None,
            "error": "Invalid or missing DOI"
        }
    
    # Ensure DOI is a string
    doi = str(doi).strip()
    
    result = download_article_xml(doi, journal, output_base, api_keys, elsevier_api_base)
    result['idx'] = idx
    result['doi'] = doi
    
    # Rate-limit: sleep 1 + random(0,1) seconds to avoid 429 Too Many Requests
    time.sleep(1 + random.random())
    
    return result


def add_dois_to_master_db(output_dir: str, dois: List[str], source: str = "consolidated_csv", downloaded: bool = False):
    """
    Add DOIs to master database with a Downloaded flag.
    DOIs added here with downloaded=False will NOT be skipped on future runs.
    
    Args:
        output_dir: Base output directory
        dois: List of DOIs to add
        source: Source of the DOIs (e.g., "consolidated_csv")
        downloaded: Whether these DOIs were successfully downloaded
    """
    if not dois:
        return
    
    master_db_path = get_master_db_path(output_dir)
    
    # Load existing database
    if os.path.exists(master_db_path):
        try:
            df_existing = pd.read_csv(master_db_path)
        except:
            df_existing = pd.DataFrame(columns=['DOI', 'Journal', 'Source', 'Downloaded'])
    else:
        df_existing = pd.DataFrame(columns=['DOI', 'Journal', 'Source', 'Downloaded'])
    
    # Ensure Downloaded column exists (backward compatibility)
    if 'Downloaded' not in df_existing.columns:
        df_existing['Downloaded'] = True  # Assume old entries were downloaded
    
    # Normalize DOI function
    def normalize_doi(doi_str):
        doi_str = str(doi_str).lower().strip()
        for prefix in ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi.org/']:
            if doi_str.startswith(prefix):
                doi_str = doi_str[len(prefix):]
        return doi_str.strip()
    
    # Get existing normalized DOIs
    if 'DOI' in df_existing.columns:
        existing_normalized = set(df_existing['DOI'].astype(str).apply(normalize_doi))
    else:
        existing_normalized = set()
    
    # Add new DOIs (only ones not already in master DB)
    new_entries = []
    for doi in dois:
        doi_normalized = normalize_doi(doi)
        if doi_normalized and doi_normalized != 'nan' and doi_normalized != '' and doi_normalized not in existing_normalized:
            new_entries.append({
                'DOI': doi,  # Store original format
                'Journal': 'Unknown',
                'Source': source,
                'Downloaded': downloaded
            })
            existing_normalized.add(doi_normalized)
    
    if new_entries:
        df_new = pd.DataFrame(new_entries)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['DOI'], keep='first')
        os.makedirs(output_dir, exist_ok=True)
        df_combined.to_csv(master_db_path, index=False)
        logger.info(f"Added {len(new_entries)} DOIs from {source} to master database (downloaded={downloaded}). Total: {len(df_combined)} DOIs")


def download_articles_tool(
    csv_path: str,
    api_keys: List[str],
    output_base: str,
    n_workers: int = 20,
    elsevier_api_base: str = "https://api.elsevier.com/content/article/doi/"
) -> dict:
    """
    Download article XMLs from DOIs in parallel
    
    Args:
        csv_path: Path to CSV file containing DOIs and Journal columns
        api_keys: List of Elsevier API keys
        output_base: Base output directory
        n_workers: Number of parallel workers
        elsevier_api_base: Base URL for Elsevier API
        
    Returns:
        Dictionary with 'success', 'total', 'successful', 'failed', 'error_log', and 'error' keys
    """
    if not os.path.exists(csv_path):
        return {
            "success": False,
            "total": 0,
            "successful": 0,
            "failed": 0,
            "error_log": None,
            "error": f"CSV file not found: {csv_path}"
        }
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "success": False,
            "total": 0,
            "successful": 0,
            "failed": 0,
            "error_log": None,
            "error": f"Error reading CSV: {str(e)}"
        }
    
    # Ensure required columns exist
    if 'DOI' not in df.columns:
        return {
            "success": False,
            "total": 0,
            "successful": 0,
            "failed": 0,
            "error_log": None,
            "error": "CSV file must contain 'DOI' column"
        }
    
    if 'Journal' not in df.columns:
        df['Journal'] = 'Unknown'
    
    # Filter out rows with NaN or empty DOIs
    initial_count = len(df)
    df = df[df['DOI'].notna() & (df['DOI'] != '') & (df['DOI'].astype(str).str.strip() != '')]
    filtered_count = len(df)
    
    if filtered_count == 0:
        return {
            "success": False,
            "total": 0,
            "successful": 0,
            "failed": 0,
            "error_log": None,
            "error": f"No valid DOIs found in CSV. Removed {initial_count - filtered_count} rows with invalid DOIs."
        }
    
    if initial_count != filtered_count:
        logger.warning(f"Filtered out {initial_count - filtered_count} rows with invalid DOIs")
    
    # Clean journal names
    df['Journal'] = df['Journal'].str.replace(" &amp; ", ' and ')
    df = df.reset_index(drop=True)
    
    # Store initial count before filtering
    initial_count = len(df)
    df_original = df.copy()  # Save original for tracking all DOIs in CSV
    
    # Load master database of already downloaded DOIs
    # output_base is like "./output/corpus_query_name", we need the base output directory
    # The master DB should be in the base output directory (e.g., ./output/master_downloaded_dois.csv)
    # Simple approach: go up one level from output_base to get base output directory
    output_dir = os.path.dirname(os.path.abspath(output_base))
    
    # If we're in a subdirectory like "corpus_xxx", go up one more level to get "output"
    # This handles cases like: ./output/corpus_query_name -> ./output
    if os.path.basename(output_dir).startswith('corpus_') or 'corpus' in os.path.basename(output_dir).lower():
        output_dir = os.path.dirname(output_dir)
    
    # Ensure we have a valid directory (at least the parent of output_base)
    if not output_dir or output_dir == os.path.abspath(output_base):
        # Fallback: use parent directory
        output_dir = os.path.dirname(os.path.abspath(output_base))
        if not output_dir:
            output_dir = "."
    
    master_db_path = get_master_db_path(output_dir)
    logger.info(f"Using master database directory: {output_dir}")
    logger.info(f"Master database path: {master_db_path}")
    
    # Load all DOIs from master database (both downloaded and seen in previous consolidated CSVs)
    downloaded_dois = load_downloaded_dois(output_dir)
    logger.info(f"Found {len(downloaded_dois)} DOIs in master database")
    
    # Filter out already downloaded/seen DOIs FIRST (before adding current CSV DOIs)
    # This ensures we only download genuinely new articles
    # Normalize DOIs the same way as in load_downloaded_dois
    def normalize_doi(doi_str):
        doi_str = str(doi_str).lower().strip()
        # Remove common DOI URL prefixes
        for prefix in ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi.org/']:
            if doi_str.startswith(prefix):
                doi_str = doi_str[len(prefix):]
        return doi_str.strip()
    
    df['DOI_normalized'] = df['DOI'].astype(str).apply(normalize_doi)
    
    # Debug: show sample DOIs
    if len(df) > 0:
        sample_dois = df['DOI_normalized'].head(3).tolist()
        logger.debug(f"Sample normalized DOIs from CSV: {sample_dois}")
        if downloaded_dois:
            sample_downloaded = list(downloaded_dois)[:3]
            logger.debug(f"Sample DOIs from master DB: {sample_downloaded}")
    
    df = df[~df['DOI_normalized'].isin(downloaded_dois)]
    df = df.drop(columns=['DOI_normalized'])
    df = df.reset_index(drop=True)
    
    skipped_count = initial_count - len(df)
    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} articles already in master database")
    elif len(downloaded_dois) > 0:
        logger.warning(f"Master database has {len(downloaded_dois)} DOIs but none matched. This might indicate a normalization issue.")
    
    total = len(df)
    if total == 0:
        logger.info("All articles already downloaded. Nothing to download.")
        return {
            "success": True,
            "total": initial_count,
            "successful": 0,
            "failed": 0,
            "skipped": skipped_count,
            "error_log": None,
            "output_base": output_base,
            "error": None
        }
    
    logger.info(f"Downloading {total} new articles (skipped {skipped_count} already downloaded) with {n_workers} workers")
    
    # Prepare download tasks
    download_tasks = [
        (idx, row['DOI'], row['Journal'], output_base, api_keys, elsevier_api_base)
        for idx, row in df.iterrows()
    ]
    
    # Execute downloads in parallel
    results = Parallel(n_jobs=n_workers)(
        delayed(download_wrapper)(task) for task in tqdm(download_tasks, desc="Downloading articles")
    )
    
    # Process results
    successful = sum(1 for r in results if r['success'])
    failed = total - successful
    
    # Update master database with successfully downloaded DOIs
    if successful > 0:
        successful_dois = [r['doi'] for r in results if r['success']]
        successful_journals = [r.get('journal', 'Unknown') for r in results if r['success']]
        # Use the same output_dir we used for loading
        update_master_db(output_dir, successful_dois, successful_journals)
    
    # Track failed DOIs in master DB with Downloaded=False so they are retried next run
    failed_dois = [r['doi'] for r in results if not r['success']]
    if failed_dois:
        add_dois_to_master_db(output_dir, failed_dois, source="consolidated_csv", downloaded=False)
        logger.info(f"Tracked {len(failed_dois)} failed DOIs in master database (Downloaded=False, will retry next run)")
    
    # Log errors
    errors = [r for r in results if not r['success']]
    error_log_path = os.path.join(output_base, 'download_errors.txt')
    if errors:
        with open(error_log_path, 'w') as f:
            for err in errors:
                f.write(f"{err['doi']}: {err['error']}\n")
        logger.warning(f"{failed} downloads failed. See {error_log_path}")
    else:
        error_log_path = None
    
    logger.info(f"Successfully downloaded {successful}/{total} new articles (skipped {skipped_count} already downloaded)")
    
    return {
        "success": True,
        "total": initial_count,
        "successful": successful,
        "failed": failed,
        "skipped": skipped_count,
        "error_log": error_log_path,
        "output_base": output_base,
        "error": None
    }


# Register the tool
registry = get_registry()
registry.register(
    name="download_articles",
    func=download_articles_tool,
    description="Download article XMLs from DOIs using Elsevier API. Reads DOIs from CSV file and downloads in parallel. Returns download statistics.",
    parameters={
        "type": "object",
        "properties": {
            "csv_path": {
                "type": "string",
                "description": "Path to CSV file containing DOI and Journal columns"
            },
            "api_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of Elsevier API keys"
            },
            "output_base": {
                "type": "string",
                "description": "Base output directory for downloaded XMLs"
            },
            "n_workers": {
                "type": "integer",
                "description": "Number of parallel workers",
                "default": 20
            }
        },
        "required": ["csv_path", "api_keys", "output_base"]
    }
)


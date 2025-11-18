"""
Tool for downloading article XMLs from DOIs
"""

import os
import random
import logging
import pandas as pd
import requests
from typing import List, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
from joblib import Parallel, delayed

from ..tool_registry import get_registry

logger = logging.getLogger(__name__)


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
        pii_element = soup.find('xocs:pii-unformatted')
        
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
    result = download_article_xml(doi, journal, output_base, api_keys, elsevier_api_base)
    result['idx'] = idx
    result['doi'] = doi
    return result


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
    
    # Clean journal names
    df['Journal'] = df['Journal'].str.replace(" &amp; ", ' and ')
    df = df.reset_index(drop=True)
    
    total = len(df)
    logger.info(f"Downloading {total} articles with {n_workers} workers")
    
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
    
    logger.info(f"Successfully downloaded {successful}/{total} articles")
    
    return {
        "success": True,
        "total": total,
        "successful": successful,
        "failed": failed,
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


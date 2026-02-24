"""
Tool for building database from downloaded XMLs
"""

import os
import logging
import pandas as pd
from typing import Optional
from bs4 import BeautifulSoup
from p_tqdm import p_map

from ..tool_registry import get_registry

logger = logging.getLogger(__name__)


def get_pii2doi(xml_path: str, _=None) -> Optional[str]:
    """Extract DOI from XML file"""
    try:
        with open(xml_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'xml')
        doi_element = soup.find('doi')
        if doi_element:
            return doi_element.text.lower()
        return None
    except Exception as e:
        logger.warning(f"Error extracting DOI from {xml_path}: {e}")
        return None


def get_abstract(xml_path: str, _=None) -> Optional[str]:
    """Extract abstract from XML file"""
    try:
        with open(xml_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'xml')
        abstract_element = soup.find('dc:description')
        if abstract_element:
            return abstract_element.text.strip()
        return None
    except Exception as e:
        logger.warning(f"Error extracting abstract from {xml_path}: {e}")
        return None


def get_title(xml_path: str, _=None) -> Optional[str]:
    """Extract title from XML file"""
    try:
        with open(xml_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'xml')
        title_element = soup.find('dc:title')
        if title_element:
            return title_element.text.strip()
        return None
    except Exception as e:
        logger.warning(f"Error extracting title from {xml_path}: {e}")
        return None


def build_database_tool(
    corpus_dir: str,
    output_csv: str,
    n_workers: int = 20
) -> dict:
    """
    Build database from downloaded XMLs by extracting metadata
    
    Args:
        corpus_dir: Directory containing downloaded XMLs organized by journal
        output_csv: Path to save the database CSV file
        n_workers: Number of parallel workers for processing
        
    Returns:
        Dictionary with 'success', 'output_csv', 'total_articles', and 'error' keys
    """
    if not os.path.exists(corpus_dir):
        return {
            "success": False,
            "output_csv": None,
            "total_articles": 0,
            "error": f"Corpus directory not found: {corpus_dir}"
        }
    
    logger.info(f"Building database from {corpus_dir}")
    
    dfbig = pd.DataFrame()
    journals = sorted([
        d for d in os.listdir(corpus_dir)
        if os.path.isdir(os.path.join(corpus_dir, d)) and d != 'combined_xmls'
    ])
    
    if not journals:
        return {
            "success": False,
            "output_csv": None,
            "total_articles": 0,
            "error": f"No journal directories found in {corpus_dir}"
        }
    
    logger.info(f"Found {len(journals)} journal(s) to process")
    
    for journal in journals:
        try:
            journal_path = os.path.join(corpus_dir, journal)
            piis = sorted([
                d for d in os.listdir(journal_path)
                if os.path.isdir(os.path.join(journal_path, d))
            ])
            
            if not piis:
                logger.warning(f"No articles (PII subdirs) found in {journal} at {journal_path}")
                continue
            
            xml_paths = [
                os.path.join(journal_path, pii, f"{pii}.xml")
                for pii in piis
            ]
            
            # Filter to only existing XML files
            xml_paths = [p for p in xml_paths if os.path.exists(p)]
            
            if not xml_paths:
                logger.warning(f"No XML files found in {journal} (expected <PII>/<PII>.xml under {journal_path})")
                continue
            
            # Extract metadata in parallel
            logger.info(f"Processing {len(xml_paths)} articles from {journal}")
            dois = p_map(get_pii2doi, xml_paths, xml_paths, num_cpus=n_workers, disable=True)
            abstracts = p_map(get_abstract, xml_paths, xml_paths, num_cpus=n_workers, disable=True)
            titles = p_map(get_title, xml_paths, xml_paths, num_cpus=n_workers, disable=True)
            
            # Build dataframe
            dfsave = pd.DataFrame({
                'title': titles,
                'abstracts': abstracts,
                'doi': dois,
                'pii': piis[:len(xml_paths)],
                'journal': [journal] * len(xml_paths)
            })
            
            dfbig = pd.concat([dfbig, dfsave], ignore_index=True)
            logger.info(f"Processed {len(dfsave)} articles from {journal}")
            
        except Exception as e:
            logger.error(f"Error processing journal {journal}: {e}")
            continue
    
    # Merge with ALL existing corpus_*.csv database files in the output directory
    # so the final database accumulates articles across all query runs
    new_articles = len(dfbig)
    output_parent = os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.'
    os.makedirs(output_parent, exist_ok=True)
    
    import glob
    existing_csvs = glob.glob(os.path.join(output_parent, 'corpus_*.csv'))
    existing_frames = []
    for csv_path in existing_csvs:
        try:
            ef = pd.read_csv(csv_path)
            if len(ef) > 0:
                existing_frames.append(ef)
                logger.info(f"Loaded {len(ef)} articles from existing database: {os.path.basename(csv_path)}")
        except Exception as e:
            logger.warning(f"Could not read {csv_path}, skipping: {e}")
    
    if existing_frames:
        existing_df = pd.concat(existing_frames, ignore_index=True)
        logger.info(f"Merging {new_articles} new articles with {len(existing_df)} existing articles from {len(existing_frames)} database file(s)")
        dfbig = pd.concat([existing_df, dfbig], ignore_index=True)
        # Deduplicate by PII (primary key), keeping the latest entry
        if 'pii' in dfbig.columns:
            before_dedup = len(dfbig)
            dfbig = dfbig.drop_duplicates(subset=['pii'], keep='last')
            dupes_removed = before_dedup - len(dfbig)
            if dupes_removed > 0:
                logger.info(f"Removed {dupes_removed} duplicate articles (by PII)")
    
    if len(dfbig) == 0:
        err_log = os.path.join(corpus_dir, "download_errors.txt")
        hint = f" Check {err_log}" if os.path.exists(err_log) else ""
        logger.warning("No articles (PII subdirs with XML) found in corpus; writing empty database CSV so workflow can continue.%s", hint)
        empty_df = pd.DataFrame(columns=['title', 'abstracts', 'doi', 'pii', 'journal'])
        empty_df.to_csv(output_csv, index=False)
        return {
            "success": True,
            "output_csv": output_csv,
            "total_articles": 0,
            "error": "No articles found in corpus directory (empty CSV written)"
        }
    
    # Save database
    dfbig.reset_index(drop=True, inplace=True)
    dfbig.to_csv(output_csv, index=False)
    
    logger.info(f"Built database with {len(dfbig)} total articles ({new_articles} new)")
    logger.info(f"Saved to {output_csv}")
    
    return {
        "success": True,
        "output_csv": output_csv,
        "total_articles": len(dfbig),
        "new_articles": new_articles,
        "error": None
    }


# Register the tool
registry = get_registry()
registry.register(
    name="build_database",
    func=build_database_tool,
    description="Extract metadata (DOI, title, abstract, PII) from downloaded XML files and create a CSV database. Processes XMLs in parallel.",
    parameters={
        "type": "object",
        "properties": {
            "corpus_dir": {
                "type": "string",
                "description": "Directory containing downloaded XMLs organized by journal"
            },
            "output_csv": {
                "type": "string",
                "description": "Path to save the database CSV file"
            },
            "n_workers": {
                "type": "integer",
                "description": "Number of parallel workers",
                "default": 20
            }
        },
        "required": ["corpus_dir", "output_csv"]
    }
)


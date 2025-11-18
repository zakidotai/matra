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
                logger.warning(f"No articles found in {journal}")
                continue
            
            xml_paths = [
                os.path.join(journal_path, pii, f"{pii}.xml")
                for pii in piis
            ]
            
            # Filter to only existing XML files
            xml_paths = [p for p in xml_paths if os.path.exists(p)]
            
            if not xml_paths:
                logger.warning(f"No XML files found in {journal}")
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
    
    if len(dfbig) == 0:
        return {
            "success": False,
            "output_csv": None,
            "total_articles": 0,
            "error": "No articles found in corpus directory"
        }
    
    # Save database
    dfbig.reset_index(drop=True, inplace=True)
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    dfbig.to_csv(output_csv, index=False)
    
    logger.info(f"Built database with {len(dfbig)} articles")
    logger.info(f"Saved to {output_csv}")
    
    return {
        "success": True,
        "output_csv": output_csv,
        "total_articles": len(dfbig),
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


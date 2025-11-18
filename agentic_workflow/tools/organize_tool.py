"""
Tool for organizing XMLs into combined directory
"""

import os
import logging
import pandas as pd
import shutil
from tqdm import tqdm

from ..tool_registry import get_registry

logger = logging.getLogger(__name__)


def organize_xmls_tool(
    corpus_dir: str,
    database_csv: str
) -> dict:
    """
    Move XMLs from journal-specific directories to combined_xmls directory
    
    Args:
        corpus_dir: Base corpus directory
        database_csv: Path to database CSV file with journal and pii columns
        
    Returns:
        Dictionary with 'success', 'moved_count', 'total_count', and 'error' keys
    """
    if not os.path.exists(corpus_dir):
        return {
            "success": False,
            "moved_count": 0,
            "total_count": 0,
            "error": f"Corpus directory not found: {corpus_dir}"
        }
    
    if not os.path.exists(database_csv):
        return {
            "success": False,
            "moved_count": 0,
            "total_count": 0,
            "error": f"Database CSV not found: {database_csv}"
        }
    
    try:
        df = pd.read_csv(database_csv)
    except Exception as e:
        return {
            "success": False,
            "moved_count": 0,
            "total_count": 0,
            "error": f"Error reading database CSV: {str(e)}"
        }
    
    if 'journal' not in df.columns or 'pii' not in df.columns:
        return {
            "success": False,
            "moved_count": 0,
            "total_count": 0,
            "error": "Database CSV must contain 'journal' and 'pii' columns"
        }
    
    combined_dir = os.path.join(corpus_dir, 'combined_xmls')
    os.makedirs(combined_dir, exist_ok=True)
    
    logger.info(f"Organizing XMLs from {corpus_dir} to {combined_dir}")
    
    # Create source and destination paths
    df['source'] = corpus_dir + '/' + df['journal'] + '/' + df['pii']
    df['destination'] = combined_dir + '/' + df['pii']
    
    total_count = len(df)
    moved_count = 0
    errors = []
    
    for i in tqdm(range(len(df)), desc="Organizing XMLs"):
        try:
            source = df['source'].iloc[i]
            dest = df['destination'].iloc[i]
            
            if os.path.exists(source):
                shutil.move(source, dest)
                moved_count += 1
            else:
                errors.append(f"Source not found: {source}")
        except Exception as e:
            errors.append(f"Error moving {df['source'].iloc[i]}: {str(e)}")
    
    if errors:
        error_log_path = os.path.join(corpus_dir, 'organize_errors.txt')
        with open(error_log_path, 'w') as f:
            for error in errors:
                f.write(f"{error}\n")
        logger.warning(f"{len(errors)} errors occurred. See {error_log_path}")
    
    logger.info(f"Moved {moved_count}/{total_count} article directories")
    
    return {
        "success": True,
        "moved_count": moved_count,
        "total_count": total_count,
        "combined_dir": combined_dir,
        "errors": len(errors),
        "error": None
    }


# Register the tool
registry = get_registry()
registry.register(
    name="organize_xmls",
    func=organize_xmls_tool,
    description="Move XML files from journal-specific directories to a combined_xmls directory. Uses database CSV to determine source and destination paths.",
    parameters={
        "type": "object",
        "properties": {
            "corpus_dir": {
                "type": "string",
                "description": "Base corpus directory containing journal subdirectories"
            },
            "database_csv": {
                "type": "string",
                "description": "Path to database CSV file with journal and pii columns"
            }
        },
        "required": ["corpus_dir", "database_csv"]
    }
)


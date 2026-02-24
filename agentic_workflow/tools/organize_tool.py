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
    
    # Sanitize journal names to match how download_tool creates directories
    # download_tool uses: jdir = '_'.join(journal_name.split())
    def sanitize_journal_name(journal_name):
        """Convert journal name to directory name format (spaces to underscores)"""
        if pd.isna(journal_name):
            return "Unknown"
        return '_'.join(str(journal_name).split())
    
    # Create source and destination paths
    # Use sanitized journal names to match directory structure created by download_tool
    df['journal_dir'] = df['journal'].apply(sanitize_journal_name)
    df['source'] = df.apply(lambda row: os.path.join(corpus_dir, row['journal_dir'], row['pii']), axis=1)
    df['destination'] = df.apply(lambda row: os.path.join(combined_dir, row['pii']), axis=1)
    
    total_count = len(df)
    moved_count = 0
    errors = []
    
    for i in tqdm(range(len(df)), desc="Organizing XMLs"):
        try:
            source = df['source'].iloc[i]
            dest = df['destination'].iloc[i]
            pii = df['pii'].iloc[i]
            
            # Check if source directory exists
            if not os.path.exists(source):
                errors.append(f"Source not found: {source}")
                continue
            
            # Check if XML file exists inside the directory
            xml_file = os.path.join(source, f"{pii}.xml")
            if not os.path.exists(xml_file):
                errors.append(f"XML file not found in {source}: {xml_file}")
                continue
            
            # Move the directory
            if os.path.exists(dest):
                # Destination already exists, skip or handle conflict
                errors.append(f"Destination already exists: {dest}")
                continue
            
            shutil.move(source, dest)
            moved_count += 1
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


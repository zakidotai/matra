"""
Tool for combining and deduplicating search results
"""

import os
import logging
import pandas as pd
from typing import List

from ..tool_registry import get_registry

logger = logging.getLogger(__name__)


def combine_and_deduplicate_tool(
    csv_paths: List[str],
    output_path: str
) -> dict:
    """
    Combine multiple CSV files and remove duplicates based on DOI
    
    Args:
        csv_paths: List of paths to CSV files to combine
        output_path: Path to save the consolidated CSV file
        
    Returns:
        Dictionary with 'success', 'output_path', 'total_count', 'unique_count', and 'error' keys
    """
    masterdf = pd.DataFrame(columns=['DOI', 'Query', 'PII', 'Title', 'Journal'])
    
    if not csv_paths:
        return {
            "success": False,
            "output_path": None,
            "total_count": 0,
            "unique_count": 0,
            "error": "No CSV files provided"
        }
    
    logger.info(f"Combining {len(csv_paths)} CSV files")
    
    loaded_count = 0
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            continue
        
        try:
            # Read CSV, handling different index column names
            tempdf = pd.read_csv(csv_path)
            
            # Handle DOI as index or column
            if 'DOI' in tempdf.columns:
                tempdf = tempdf.set_index('DOI')
            elif tempdf.index.name == 'DOI' or 'Unnamed: 0' in tempdf.columns:
                # Reset index if DOI is the index
                if tempdf.index.name == 'DOI':
                    tempdf = tempdf.reset_index()
                elif 'Unnamed: 0' in tempdf.columns:
                    # Check if Unnamed: 0 contains DOIs
                    if tempdf['Unnamed: 0'].dtype == 'object':
                        tempdf = tempdf.rename(columns={'Unnamed: 0': 'DOI'})
                        tempdf = tempdf.set_index('DOI')
            
            # Ensure required columns exist
            required_cols = ['Query', 'PII', 'Title', 'Journal']
            for col in required_cols:
                if col not in tempdf.columns:
                    tempdf[col] = 'None'
            
            masterdf = pd.concat([masterdf, tempdf], ignore_index=False)
            loaded_count += 1
            logger.info(f"Loaded {len(tempdf)} records from {csv_path}")
        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")
            continue
    
    if len(masterdf) == 0:
        return {
            "success": False,
            "output_path": None,
            "total_count": 0,
            "unique_count": 0,
            "error": "No valid data found in CSV files"
        }
    
    # Remove duplicates based on DOI
    total_count = len(masterdf)
    masterdf = masterdf.drop_duplicates(subset=['DOI'], keep='first')
    unique_count = len(masterdf)
    
    # Reset index to make DOI a column
    masterdf = masterdf.reset_index()
    if 'DOI' not in masterdf.columns and masterdf.index.name == 'DOI':
        masterdf = masterdf.reset_index()
    
    # Ensure DOI column exists
    if 'DOI' not in masterdf.columns:
        logger.warning("DOI column not found after processing")
    
    # Save consolidated CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    masterdf.to_csv(output_path, index=False)
    
    logger.info(f"Consolidated {total_count} records into {unique_count} unique DOIs")
    logger.info(f"Removed {total_count - unique_count} duplicates")
    logger.info(f"Saved to {output_path}")
    
    return {
        "success": True,
        "output_path": output_path,
        "total_count": total_count,
        "unique_count": unique_count,
        "duplicates_removed": total_count - unique_count,
        "error": None
    }


# Register the tool
registry = get_registry()
registry.register(
    name="combine_and_deduplicate",
    func=combine_and_deduplicate_tool,
    description="Combine multiple CSV files containing search results and remove duplicate entries based on DOI. Returns path to consolidated CSV file.",
    parameters={
        "type": "object",
        "properties": {
            "csv_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of paths to CSV files to combine"
            },
            "output_path": {
                "type": "string",
                "description": "Path to save the consolidated CSV file"
            }
        },
        "required": ["csv_paths", "output_path"]
    }
)


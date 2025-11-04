#!/usr/bin/env python3
"""
Material Science Table Analyzer using LLM

This script analyzes extracted tables from research papers to identify:
1. Material compositions
2. Spall strength values
3. Matches between composition and properties across tables
"""

import json
import pickle
import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MaterialProperty:
    """Data class for material property information"""
    material_id: str
    composition: Optional[Dict[str, float]] = None
    spall_strength: Optional[float] = None
    units: Optional[str] = None
    confidence: float = 0.0
    source_table: str = ""
    source_paper: str = ""

class LLMClient:
    """Client for interacting with the hosted LLM"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def analyze_table(self, table_data: List[List[str]], caption: str = "", doi: str = "") -> Dict[str, Any]:
        """Analyze a single table for material properties"""
        
        # Convert table to readable format
        table_text = self._format_table(table_data, caption)
        
        prompt = f"""
You are a materials science expert. Analyze the following table from a research paper and extract material composition and spall strength information.

Paper DOI: {doi}
Table Caption: {caption}

Table Data:
{table_text}

Please analyze this table and return a JSON response with the following structure:
{{
    "has_composition": boolean,
    "has_spall_strength": boolean,
    "materials": [
        {{
            "material_id": "unique identifier (e.g., sample name, composition)",
            "composition": {{"element": weight_percentage, ...}},
            "spall_strength": numeric_value,
            "units": "MPa or other units",
            "confidence": 0.0-1.0,
            "notes": "additional information"
        }}
    ],
    "table_type": "composition_only|property_only|both|neither",
    "common_identifiers": ["list of columns that could be used to match with other tables"]
}}

Focus on:
1. Material compositions (element percentages, alloy compositions)
2. Spall strength values (dynamic strength, impact resistance, fragmentation)
3. Sample identifiers that could link composition to properties

Return only valid JSON, no additional text.
"""

        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": "/home/mzaki4/scr16_mshiel10/mzaki4/mzaki4/cache/models--m3rg-iitd--llamat-2-chat/snapshots/2b67f6910c90d34e04ef5cb39ae0e5d7ae2e1259",
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.1,
                    "stop": ["\n\n", "Human:", "Assistant:"]
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["text"]
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning(f"Could not extract JSON from LLM response: {content[:200]}...")
                return {"error": "Invalid JSON response"}
                
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return {"error": str(e)}
    
    def match_tables(self, table1_data: Dict[str, Any], table2_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match materials between two tables based on common identifiers"""
        
        prompt = f"""
You are a materials science expert. Match materials between two tables based on common identifiers.

Table 1 Analysis:
{json.dumps(table1_data, indent=2)}

Table 2 Analysis:
{json.dumps(table2_data, indent=2)}

Please match materials between these tables and return a JSON response:
{{
    "matches": [
        {{
            "material_id": "matched identifier",
            "composition": {{"element": percentage, ...}},
            "spall_strength": numeric_value,
            "units": "MPa or other",
            "confidence": 0.0-1.0,
            "matching_criteria": "explanation of how they were matched"
        }}
    ],
    "unmatched_compositions": [
        {{
            "material_id": "identifier",
            "composition": {{"element": percentage, ...}},
            "reason": "why no match found"
        }}
    ],
    "unmatched_properties": [
        {{
            "material_id": "identifier", 
            "spall_strength": numeric_value,
            "units": "MPa or other",
            "reason": "why no match found"
        }}
    ]
}}

Return only valid JSON, no additional text.
"""

        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": "/home/mzaki4/scr16_mshiel10/mzaki4/mzaki4/cache/models--m3rg-iitd--llamat-2-chat/snapshots/2b67f6910c90d34e04ef5cb39ae0e5d7ae2e1259",
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.1,
                    "stop": ["\n\n", "Human:", "Assistant:"]
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["text"]
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning(f"Could not extract JSON from LLM response: {content[:200]}...")
                return {"error": "Invalid JSON response"}
                
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return {"error": str(e)}
    
    def _format_table(self, table_data: List[List[str]], caption: str) -> str:
        """Format table data for LLM input"""
        if not table_data:
            return "Empty table"
        
        # Convert to DataFrame for better formatting
        df = pd.DataFrame(table_data)
        
        # Create a readable table format
        table_text = f"Caption: {caption}\n\n"
        table_text += df.to_string(index=False, header=False)
        
        return table_text

class MaterialAnalyzer:
    """Main analyzer class for processing table data"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.results = []
    
    def load_tables(self, pkl_path: str) -> Dict[str, List[Dict]]:
        """Load table data from pickle file"""
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded {len(data)} papers with table data")
            return data
        except Exception as e:
            logger.error(f"Failed to load pickle file: {e}")
            return {}
    
    def analyze_paper(self, pii: str, tables: List[Dict], doi: str = "") -> Dict[str, Any]:
        """Analyze all tables in a single paper"""
        logger.info(f"Analyzing paper {pii} with {len(tables)} tables")
        
        paper_results = {
            "pii": pii,
            "doi": doi,
            "total_tables": len(tables),
            "analyzed_tables": [],
            "matched_materials": [],
            "unmatched_compositions": [],
            "unmatched_properties": []
        }
        
        table_analyses = []
        
        # Analyze each table individually
        for i, table_info in enumerate(tables):
            table_data = table_info.get("act_table", [])
            caption = table_info.get("caption", "")
            
            if not table_data:
                continue
                
            logger.info(f"Analyzing table {i+1}/{len(tables)} in paper {pii}")
            analysis = self.llm_client.analyze_table(table_data, caption, doi)
            
            if "error" not in analysis:
                analysis["table_index"] = i
                analysis["table_caption"] = caption
                table_analyses.append(analysis)
                paper_results["analyzed_tables"].append(analysis)
        
        # Try to match materials between tables
        if len(table_analyses) > 1:
            self._match_across_tables(table_analyses, paper_results)
        
        return paper_results
    
    def _match_across_tables(self, table_analyses: List[Dict], paper_results: Dict[str, Any]):
        """Match materials across different tables in the same paper"""
        
        # Find tables with compositions and properties
        composition_tables = [t for t in table_analyses if t.get("has_composition", False)]
        property_tables = [t for t in table_analyses if t.get("has_spall_strength", False)]
        
        # Try to match composition tables with property tables
        for comp_table in composition_tables:
            for prop_table in property_tables:
                if comp_table["table_index"] != prop_table["table_index"]:
                    logger.info(f"Matching tables {comp_table['table_index']} and {prop_table['table_index']}")
                    match_result = self.llm_client.match_tables(comp_table, prop_table)
                    
                    if "error" not in match_result:
                        paper_results["matched_materials"].extend(match_result.get("matches", []))
                        paper_results["unmatched_compositions"].extend(match_result.get("unmatched_compositions", []))
                        paper_results["unmatched_properties"].extend(match_result.get("unmatched_properties", []))
    
    def analyze_all_papers(self, pkl_path: str) -> List[Dict[str, Any]]:
        """Analyze all papers in the dataset"""
        data = self.load_tables(pkl_path)
        
        if not data:
            logger.error("No data loaded")
            return []
        
        all_results = []
        
        for pii, tables in data.items():
            try:
                # Extract DOI from first table if available
                doi = ""
                if tables and tables[0].get("doi"):
                    doi = tables[0]["doi"]
                
                paper_result = self.analyze_paper(pii, tables, doi)
                all_results.append(paper_result)
                
            except Exception as e:
                logger.error(f"Error analyzing paper {pii}: {e}")
                continue
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_path: str):
        """Save analysis results to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the analysis"""
        if not self.results:
            return {}
        
        total_papers = len(self.results)
        total_tables = sum(r["total_tables"] for r in self.results)
        analyzed_tables = sum(len(r["analyzed_tables"]) for r in self.results)
        
        total_matched = sum(len(r["matched_materials"]) for r in self.results)
        total_unmatched_comp = sum(len(r["unmatched_compositions"]) for r in self.results)
        total_unmatched_prop = sum(len(r["unmatched_properties"]) for r in self.results)
        
        return {
            "total_papers": total_papers,
            "total_tables": total_tables,
            "analyzed_tables": analyzed_tables,
            "matched_materials": total_matched,
            "unmatched_compositions": total_unmatched_comp,
            "unmatched_properties": total_unmatched_prop,
            "success_rate": analyzed_tables / total_tables if total_tables > 0 else 0
        }

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze material science tables using LLM")
    parser.add_argument("--pkl_path", required=True, help="Path to pii_table_dict.pkl file")
    parser.add_argument("--output", default="analysis_results.json", help="Output JSON file")
    parser.add_argument("--llm_url", default="http://localhost:8000", help="LLM server URL")
    
    args = parser.parse_args()
    
    # Initialize LLM client and analyzer
    llm_client = LLMClient(args.llm_url)
    analyzer = MaterialAnalyzer(llm_client)
    
    # Run analysis
    logger.info("Starting material analysis...")
    results = analyzer.analyze_all_papers(args.pkl_path)
    
    # Save results
    analyzer.save_results(args.output)
    
    # Print summary
    stats = analyzer.get_summary_stats()
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*50)

if __name__ == "__main__":
    main()

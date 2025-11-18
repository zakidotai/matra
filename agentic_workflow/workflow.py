"""
Main workflow orchestration for the research paper pipeline
"""

import os
import logging
from typing import List, Tuple, Optional

from .agent import Agent
from .config import Config

# Import tools to register them
from .tools import crossref_tool, deduplication_tool, download_tool, database_tool, organize_tool

logger = logging.getLogger(__name__)


class Workflow:
    """Main workflow class for orchestrating the research paper pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agent = Agent(config)
    
    def run(
        self,
        queries: List[str],
        journal_issns: List[str],
        date_ranges: List[Tuple[str, str]],
        query_name: Optional[str] = None
    ) -> dict:
        """
        Run the complete workflow
        
        Args:
            queries: List of search queries
            journal_issns: List of journal ISSN codes
            date_ranges: List of (start_date, end_date) tuples
            query_name: Optional name for the query (used for output directories)
            
        Returns:
            Dictionary with workflow results
        """
        # Generate query name if not provided
        if not query_name:
            query_name = '_'.join(queries[0].split()) if queries else 'default'
        
        # Set up output directories
        dois_output_dir = os.path.join(self.config.output_dir, f'dois_elsevier_{query_name}_')
        corpus_dir = os.path.join(self.config.output_dir, f'corpus_{query_name}')
        consolidated_csv = os.path.join(self.config.output_dir, f'consolidated_{query_name}.csv')
        database_csv = os.path.join(self.config.output_dir, f'corpus_{query_name}.csv')
        
        # Create user input for agent
        user_input = f"""Execute the research paper collection workflow with the following parameters:
- Search queries: {', '.join(queries)}
- Journal ISSNs: {', '.join(journal_issns)}
- Date ranges: {', '.join([f'{start} to {end}' for start, end in date_ranges])}
- Email: {self.config.email}
- Output directory for DOIs: {dois_output_dir}
- Output directory for corpus: {corpus_dir}
- Consolidated CSV path: {consolidated_csv}
- Database CSV path: {database_csv}
- Number of workers for search: {self.config.n_workers_search}
- Number of workers for download: {self.config.n_workers_download}
- Number of workers for database: {self.config.n_workers_database}

Please execute the following steps:
1. Search CrossRef API for all query/journal/date combinations using crossref_search tool
2. Combine all search results and remove duplicates using combine_and_deduplicate tool
3. Download articles from the consolidated CSV using download_articles tool
4. Build database from downloaded XMLs using build_database tool
5. Organize XMLs into combined directory using organize_xmls tool

Use the provided API keys: {', '.join(self.config.api_keys[:2])} (showing first 2)"""
        
        logger.info("Starting workflow execution")
        logger.info(f"Queries: {queries}")
        logger.info(f"Journals: {journal_issns}")
        logger.info(f"Date ranges: {date_ranges}")
        
        # Execute workflow using agent
        result = self.agent.execute_workflow(user_input)
        
        return result
    
    def run_direct(
        self,
        queries: List[str],
        journal_issns: List[str],
        date_ranges: List[Tuple[str, str]],
        query_name: Optional[str] = None
    ) -> dict:
        """
        Run workflow directly without agent (for testing or when agent is not available)
        
        Args:
            queries: List of search queries
            journal_issns: List of journal ISSN codes
            date_ranges: List of (start_date, end_date) tuples
            query_name: Optional name for the query
            
        Returns:
            Dictionary with workflow results
        """
        from .tools.crossref_tool import crossref_search_tool
        from .tools.deduplication_tool import combine_and_deduplicate_tool
        from .tools.download_tool import download_articles_tool
        from .tools.database_tool import build_database_tool
        from .tools.organize_tool import organize_xmls_tool
        
        # Generate query name if not provided
        if not query_name:
            query_name = '_'.join(queries[0].split()) if queries else 'default'
        
        # Set up output directories
        dois_output_dir = os.path.join(self.config.output_dir, f'dois_elsevier_{query_name}_')
        corpus_dir = os.path.join(self.config.output_dir, f'corpus_{query_name}')
        consolidated_csv = os.path.join(self.config.output_dir, f'consolidated_{query_name}.csv')
        database_csv = os.path.join(self.config.output_dir, f'corpus_{query_name}.csv')
        
        results = {
            "steps": [],
            "success": True,
            "errors": []
        }
        
        # Step 1: Crossref search
        logger.info("Step 1: Searching CrossRef API")
        search_result = crossref_search_tool(
            queries=queries,
            journal_issns=journal_issns,
            date_ranges=date_ranges,
            email=self.config.email,
            output_dir=dois_output_dir,
            n_workers=self.config.n_workers_search,
            max_rows=self.config.max_rows,
            base_url=self.config.crossref_api_base
        )
        results["steps"].append({"name": "crossref_search", "result": search_result})
        if not search_result["success"]:
            results["success"] = False
            results["errors"].append(f"Crossref search failed: {search_result.get('error')}")
            return results
        
        # Step 2: Combine and deduplicate
        logger.info("Step 2: Combining and deduplicating results")
        csv_paths = search_result.get("csv_files", [])
        if not csv_paths:
            results["success"] = False
            results["errors"].append("No CSV files from crossref search")
            return results
        
        dedup_result = combine_and_deduplicate_tool(
            csv_paths=csv_paths,
            output_path=consolidated_csv
        )
        results["steps"].append({"name": "deduplication", "result": dedup_result})
        if not dedup_result["success"]:
            results["success"] = False
            results["errors"].append(f"Deduplication failed: {dedup_result.get('error')}")
            return results
        
        # Step 3: Download articles
        logger.info("Step 3: Downloading articles")
        download_result = download_articles_tool(
            csv_path=consolidated_csv,
            api_keys=self.config.api_keys,
            output_base=corpus_dir,
            n_workers=self.config.n_workers_download,
            elsevier_api_base=self.config.elsevier_api_base
        )
        results["steps"].append({"name": "download", "result": download_result})
        if not download_result["success"]:
            results["success"] = False
            results["errors"].append(f"Download failed: {download_result.get('error')}")
            return results
        
        # Step 4: Build database
        logger.info("Step 4: Building database")
        database_result = build_database_tool(
            corpus_dir=corpus_dir,
            output_csv=database_csv,
            n_workers=self.config.n_workers_database
        )
        results["steps"].append({"name": "build_database", "result": database_result})
        if not database_result["success"]:
            results["success"] = False
            results["errors"].append(f"Database build failed: {database_result.get('error')}")
            return results
        
        # Step 5: Organize XMLs
        logger.info("Step 5: Organizing XMLs")
        organize_result = organize_xmls_tool(
            corpus_dir=corpus_dir,
            database_csv=database_csv
        )
        results["steps"].append({"name": "organize_xmls", "result": organize_result})
        if not organize_result["success"]:
            results["success"] = False
            results["errors"].append(f"Organization failed: {organize_result.get('error')}")
        
        results["summary"] = {
            "total_dois": dedup_result.get("unique_count", 0),
            "downloaded": download_result.get("successful", 0),
            "database_articles": database_result.get("total_articles", 0),
            "organized": organize_result.get("moved_count", 0)
        }
        
        return results

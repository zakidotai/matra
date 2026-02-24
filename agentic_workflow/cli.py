"""
Command-line interface for the agentic workflow
"""

import argparse
import sys
import logging
from typing import List, Tuple

from .config import Config
from .workflow import Workflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_date_range(date_str: str) -> Tuple[str, str]:
    """Parse date range string in format 'YYYY-MM:YYYY-MM'"""
    parts = date_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid date range format: {date_str}. Expected 'YYYY-MM:YYYY-MM'")
    return tuple(parts)


def parse_journals(journal_input: str) -> List[str]:
    """Parse journals from comma-separated string or file path"""
    if journal_input.endswith('.txt') or journal_input.endswith('.csv'):
        # Read from file
        try:
            with open(journal_input, 'r') as f:
                journals = [line.strip() for line in f if line.strip()]
            return journals
        except Exception as e:
            logger.error(f"Error reading journal file {journal_input}: {e}")
            sys.exit(1)
    else:
        # Comma-separated list
        return [j.strip() for j in journal_input.split(',') if j.strip()]


def parse_api_keys(keys_input: str) -> List[str]:
    """Parse API keys from comma-separated string or file path"""
    if keys_input.endswith('.txt') or keys_input.endswith('.csv'):
        # Read from file
        try:
            with open(keys_input, 'r') as f:
                keys = [line.strip() for line in f if line.strip()]
            return keys
        except Exception as e:
            logger.error(f"Error reading API keys file {keys_input}: {e}")
            sys.exit(1)
    else:
        # Comma-separated list
        return [k.strip() for k in keys_input.split(',') if k.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Agentic workflow for research paper collection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m agentic_workflow.cli --query "carbide fracture" --journals 0272-8842 --start-date 2020-01 --end-date 2024-12
  
  # Multiple queries and journals
  python -m agentic_workflow.cli -q "query1" -q "query2" --journals 0272-8842,0925-8388 --start-date 2020-01 --end-date 2024-12
  
  # Using files for journals and API keys
  python -m agentic_workflow.cli -q "query" --journals journals.txt --api-keys keys.txt --start-date 2020-01 --end-date 2024-12
  
  # Custom parallelization
  python -m agentic_workflow.cli -q "query" --journals 0272-8842 --start-date 2020-01 --end-date 2024-12 \\
    --n-workers-search 8 --n-workers-download 40 --n-workers-database 20
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-q', '--query',
        action='append',
        dest='queries',
        required=True,
        help='Search query (can specify multiple times)'
    )
    parser.add_argument(
        '--journals',
        required=True,
        help='Journal ISSNs (comma-separated) or path to file with one ISSN per line'
    )
    parser.add_argument(
        '--start-date',
        required=True,
        help='Start date in format YYYY-MM'
    )
    parser.add_argument(
        '--end-date',
        required=True,
        help='End date in format YYYY-MM'
    )
    
    # Optional arguments
    parser.add_argument(
        '--email',
        help='Email for CrossRef API (default: from config or env)'
    )
    parser.add_argument(
        '--api-keys',
        help='Elsevier API keys (comma-separated) or path to file with one key per line'
    )
    parser.add_argument(
        '--llm-provider',
        choices=['none', 'local', 'olmo', 'openai'],
        help='LLM provider (none, local, olmo, openai). Overrides LLM_PROVIDER env var.'
    )
    parser.add_argument(
        '--vllm-url',
        help='URL for vllm server (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--openai-api-key',
        help='OpenAI API key (overrides OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--openai-base-url',
        help='OpenAI base URL (default: https://api.openai.com/v1)'
    )
    parser.add_argument(
        '--openai-model',
        help='OpenAI model for tool calls (default: gpt-4o)'
    )
    parser.add_argument(
        '--openai-query-model',
        help='OpenAI model for query generation (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--output-dir',
        help='Base output directory (default: ./output)'
    )
    
    # Parallelization
    parser.add_argument(
        '--n-workers-search',
        type=int,
        help='Number of workers for crossref search (default: 4)'
    )
    parser.add_argument(
        '--n-workers-download',
        type=int,
        help='Number of workers for downloads (default: 20)'
    )
    parser.add_argument(
        '--n-workers-database',
        type=int,
        help='Number of workers for database building (default: 20)'
    )
    
    # Workflow options
    parser.add_argument(
        '--direct',
        action='store_true',
        help='Run workflow directly without agent (bypasses LLM)'
    )
    parser.add_argument(
        '--query-name',
        help='Name for the query (used for output directories, default: derived from first query)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Override config with command-line arguments
    if args.email:
        config.email = args.email
    if args.api_keys:
        config.api_keys = parse_api_keys(args.api_keys)
    if args.llm_provider:
        config.llm_provider = args.llm_provider
    if args.vllm_url:
        config.vllm_url = args.vllm_url
    if args.openai_api_key:
        config.openai_api_key = args.openai_api_key
    if args.openai_base_url:
        config.openai_base_url = args.openai_base_url
    if args.openai_model:
        config.openai_model = args.openai_model
    if args.openai_query_model:
        config.openai_query_model = args.openai_query_model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.n_workers_search:
        config.n_workers_search = args.n_workers_search
    if args.n_workers_download:
        config.n_workers_download = args.n_workers_download
    if args.n_workers_database:
        config.n_workers_database = args.n_workers_database
    
    # Validate config
    errors = config.validate()
    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Parse inputs
    queries = args.queries
    journal_issns = parse_journals(args.journals)
    date_ranges = [(args.start_date, args.end_date)]
    
    logger.info("="*60)
    logger.info("Agentic Workflow for Research Paper Pipeline")
    logger.info("="*60)
    logger.info(f"Queries: {queries}")
    logger.info(f"Journals: {journal_issns}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Workers - Search: {config.n_workers_search}, Download: {config.n_workers_download}, Database: {config.n_workers_database}")
    logger.info("="*60)
    
    # Create workflow
    workflow = Workflow(config)
    
    # Run workflow
    try:
        if args.direct:
            logger.info("Running workflow in direct mode (without agent)")
            result = workflow.run_direct(
                queries=queries,
                journal_issns=journal_issns,
                date_ranges=date_ranges,
                query_name=args.query_name
            )
        else:
            logger.info("Running workflow with agent")
            result = workflow.run(
                queries=queries,
                journal_issns=journal_issns,
                date_ranges=date_ranges,
                query_name=args.query_name
            )
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("Workflow Results")
        logger.info("="*60)
        
        if result.get("success"):
            logger.info("✓ Workflow completed successfully")
            if "summary" in result:
                summary = result["summary"]
                logger.info(f"  Total DOIs: {summary.get('total_dois', 0)}")
                logger.info(f"  Downloaded: {summary.get('downloaded', 0)}")
                logger.info(f"  Database articles: {summary.get('database_articles', 0)}")
                logger.info(f"  Organized: {summary.get('organized', 0)}")
        else:
            logger.error("✗ Workflow failed")
            if "errors" in result:
                for error in result["errors"]:
                    logger.error(f"  - {error}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

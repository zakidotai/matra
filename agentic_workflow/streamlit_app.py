"""
Streamlit web interface for the agentic workflow
"""

import streamlit as st
import logging
import os
import pandas as pd
from typing import List, Tuple

from .config import Config
from .workflow import Workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Research Paper Collection Pipeline",
    page_icon="📚",
    layout="wide"
)


def parse_date_range(start_date, end_date) -> List[Tuple[str, str]]:
    """Parse date range from date inputs"""
    if start_date and end_date:
        return [(start_date.strftime("%Y-%m"), end_date.strftime("%Y-%m"))]
    return []


def main():
    st.title("📚 Research Paper Collection Pipeline")
    st.markdown("Agentic workflow for collecting research papers from CrossRef and downloading from Elsevier")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Configuration
        st.subheader("API Settings")
        email = st.text_input(
            "Email for CrossRef API",
            value=os.getenv("CROSSREF_EMAIL", "mzaki4@jh.edu"),
            help="Required for CrossRef API access"
        )
        
        api_keys_input = st.text_area(
            "Elsevier API Keys",
            help="Enter API keys separated by commas or newlines",
            height=100
        )
        
        # vllm Configuration
        st.subheader("LLM Settings")
        vllm_url = st.text_input(
            "vLLM Server URL",
            value=os.getenv("VLLM_URL", "http://localhost:8000"),
            help="URL of the vLLM server"
        )
        
        use_agent = st.checkbox(
            "Use Agent (LLM)",
            value=False,
            help="If unchecked, runs workflow directly without LLM agent"
        )
        
        # Parallelization Settings
        st.subheader("Parallelization")
        n_workers_search = st.slider(
            "Workers for Search",
            min_value=1,
            max_value=20,
            value=4,
            help="Number of parallel workers for CrossRef search"
        )
        n_workers_download = st.slider(
            "Workers for Download",
            min_value=1,
            max_value=50,
            value=20,
            help="Number of parallel workers for article downloads"
        )
        n_workers_database = st.slider(
            "Workers for Database",
            min_value=1,
            max_value=50,
            value=20,
            help="Number of parallel workers for database building"
        )
        
        # Output Settings
        st.subheader("Output")
        output_dir = st.text_input(
            "Output Directory",
            value="./output",
            help="Base directory for output files"
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Input", "Progress", "Results"])
    
    with tab1:
        st.header("Search Parameters")
        
        # Search queries
        st.subheader("Search Queries")
        query_input = st.text_area(
            "Enter search queries",
            help="Enter one query per line",
            height=100,
            placeholder="carbide fracture\nhigh entropy ceramics"
        )
        
        # Journals
        st.subheader("Journal ISSNs")
        journal_input_method = st.radio(
            "Input method",
            ["Text input", "File upload"],
            horizontal=True
        )
        
        if journal_input_method == "Text input":
            journal_input = st.text_area(
                "Enter journal ISSNs",
                help="Enter one ISSN per line or comma-separated",
                height=100,
                placeholder="0272-8842\n0925-8388"
            )
            journal_file = None
        else:
            journal_file = st.file_uploader(
                "Upload journal file",
                type=["txt", "csv"],
                help="File with one ISSN per line"
            )
            journal_input = None
        
        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime("2020-01-01"),
                help="Start date for search"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime("2024-12-31"),
                help="End date for search"
            )
        
        # Query name
        query_name = st.text_input(
            "Query Name (optional)",
            help="Name for output directories (default: derived from first query)"
        )
        
        # Run button
        st.markdown("---")
        run_button = st.button("🚀 Run Workflow", type="primary", use_container_width=True)
    
    # Process inputs
    if run_button:
        # Validate inputs
        if not query_input or not query_input.strip():
            st.error("Please enter at least one search query")
            st.stop()
        
        if journal_input_method == "Text input" and (not journal_input or not journal_input.strip()):
            st.error("Please enter journal ISSNs")
            st.stop()
        elif journal_input_method == "File upload" and journal_file is None:
            st.error("Please upload a journal file")
            st.stop()
        
        if not email:
            st.error("Please enter an email for CrossRef API")
            st.stop()
        
        if not api_keys_input or not api_keys_input.strip():
            st.error("Please enter at least one Elsevier API key")
            st.stop()
        
        # Parse inputs
        queries = [q.strip() for q in query_input.split('\n') if q.strip()]
        
        if journal_input_method == "Text input":
            # Parse from text (handle both newline and comma separated)
            journal_lines = journal_input.replace(',', '\n').split('\n')
            journal_issns = [j.strip() for j in journal_lines if j.strip()]
        else:
            # Read from file
            journal_content = journal_file.read().decode('utf-8')
            journal_issns = [j.strip() for j in journal_content.split('\n') if j.strip()]
        
        # Parse API keys
        api_keys = [k.strip() for k in api_keys_input.replace(',', '\n').split('\n') if k.strip()]
        
        # Parse date range
        date_ranges = parse_date_range(start_date, end_date)
        if not date_ranges:
            st.error("Please select valid start and end dates")
            st.stop()
        
        # Create config
        config = Config()
        config.email = email
        config.api_keys = api_keys
        config.vllm_url = vllm_url
        config.output_dir = output_dir
        config.n_workers_search = n_workers_search
        config.n_workers_download = n_workers_download
        config.n_workers_database = n_workers_database
        
        # Validate config
        errors = config.validate()
        if errors:
            st.error("Configuration errors:")
            for error in errors:
                st.error(f"  - {error}")
            st.stop()
        
        # Store in session state
        st.session_state['workflow_config'] = config
        st.session_state['workflow_inputs'] = {
            'queries': queries,
            'journal_issns': journal_issns,
            'date_ranges': date_ranges,
            'query_name': query_name if query_name else None
        }
        st.session_state['workflow_started'] = True
        st.session_state['use_agent'] = use_agent
    
    # Progress and Results tabs
    if st.session_state.get('workflow_started', False):
        config = st.session_state['workflow_config']
        inputs = st.session_state['workflow_inputs']
        use_agent = st.session_state.get('use_agent', False)
        
        with tab2:
            st.header("Workflow Progress")
            
            if 'workflow_result' not in st.session_state:
                # Run workflow
                with st.spinner("Running workflow..."):
                    workflow = Workflow(config)
                    
                    try:
                        if use_agent:
                            result = workflow.run(
                                queries=inputs['queries'],
                                journal_issns=inputs['journal_issns'],
                                date_ranges=inputs['date_ranges'],
                                query_name=inputs['query_name']
                            )
                        else:
                            result = workflow.run_direct(
                                queries=inputs['queries'],
                                journal_issns=inputs['journal_issns'],
                                date_ranges=inputs['date_ranges'],
                                query_name=inputs['query_name']
                            )
                        
                        st.session_state['workflow_result'] = result
                    except Exception as e:
                        st.error(f"Workflow error: {str(e)}")
                        st.session_state['workflow_result'] = {
                            'success': False,
                            'error': str(e)
                        }
            
            result = st.session_state.get('workflow_result', {})
            
            if result.get('success'):
                st.success("✓ Workflow completed successfully!")
                
                # Show progress for each step
                if 'steps' in result:
                    for i, step in enumerate(result['steps'], 1):
                        step_name = step['name']
                        step_result = step['result']
                        
                        if step_result.get('success'):
                            st.success(f"Step {i}: {step_name} - Completed")
                        else:
                            st.error(f"Step {i}: {step_name} - Failed: {step_result.get('error')}")
            else:
                st.error("✗ Workflow failed")
                if 'errors' in result:
                    for error in result['errors']:
                        st.error(f"  - {error}")
        
        with tab3:
            st.header("Results")
            
            result = st.session_state.get('workflow_result', {})
            
            if result.get('success'):
                # Summary
                if 'summary' in result:
                    summary = result['summary']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total DOIs", summary.get('total_dois', 0))
                    with col2:
                        st.metric("Downloaded", summary.get('downloaded', 0))
                    with col3:
                        st.metric("Database Articles", summary.get('database_articles', 0))
                    with col4:
                        st.metric("Organized", summary.get('organized', 0))
                
                # File paths
                st.subheader("Output Files")
                query_name = inputs['query_name'] or '_'.join(inputs['queries'][0].split())
                
                output_files = {
                    "Consolidated CSV": os.path.join(config.output_dir, f'consolidated_{query_name}.csv'),
                    "Database CSV": os.path.join(config.output_dir, f'corpus_{query_name}.csv'),
                    "Corpus Directory": os.path.join(config.output_dir, f'corpus_{query_name}'),
                    "DOIs Directory": os.path.join(config.output_dir, f'dois_elsevier_{query_name}_')
                }
                
                for name, path in output_files.items():
                    if os.path.exists(path):
                        st.success(f"✓ {name}: `{path}`")
                    else:
                        st.warning(f"⚠ {name}: `{path}` (not found)")
                
                # Step details
                if 'steps' in result:
                    st.subheader("Step Details")
                    for step in result['steps']:
                        with st.expander(f"Step: {step['name']}"):
                            st.json(step['result'])
            else:
                st.error("Workflow did not complete successfully. Check the Progress tab for details.")


if __name__ == "__main__":
    main()

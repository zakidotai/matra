"""
Streamlit web interface for the agentic workflow
"""

import streamlit as st
import logging
import os
import sys
import pandas as pd
from typing import List, Tuple
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Also try current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip
    pass

# Handle imports for both package and direct script execution
try:
    from .config import Config
    from .workflow import Workflow
except ImportError:
    # Add parent directory to path for direct script execution (e.g., when run via streamlit)
    # __file__ is always defined when Streamlit runs the script
    try:
        parent_dir = str(Path(__file__).parent.parent)
    except NameError:
        # Fallback if __file__ is somehow not defined
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath('.')))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from agentic_workflow.config import Config
    from agentic_workflow.workflow import Workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="MatRA",
    page_icon="📚",
    layout="wide"
)


def parse_date_range(start_date, end_date) -> List[Tuple[str, str]]:
    """Parse date range from date inputs"""
    if start_date and end_date:
        return [(start_date.strftime("%Y-%m"), end_date.strftime("%Y-%m"))]
    return []


def filter_stop_words(queries: List[str]) -> List[str]:
    """Filter stop words from queries"""
    # Common English stop words
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
        'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
        'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
        'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
        'who', 'oil', 'its', 'now', 'find', 'down', 'day', 'did', 'get',
        'come', 'made', 'may', 'part'
    }
    
    filtered_queries = []
    for query in queries:
        # Split query into words and filter stop words
        words = query.split()
        filtered_words = [w.lower() for w in words if w.lower() not in stop_words]
        
        # Only keep query if it has meaningful words left
        if filtered_words:
            # Preserve original capitalization for important terms
            # Reconstruct query keeping original word order but removing stop words
            original_words = query.split()
            filtered_query = ' '.join([w for w in original_words if w.lower() not in stop_words])
            if filtered_query.strip():
                filtered_queries.append(filtered_query.strip())
        else:
            # If all words are stop words, keep original (edge case)
            filtered_queries.append(query.strip())
    
    return filtered_queries


def get_workflow_database_path(result: dict, config, query_name: str) -> str:
    """Get database CSV path from workflow result (tool_results or steps) if present, else derive from query_name."""
    if not result:
        return os.path.join(config.output_dir, f'corpus_{query_name}.csv')
    for tr in result.get('tool_results', []):
        if tr.get('tool') == 'build_database':
            path = (tr.get('result') or {}).get('output_csv')
            if path and os.path.exists(path):
                return path
            if path:
                return path
    for step in result.get('steps', []):
        if step.get('name') == 'build_database':
            path = (step.get('result') or {}).get('output_csv')
            if path and os.path.exists(path):
                return path
            if path:
                return path
    return os.path.join(config.output_dir, f'corpus_{query_name}.csv')


def main():
    st.title("MatRA")
    st.markdown("Materials Research Agent")
    
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
        
        # LLM Configuration
        st.subheader("LLM Settings")
        
        llm_provider = st.radio(
            "LLM Provider",
            ["No LLM (Direct)", "Local LLM (vLLM)", "OpenAI", "Olmo (OpenRouter)"],
            index=0,
            help="Select LLM provider for workflow orchestration"
        )
        
        use_agent = llm_provider != "No LLM (Direct)"
        
        if llm_provider == "Local LLM (vLLM)":
            vllm_url = st.text_input(
                "vLLM Server URL",
                value=os.getenv("VLLM_URL", "http://localhost:8000"),
                help="URL of the vLLM server"
            )
            model_name = st.text_input(
                "Model Name",
                value=os.getenv("VLLM_MODEL", "llama3.1-8b"),
                help="Model name on vLLM server"
            )
            openrouter_api_key = None
            openrouter_model = None
            openrouter_query_model = None
            openai_api_key = None
            openai_base_url = None
            openai_model = None
            openai_query_model = None
        elif llm_provider == "OpenAI":
            vllm_url = None
            model_name = None
            openrouter_api_key = None
            openrouter_model = None
            openrouter_query_model = None
            openai_api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                help="Your OpenAI API key"
            )
            openai_base_url = st.text_input(
                "OpenAI Base URL",
                value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                help="Base URL for OpenAI API (use https://api.openai.com/v1, without /responses)"
            )
            openai_model = st.text_input(
                "Tool Call Model",
                value=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                help="Model for workflow execution and tool calls"
            )
            openai_query_model = st.text_input(
                "Query Generation Model (Optional)",
                value=os.getenv("OPENAI_QUERY_MODEL", "gpt-5-nano"),
                help="Model for generating search queries (optional). Leave blank to reuse tool model."
            )
        elif llm_provider == "Olmo (OpenRouter)":
            vllm_url = None
            model_name = None
            openrouter_api_key = st.text_input(
                "OpenRouter API Key",
                value=os.getenv("OPENROUTER_API_KEY", ""),
                type="password",
                help="Your OpenRouter API key (get it from https://openrouter.ai/keys)"
            )
            
            st.markdown("**Model for Tool Calls (Workflow Execution):**")
            st.info("💡 Use instruction-tuned models (e.g., olmo-3-7b-instruct) for tool calls. Reasoning models are NOT suitable for tool calls.")
            openrouter_model = st.selectbox(
                "Tool Call Model",
                options=[
                    "meta-llama/llama-3.3-70b-instruct:free",
                ],
                index=0,
                help="Model for workflow execution and tool calls. Must be instruction-tuned, NOT reasoning."
            )
            
            st.markdown("**Model for Query Generation (Optional):**")
            st.info("💡 Reasoning models (e.g., olmo-3-32b-think) can be used for query generation only.")
            openrouter_query_model = st.selectbox(
                "Query Generation Model (Optional)",
                options=[
                    "allenai/olmo-3-32b-think:free",  # Can also use instruction model
                    "None (use tool call model)",
                ],
                index=0,
                help="Model for generating search queries. Can use reasoning models. If 'None', uses tool call model."
            )
            
            if openrouter_query_model == "None (use tool call model)":
                openrouter_query_model = None
            openai_api_key = None
            openai_base_url = None
            openai_model = None
            openai_query_model = None
        else:
            vllm_url = None
            model_name = None
            openrouter_api_key = None
            openrouter_model = None
            openrouter_query_model = None
            openai_api_key = None
            openai_base_url = None
            openai_model = None
            openai_query_model = None
        
        # Parallelization Settings
        st.subheader("Parallelization")
        n_workers_search = st.slider(
            "Workers for Search",
            min_value=1,
            max_value=8,
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Input", "Progress", "Results", "Database Search", "ML Analysis"])
    
    with tab1:
        st.header("Search Parameters")
        
        # Search queries - different input based on agent mode
        st.subheader("Search Queries")
        
        if use_agent:
            # Natural language input for agent mode
            research_description = st.text_area(
                "Describe your research area in natural language",
                help="The LLM will generate search queries from your description",
                height=150,
                placeholder="e.g., I'm researching silicon carbide ceramics and their fracture behavior. I'm interested in studies on mechanical properties, thermal conductivity, and manufacturing processes.",
                key="research_description_input"
            )
            
            # Query generation section
            if research_description and research_description.strip():
                # Ask user for number of queries
                num_queries = st.number_input(
                    "Number of queries to generate",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="How many search queries should be generated?",
                    key="num_queries_input"
                )
                
                # Generate queries button
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button("🤖 Generate Search Queries", type="primary", width='stretch'):
                        # Create config based on selected LLM provider
                        try:
                            temp_config = Config()
                            
                            # Set LLM provider based on selection
                            if llm_provider == "Local LLM (vLLM)":
                                temp_config.llm_provider = "local"
                                temp_config.vllm_url = vllm_url
                                temp_config.model_name = model_name
                            elif llm_provider == "OpenAI":
                                temp_config.llm_provider = "openai"
                                temp_config.openai_api_key = openai_api_key
                                temp_config.openai_base_url = openai_base_url
                                temp_config.openai_model = openai_model
                                temp_config.openai_query_model = openai_query_model or None
                                if not openai_api_key:
                                    st.error("Please enter OpenAI API key in the sidebar")
                                    st.stop()
                            elif llm_provider == "Olmo (OpenRouter)":
                                temp_config.llm_provider = "olmo"
                                temp_config.openrouter_api_key = openrouter_api_key
                                temp_config.openrouter_model = openrouter_model
                                temp_config.openrouter_query_model = openrouter_query_model
                                if not openrouter_api_key:
                                    st.error("Please enter OpenRouter API key in the sidebar")
                                    st.stop()
                            else:
                                st.error("Please select an LLM provider (Local LLM, OpenAI, or Olmo) to generate queries")
                                st.stop()
                            
                            temp_workflow = Workflow(temp_config)
                            
                            if temp_workflow.agent is None:
                                st.error("Agent not initialized. Please check LLM configuration.")
                                st.stop()
                            
                            with st.spinner(f"Generating {num_queries} search queries using {llm_provider}..."):
                                try:
                                    agent = temp_workflow.agent
                                    generated_queries = agent.generate_search_queries(research_description, num_queries=num_queries)
                                    # Filter stop words from generated queries
                                    generated_queries = filter_stop_words(generated_queries)
                                    st.session_state['generated_queries'] = generated_queries
                                    st.session_state['edited_queries'] = None
                                    st.session_state['last_research_desc'] = research_description
                                    # Sync text area so "Generated Queries (Editable)" shows the new list (keyed widget uses session state)
                                    st.session_state['queries_editor_input'] = '\n'.join(generated_queries)
                                    st.success(f"✅ Generated {len(generated_queries)} search queries using {llm_provider}!")
                                except Exception as e:
                                    st.error(f"Error generating queries: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        except Exception as e:
                            st.error(f"Could not initialize workflow: {e}")
                            st.error("Please configure LLM settings in the sidebar first")
                
                # Display and edit queries
                if 'generated_queries' in st.session_state or 'edited_queries' in st.session_state:
                    st.markdown("---")
                    st.subheader("Generated Queries (Editable)")
                    st.info("💡 Review and edit the queries below. Stop words have been automatically removed.")
                    
                    # Get current queries (edited if available, otherwise generated)
                    current_queries = st.session_state.get('edited_queries') or st.session_state.get('generated_queries', [])
                    
                    # Editable text area for queries
                    queries_text = '\n'.join(current_queries)
                    edited_queries_text = st.text_area(
                        "Edit queries (one per line)",
                        value=queries_text,
                        height=150,
                        help="Each line represents a separate search query. You can add, remove, or modify queries.",
                        key="queries_editor_input"
                    )
                    
                    # Parse edited queries and filter stop words
                    edited_queries = [q.strip() for q in edited_queries_text.split('\n') if q.strip()]
                    edited_queries = filter_stop_words(edited_queries)
                    
                    # Update edited queries in session state
                    if edited_queries != current_queries:
                        st.session_state['edited_queries'] = edited_queries
                    
                    # Show query count and regenerate button
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of queries", len(edited_queries))
                    with col2:
                        if st.button("🔄 Regenerate Queries", help="Generate new queries from the research description"):
                            if 'generated_queries' in st.session_state:
                                del st.session_state['generated_queries']
                            if 'edited_queries' in st.session_state:
                                del st.session_state['edited_queries']
                            if 'queries_editor_input' in st.session_state:
                                del st.session_state['queries_editor_input']
                            st.rerun()
                    
                    # Show preview
                    with st.expander("📋 Query Preview", expanded=False):
                        for i, q in enumerate(edited_queries, 1):
                            st.write(f"{i}. {q}")
                    
                    # Store queries for later use
                    query_input = '\n'.join(edited_queries)
                else:
                    query_input = None
            else:
                query_input = None
        else:
            # Manual query input for direct mode
            query_input = st.text_area(
                "Enter search queries",
                help="Enter one query per line",
                height=100,
                placeholder="carbide fracture\nhigh entropy ceramics",
                key="query_input_manual"
            )
            research_description = None
        
        # Journals
        st.subheader("Journal ISSNs")
        journal_input_method = st.radio(
            "Input method",
            ["Text input", "File upload"],
            horizontal=True,
            key="journal_input_method"
        )
        
        if journal_input_method == "Text input":
            journal_input = st.text_area(
                "Enter journal ISSNs",
                help="Enter one ISSN per line or comma-separated",
                height=100,
                placeholder="0272-8842\n0925-8388",
                key="journal_input"
            )
            journal_file = None
        else:
            journal_file = st.file_uploader(
                "Upload journal file",
                type=["txt", "csv"],
                help="File with one ISSN per line",
                key="journal_file"
            )
            journal_input = None
        
        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime("2020-01-01"),
                help="Start date for search",
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime("2024-12-31"),
                help="End date for search",
                key="end_date"
            )
        
        # Query name
        query_name = st.text_input(
            "Query Name (optional)",
            help="Name for output directories (default: derived from first query)",
            key="query_name_input"
        )
        
        # Run button
        st.markdown("---")
        run_button = st.button("🚀 Run Workflow", type="primary", width='stretch')
    
    # Process inputs
    if run_button:
        # Get values from session state
        if use_agent:
            research_description = st.session_state.get('research_description_input', '')
        else:
            research_description = None
            query_input = st.session_state.get('query_input_manual', '')
        
        journal_input_method = st.session_state.get('journal_input_method', 'Text input')
        journal_input = st.session_state.get('journal_input', '') if journal_input_method == "Text input" else None
        journal_file = st.session_state.get('journal_file', None) if journal_input_method == "File upload" else None
        start_date = st.session_state.get('start_date', pd.to_datetime("2020-01-01"))
        end_date = st.session_state.get('end_date', pd.to_datetime("2024-12-31"))
        query_name = st.session_state.get('query_name_input', '')
        
        # Validate inputs based on mode
        if use_agent:
            # Check if queries have been generated and are available
            final_queries = st.session_state.get('edited_queries') or st.session_state.get('generated_queries')
            if not final_queries or len(final_queries) == 0:
                st.error("Please generate search queries first by clicking 'Generate Search Queries' button")
                st.stop()
            queries = final_queries
        else:
            if not query_input or not query_input.strip():
                st.error("Please enter at least one search query")
                st.stop()
            queries = [q.strip() for q in query_input.split('\n') if q.strip()]
            # Filter stop words from manually entered queries too
            queries = filter_stop_words(queries)
        
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
        config.output_dir = output_dir
        config.n_workers_search = n_workers_search
        config.n_workers_download = n_workers_download
        config.n_workers_database = n_workers_database
        
        # Set LLM provider configuration
        if llm_provider == "Local LLM (vLLM)":
            config.llm_provider = "local"
            config.vllm_url = vllm_url
            config.model_name = model_name
        elif llm_provider == "OpenAI":
            config.llm_provider = "openai"
            config.openai_api_key = openai_api_key
            config.openai_base_url = openai_base_url
            config.openai_model = openai_model
            config.openai_query_model = openai_query_model or None
        elif llm_provider == "Olmo (OpenRouter)":
            config.llm_provider = "olmo"
            config.openrouter_api_key = openrouter_api_key
            config.openrouter_model = openrouter_model
            config.openrouter_query_model = openrouter_query_model
        else:
            config.llm_provider = "none"
        
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
            'research_description': research_description if use_agent else None,
            'journal_issns': journal_issns,
            'date_ranges': date_ranges,
            'query_name': query_name if query_name else None
        }
        st.session_state['workflow_started'] = True
        st.session_state['use_agent'] = use_agent
        # Clear previous workflow result (but keep queries for Progress tab)
        if 'workflow_result' in st.session_state:
            del st.session_state['workflow_result']
    
    # Progress and Results tabs
    if st.session_state.get('workflow_started', False):
        config = st.session_state['workflow_config']
        inputs = st.session_state['workflow_inputs']
        use_agent = st.session_state.get('use_agent', False)
        
        with tab2:
            st.header("Workflow Progress")
            
            if 'workflow_result' not in st.session_state:
                workflow = Workflow(config)
                
                # Get queries from session state (already generated and edited in Input tab)
                if use_agent:
                    queries_to_use = st.session_state.get('edited_queries') or st.session_state.get('generated_queries') or inputs.get('queries', [])
                else:
                    queries_to_use = inputs.get('queries', [])
                
                # Validate queries
                if not queries_to_use or len(queries_to_use) == 0:
                    st.error("⚠️ No queries available. Please go to the Input tab to generate or enter queries.")
                    st.stop()
                
                # Display queries being used
                st.info(f"📋 Using {len(queries_to_use)} search queries:")
                with st.expander("View queries", expanded=False):
                    for i, q in enumerate(queries_to_use, 1):
                        st.write(f"{i}. {q}")
                st.markdown("---")
                
                # Run workflow
                with st.spinner("Running workflow..."):
                    try:
                        if use_agent:
                            result = workflow.run(
                                queries=queries_to_use,
                                journal_issns=inputs['journal_issns'],
                                date_ranges=inputs['date_ranges'],
                                query_name=inputs['query_name']
                            )
                        else:
                            result = workflow.run_direct(
                                queries=queries_to_use,
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
                
                # Show reasoning history if available (Olmo models)
                if 'reasoning_history' in result and result['reasoning_history']:
                    st.markdown("---")
                    st.subheader("🤔 Model Reasoning Steps")
                    st.info(f"Olmo reasoning model generated {len(result['reasoning_history'])} reasoning iterations")
                    with st.expander("View Reasoning Details", expanded=False):
                        for i, reasoning_entry in enumerate(result['reasoning_history'], 1):
                            st.markdown(f"**Iteration {reasoning_entry.get('iteration', i)}:**")
                            reasoning_steps = reasoning_entry.get('reasoning_steps', [])
                            if reasoning_steps:
                                for j, step in enumerate(reasoning_steps[:5], 1):  # Show first 5 reasoning details
                                    if isinstance(step, dict):
                                        detail_type = step.get('type', 'unknown')
                                        
                                        if detail_type == 'reasoning.text':
                                            reasoning_text = step.get('text', '')
                                            if reasoning_text:
                                                st.markdown(f"**Reasoning Text {j}:**")
                                                st.text_area(
                                                    "",
                                                    value=reasoning_text[:1000] + ("..." if len(reasoning_text) > 1000 else ""),
                                                    height=150,
                                                    key=f"reasoning_text_{i}_{j}",
                                                    disabled=True,
                                                    label_visibility="collapsed"
                                                )
                                        elif detail_type == 'reasoning.summary':
                                            summary = step.get('summary', '')
                                            if summary:
                                                st.markdown(f"**Reasoning Summary {j}:**")
                                                st.info(summary)
                                        elif detail_type == 'reasoning.encrypted':
                                            st.markdown(f"**Reasoning Encrypted {j}:** (Redacted)")
                                            st.warning("This reasoning step is encrypted/redacted")
                                        else:
                                            st.json(step)
                            st.markdown("---")
                
                # Show progress for each step
                if 'steps' in result:
                    for i, step in enumerate(result['steps'], 1):
                        step_name = step['name']
                        step_result = step['result']
                        
                        if step_result.get('success'):
                            st.success(f"Step {i}: {step_name} - Completed")
                        else:
                            st.error(f"Step {i}: {step_name} - Failed: {step_result.get('error')}")
                
                # Show tool calls if available (agent mode)
                if 'tool_results' in result and result['tool_results']:
                    st.markdown("---")
                    st.subheader("Tool Execution Summary")
                    for i, tool_result in enumerate(result['tool_results'], 1):
                        tool_name = tool_result.get('tool', 'unknown')
                        success = tool_result.get('result', {}).get('success', False)
                        if success:
                            st.success(f"✓ {i}. {tool_name}")
                        else:
                            st.error(f"✗ {i}. {tool_name}: {tool_result.get('result', {}).get('error', 'Unknown error')}")
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
                    # Handle both string (from agent) and dict (from run_direct) summaries
                    if isinstance(summary, dict):
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total DOIs", summary.get('total_dois', 0))
                        with col2:
                            st.metric("Downloaded", summary.get('downloaded', 0))
                        with col3:
                            st.metric("New Articles", summary.get('new_articles', 0))
                        with col4:
                            st.metric("Total in Database", summary.get('database_articles', 0))
                        with col5:
                            st.metric("Organized", summary.get('organized', 0))
                    else:
                        # String summary from agent
                        st.info(f"**Summary:** {summary}")
                
                # File paths
                st.subheader("Output Files")
                # Get query name - handle both agent and direct mode
                # Prefer edited queries if available, otherwise use generated queries
                if use_agent:
                    queries_for_name = st.session_state.get('edited_queries') or st.session_state.get('generated_queries')
                    if queries_for_name:
                        query_name = inputs['query_name'] or '_'.join(queries_for_name[0].split())
                    else:
                        query_name = inputs.get('query_name', 'default')
                elif inputs.get('queries'):
                    query_name = inputs['query_name'] or '_'.join(inputs['queries'][0].split())
                else:
                    query_name = inputs.get('query_name', 'default')
                
                db_path_from_result = get_workflow_database_path(result, config, query_name)
                output_files = {
                    "Consolidated CSV": os.path.join(config.output_dir, f'consolidated_{query_name}.csv'),
                    "Database CSV": db_path_from_result,
                    "Corpus Directory": os.path.join(config.output_dir, f'corpus_{query_name}'),
                    "DOIs Directory": os.path.join(config.output_dir, f'dois_elsevier_{query_name}_')
                }
                
                for name, path in output_files.items():
                    if os.path.exists(path):
                        st.success(f"✓ {name}: `{path}`")
                        # Add download button for CSV files
                        if name.endswith('.csv'):
                            try:
                                with open(path, 'rb') as f:
                                    st.download_button(
                                        label=f"📥 Download {name}",
                                        data=f.read(),
                                        file_name=os.path.basename(path),
                                        mime="text/csv",
                                        key=f"download_{name}"
                                    )
                            except Exception as e:
                                st.error(f"Error reading {name}: {e}")
                    else:
                        st.warning(f"⚠ {name}: `{path}` (not found)")
                
                # Quick link to Database Search (use path from workflow result if available)
                db_csv_path = db_path_from_result
                if os.path.exists(db_csv_path):
                    st.info("💡 **Tip:** Go to the 'Database Search' tab to search and explore papers from the generated database!")
                
                # Step details
                if 'steps' in result:
                    st.subheader("Step Details")
                    for step in result['steps']:
                        with st.expander(f"Step: {step['name']}"):
                            st.json(step['result'])
            else:
                st.error("Workflow did not complete successfully. Check the Progress tab for details.")
        
        with tab4:
            st.header("📚 Database Search")
            st.markdown("Search and explore papers from generated database CSV files")
            
            # Database file selection
            st.subheader("Load Database")
            db_file_method = st.radio(
                "Select database file",
                ["From workflow output", "Upload file", "Enter path"],
                horizontal=True
            )
            
            db_df = None
            db_path = None
            
            if db_file_method == "From workflow output":
                # Try to load from most recent workflow (use path from result if build_database ran)
                if st.session_state.get('workflow_started', False):
                    config = st.session_state.get('workflow_config')
                    inputs = st.session_state.get('workflow_inputs', {})
                    result = st.session_state.get('workflow_result', {})
                    if use_agent:
                        queries_for_name = st.session_state.get('edited_queries') or st.session_state.get('generated_queries')
                        if queries_for_name:
                            query_name = inputs.get('query_name') or '_'.join(queries_for_name[0].split())
                        else:
                            query_name = inputs.get('query_name', 'default')
                    elif inputs.get('queries'):
                        query_name = inputs.get('query_name') or '_'.join(inputs.get('queries', ['default'])[0].split())
                    else:
                        query_name = inputs.get('query_name', 'default')
                    db_path = get_workflow_database_path(result, config, query_name)
                    
                    if os.path.exists(db_path):
                        try:
                            db_df = pd.read_csv(db_path)
                            st.success(f"✓ Loaded database from: `{db_path}`")
                        except Exception as e:
                            st.error(f"Error loading database: {e}")
                    else:
                        st.warning(f"Database file not found: `{db_path}`")
                        st.info("Complete the workflow (all 5 steps: search → dedupe → download → build_database → organize) to generate the database, or use **Upload file** / **Enter path** below to load an existing CSV.")
                else:
                    st.info("No workflow results available. Please run a workflow first or use 'Upload file' or 'Enter path' option")
            
            elif db_file_method == "Upload file":
                uploaded_file = st.file_uploader(
                    "Upload database CSV file",
                    type=["csv"],
                    help="Upload a CSV file with columns: title, abstracts, doi, pii, journal"
                )
                if uploaded_file is not None:
                    try:
                        db_df = pd.read_csv(uploaded_file)
                        st.success("✓ Database loaded successfully")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
            
            elif db_file_method == "Enter path":
                db_path_input = st.text_input(
                    "Enter database CSV file path",
                    help="Full path to the database CSV file"
                )
                if db_path_input:
                    if os.path.exists(db_path_input):
                        try:
                            db_df = pd.read_csv(db_path_input)
                            db_path = db_path_input
                            st.success(f"✓ Loaded database from: `{db_path}`")
                        except Exception as e:
                            st.error(f"Error loading database: {e}")
                    else:
                        st.error(f"File not found: `{db_path_input}`")
            
            # Display and search database
            if db_df is not None and len(db_df) > 0:
                st.markdown("---")
                st.subheader("Database Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Papers", len(db_df))
                with col2:
                    st.metric("Journals", db_df['journal'].nunique() if 'journal' in db_df.columns else 0)
                with col3:
                    st.metric("With DOI", db_df['doi'].notna().sum() if 'doi' in db_df.columns else 0)
                with col4:
                    st.metric("With Abstract", db_df['abstracts'].notna().sum() if 'abstracts' in db_df.columns else 0)
                
                st.markdown("---")
                st.subheader("Search Papers")
                
                # Search filters
                col1, col2 = st.columns(2)
                with col1:
                    search_query = st.text_input(
                        "Search in title/abstract",
                        help="Search for keywords in title or abstract fields",
                        placeholder="e.g., silicon carbide"
                    )
                with col2:
                    journal_filter = st.multiselect(
                        "Filter by journal",
                        options=sorted(db_df['journal'].unique().tolist()) if 'journal' in db_df.columns else [],
                        help="Select journals to filter"
                    )
                
                # Apply filters
                filtered_df = db_df.copy()
                
                if search_query:
                    search_lower = search_query.lower()
                    if 'title' in filtered_df.columns and 'abstracts' in filtered_df.columns:
                        mask = (
                            filtered_df['title'].astype(str).str.lower().str.contains(search_lower, na=False) |
                            filtered_df['abstracts'].astype(str).str.lower().str.contains(search_lower, na=False)
                        )
                        filtered_df = filtered_df[mask]
                    elif 'title' in filtered_df.columns:
                        mask = filtered_df['title'].astype(str).str.lower().str.contains(search_lower, na=False)
                        filtered_df = filtered_df[mask]
                    elif 'abstracts' in filtered_df.columns:
                        mask = filtered_df['abstracts'].astype(str).str.lower().str.contains(search_lower, na=False)
                        filtered_df = filtered_df[mask]
                
                if journal_filter and 'journal' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['journal'].isin(journal_filter)]
                
                st.info(f"Showing {len(filtered_df)} of {len(db_df)} papers")
                
                # Display results
                if len(filtered_df) > 0:
                    # View mode selection
                    view_mode = st.radio(
                        "View mode",
                        ["Card View", "Table View"],
                        horizontal=True,
                        help="Choose how to display the papers"
                    )
                    
                    # Pagination
                    items_per_page = st.slider("Items per page", 10, 100, 20, 10)
                    total_pages = (len(filtered_df) - 1) // items_per_page + 1
                    
                    if total_pages > 1:
                        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                        start_idx = (page - 1) * items_per_page
                        end_idx = start_idx + items_per_page
                        page_df = filtered_df.iloc[start_idx:end_idx]
                    else:
                        page_df = filtered_df
                    
                    if view_mode == "Table View":
                        # Display as table
                        display_columns = ['title', 'journal', 'doi', 'pii']
                        available_columns = [col for col in display_columns if col in page_df.columns]
                        
                        # Create a display dataframe with truncated titles
                        display_df = page_df[available_columns].copy()
                        if 'title' in display_df.columns:
                            display_df['title'] = display_df['title'].astype(str).apply(
                                lambda x: x[:100] + "..." if len(x) > 100 else x
                            )
                        if 'doi' in display_df.columns:
                            display_df['doi'] = display_df['doi'].apply(
                                lambda x: f"[{x}](https://doi.org/{x})" if pd.notna(x) else "N/A"
                            )
                        
                        st.dataframe(
                            display_df,
                            width='stretch',
                            height=400
                        )
                        
                        # Show full details on selection
                        st.markdown("---")
                        st.subheader("Paper Details")
                        selected_idx = st.selectbox(
                            "Select a paper to view details",
                            options=page_df.index.tolist(),
                            format_func=lambda x: page_df.loc[x, 'title'] if pd.notna(page_df.loc[x, 'title']) else f"Paper {x}"
                        )
                        
                        if selected_idx is not None:
                            row = page_df.loc[selected_idx]
                            st.markdown(f"### {row.get('title', 'No title')}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if pd.notna(row.get('journal')):
                                    st.markdown(f"**Journal:** {row['journal']}")
                            with col2:
                                if pd.notna(row.get('doi')):
                                    st.markdown(f"**DOI:** [{row['doi']}](https://doi.org/{row['doi']})")
                            with col3:
                                if pd.notna(row.get('pii')):
                                    st.markdown(f"**PII:** {row['pii']}")
                            
                            st.markdown("---")
                            if pd.notna(row.get('abstracts')):
                                st.markdown("**Abstract:**")
                                st.markdown(row['abstracts'])
                            else:
                                st.info("No abstract available")
                    else:
                        # Card view (existing implementation)
                        # Display papers
                        for idx, row in page_df.iterrows():
                            title_display = row.get('title', 'No title')
                            if pd.notna(title_display) and len(str(title_display)) > 100:
                                title_display = str(title_display)[:100] + "..."
                            
                            with st.expander(f"📄 {title_display}"):
                                # Title
                                if pd.notna(row.get('title')):
                                    st.markdown(f"### {row['title']}")
                                
                                # Metadata in columns
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if pd.notna(row.get('journal')):
                                        st.markdown(f"**Journal:** {row['journal']}")
                                with col2:
                                    if pd.notna(row.get('doi')):
                                        st.markdown(f"**DOI:** [{row['doi']}](https://doi.org/{row['doi']})")
                                with col3:
                                    if pd.notna(row.get('pii')):
                                        st.markdown(f"**PII:** {row['pii']}")
                                
                                # Abstract
                                st.markdown("---")
                                if pd.notna(row.get('abstracts')):
                                    st.markdown("**Abstract:**")
                                    abstract_text = str(row['abstracts'])
                                    if len(abstract_text) > 1000:
                                        st.text_area(
                                            "",
                                            value=abstract_text,
                                            height=200,
                                            key=f"abstract_{idx}",
                                            disabled=True,
                                            label_visibility="collapsed"
                                        )
                                    else:
                                        st.markdown(abstract_text)
                                else:
                                    st.info("No abstract available")
                    
                    # Download filtered results
                    st.markdown("---")
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Results (CSV)",
                        data=csv_data,
                        file_name=f"filtered_database_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No papers match your search criteria")
            elif db_df is not None and len(db_df) == 0:
                st.warning("Database file is empty")
            else:
                st.info("👆 Please load a database file to begin searching")
    
    with tab5:
        st.header("🤖 ML Analysis")
        st.markdown("Perform machine learning workflow for materials property prediction with Bayesian neural networks and SHAP explainability")
        
        # Input Section
        st.subheader("Input Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            data_path = st.text_input(
                "Data File Path",
                value="/Users/mohdzaki/Documents/github/cerie/bmg_polak_morgan/cursor_carbides/density_clean.csv",
                help="Full path to the CSV data file"
            )
        with col2:
            ml_output_dir = st.text_input(
                "Output Directory",
                value=os.path.join(output_dir, "ml_output"),
                help="Directory for saving models and outputs"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_column = st.text_input(
                "Target Column",
                value="Density",
                help="Name of the target column (e.g. Density, YM, Young's modulus)"
            )
        with col2:
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Train/test split ratio"
            )
        with col3:
            random_state = st.number_input(
                "Random State",
                min_value=0,
                value=94,
                help="Random seed for reproducibility"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            n_trials = st.number_input(
                "Optuna Trials",
                min_value=10,
                max_value=500,
                value=100,
                help="Number of hyperparameter optimization trials"
            )
        with col2:
            n_samples_uncertainty = st.number_input(
                "Uncertainty Samples",
                min_value=50,
                max_value=1000,
                value=300,
                help="Number of samples for Bayesian uncertainty estimation"
            )
        
        # Target distribution: load data and show histogram in dashboard
        st.subheader("Target distribution")
        st.markdown("Load the data file to view the distribution of the target property. Use this to set an optional value range below.")
        if st.button("📊 Load and show histogram", key="ml_load_histogram"):
            if not data_path or not os.path.exists(data_path):
                st.warning("Enter a valid data file path that exists.")
            else:
                try:
                    try:
                        from agentic_workflow.tools.ml_tool import _resolve_target_column
                    except ImportError:
                        from .tools.ml_tool import _resolve_target_column
                    import matplotlib.pyplot as plt
                    df_hist = pd.read_csv(data_path)
                    index_like = [c for c in df_hist.columns if str(c).startswith("Unnamed") or (isinstance(c, str) and c.strip() == "")]
                    if index_like:
                        df_hist = df_hist.drop(columns=index_like)
                    resolved = _resolve_target_column(df_hist, target_column)
                    vals = df_hist[resolved].dropna()
                    vals = vals[vals > 0.01]
                    if len(vals) == 0:
                        st.warning("No valid target values > 0.01 in the file.")
                    else:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(vals, bins=min(50, max(10, len(vals) // 5)), edgecolor="gray", alpha=0.8)
                        ax.set_xlabel(resolved)
                        ax.set_ylabel("Count")
                        ax.set_title(f"Distribution of {resolved} (before cleaning)\nmin={vals.min():.3g}, max={vals.max():.3g}, n={len(vals)}")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(f"Resolved column: **{resolved}** · min = {vals.min():.3g}, max = {vals.max():.3g}, n = {len(vals)}. Set optional range below and run the workflow.")
                except Exception as e:
                    st.error(f"Could not load histogram: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Optional target range (set after inspecting histogram)
        st.markdown("**Optional: limit target value range** (set after inspecting histogram above)")
        use_target_range = st.checkbox("Apply target range filter", value=False, help="Filter rows to keep only target values within min/max before cleaning and training")
        target_min_input = None
        target_max_input = None
        if use_target_range:
            cr1, cr2 = st.columns(2)
            with cr1:
                target_min_input = st.number_input("Target minimum", value=0.0, format="%g", key="ml_target_min", help="Rows with target < this are excluded")
            with cr2:
                target_max_input = st.number_input("Target maximum", value=1000.0, format="%g", key="ml_target_max", help="Rows with target > this are excluded")
        
        # Hyperparameter Configuration
        st.markdown("---")
        with st.expander("🔧 Hyperparameter Search Space Configuration", expanded=False):
            st.markdown("Configure the search space for Optuna hyperparameter optimization")
            
            col1, col2 = st.columns(2)
            with col1:
                hidden_layers_min = st.number_input(
                    "Hidden Layers (Min)",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Minimum number of hidden layers"
                )
            with col2:
                hidden_layers_max = st.number_input(
                    "Hidden Layers (Max)",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="Maximum number of hidden layers"
                )
            
            st.markdown("**Hidden Units Options:**")
            hidden_units_options = st.multiselect(
                "Select hidden unit sizes to search",
                options=[32, 64, 128, 256, 512, 1024],
                default=[32, 64, 128, 256, 512],
                help="Select which hidden unit sizes to include in the search"
            )
            if not hidden_units_options:
                hidden_units_options = [32, 64, 128, 256, 512]
            
            col1, col2 = st.columns(2)
            with col1:
                lr_min = st.number_input(
                    "Learning Rate (Min)",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=1e-4,
                    format="%.2e",
                    help="Minimum learning rate"
                )
            with col2:
                lr_max = st.number_input(
                    "Learning Rate (Max)",
                    min_value=1e-4,
                    max_value=1e-1,
                    value=5e-2,
                    format="%.2e",
                    help="Maximum learning rate"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                weight_decay_min = st.number_input(
                    "Weight Decay (Min)",
                    min_value=1e-10,
                    max_value=1e-3,
                    value=1e-8,
                    format="%.2e",
                    help="Minimum weight decay"
                )
            with col2:
                weight_decay_max = st.number_input(
                    "Weight Decay (Max)",
                    min_value=1e-8,
                    max_value=1e-1,
                    value=1e-2,
                    format="%.2e",
                    help="Maximum weight decay"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                beta_min = st.number_input(
                    "Beta (KL Divergence) (Min)",
                    min_value=1e-10,
                    max_value=1e-3,
                    value=1e-8,
                    format="%.2e",
                    help="Minimum beta for KL divergence regularization"
                )
            with col2:
                beta_max = st.number_input(
                    "Beta (KL Divergence) (Max)",
                    min_value=1e-8,
                    max_value=1e-2,
                    value=1e-4,
                    format="%.2e",
                    help="Maximum beta for KL divergence regularization"
                )
            
            st.markdown("**Batch Size Options:**")
            batch_size_options = st.multiselect(
                "Select batch sizes to search",
                options=[16, 32, 64, 128, 256],
                default=[32, 64, 128],
                help="Select which batch sizes to include in the search"
            )
            if not batch_size_options:
                batch_size_options = [32, 64, 128]
            
            col1, col2 = st.columns(2)
            with col1:
                epochs_min = st.number_input(
                    "Epochs (Min)",
                    min_value=10,
                    max_value=500,
                    value=80,
                    help="Minimum number of training epochs"
                )
            with col2:
                epochs_max = st.number_input(
                    "Epochs (Max)",
                    min_value=50,
                    max_value=1000,
                    value=200,
                    help="Maximum number of training epochs"
                )
        
        # Run button
        st.markdown("---")
        run_ml_button = st.button("🚀 Run ML Workflow", type="primary", width='stretch')
        
        # Process ML workflow
        if run_ml_button:
            if not data_path or not os.path.exists(data_path):
                st.error(f"Data file not found: {data_path}")
                st.stop()
            
            # Create config for workflow
            ml_config = Config()
            ml_config.output_dir = output_dir
            
            # Create workflow instance
            workflow = Workflow(ml_config)
            
            # Prepare hyperparameter config
            hyperparams = {
                "hidden_layers_min": hidden_layers_min,
                "hidden_layers_max": hidden_layers_max,
                "hidden_units_options": hidden_units_options,
                "lr_min": lr_min,
                "lr_max": lr_max,
                "weight_decay_min": weight_decay_min,
                "weight_decay_max": weight_decay_max,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "batch_size_options": batch_size_options,
                "epochs_min": epochs_min,
                "epochs_max": epochs_max
            }
            
            # Optional target range (from dashboard inputs)
            target_min_arg = target_min_input if use_target_range and target_min_input is not None else None
            target_max_arg = target_max_input if use_target_range and target_max_input is not None else None
            
            # Run ML workflow
            with st.spinner("Running ML workflow... This may take several minutes."):
                try:
                    result = workflow.run_ml_workflow_direct(
                        data_path=data_path,
                        output_dir=ml_output_dir,
                        target_column=target_column,
                        test_size=test_size,
                        random_state=random_state,
                        n_trials=n_trials,
                        n_samples_uncertainty=n_samples_uncertainty,
                        hyperparams=hyperparams,
                        target_min=target_min_arg,
                        target_max=target_max_arg,
                    )
                    
                    # Store result in session state
                    st.session_state['ml_result'] = result
                    
                except Exception as e:
                    st.error(f"ML workflow error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        # Display Results
        if 'ml_result' in st.session_state:
            result = st.session_state['ml_result']
            
            if result.get('success'):
                st.success("✅ ML Workflow completed successfully!")
                
                # Data Info
                st.subheader("Data Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Data Shape", f"{result.get('data_shape', 'N/A')}")
                with col2:
                    st.metric("Output Directory", ml_output_dir)
                
                # Metrics
                st.subheader("Model Performance Metrics")
                
                metrics = result.get('metrics', {})
                
                # Tree Model Metrics
                if 'tree_model' in metrics:
                    st.markdown("#### Tree Model (Random Forest)")
                    tree_metrics = metrics['tree_model']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Train R²", f"{tree_metrics.get('train_r2', 0):.4f}")
                    with col2:
                        st.metric("Test R²", f"{tree_metrics.get('test_r2', 0):.4f}")
                    with col3:
                        st.metric("Train RMSE", f"{tree_metrics.get('train_rmse', 0):.4f}")
                    with col4:
                        st.metric("Test RMSE", f"{tree_metrics.get('test_rmse', 0):.4f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train MAE", f"{tree_metrics.get('train_mae', 0):.4f}")
                    with col2:
                        st.metric("Test MAE", f"{tree_metrics.get('test_mae', 0):.4f}")
                
                # Bayesian Model Metrics
                if 'bayesian_model' in metrics:
                    st.markdown("#### Bayesian Neural Network Model")
                    bayesian_metrics = metrics['bayesian_model']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Train R²", f"{bayesian_metrics.get('train_r2', 0):.4f}")
                    with col2:
                        st.metric("Test R²", f"{bayesian_metrics.get('test_r2', 0):.4f}")
                    with col3:
                        st.metric("Train RMSE", f"{bayesian_metrics.get('train_rmse', 0):.4f}")
                    with col4:
                        st.metric("Test RMSE", f"{bayesian_metrics.get('test_rmse', 0):.4f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train MAE", f"{bayesian_metrics.get('train_mae', 0):.4f}")
                    with col2:
                        st.metric("Test MAE", f"{bayesian_metrics.get('test_mae', 0):.4f}")
                    
                    # Best hyperparameters
                    if 'best_params' in bayesian_metrics:
                        with st.expander("Best Hyperparameters"):
                            st.json(bayesian_metrics['best_params'])
                
                # Visualizations
                st.subheader("Visualizations")
                viz_paths = result.get('visualization_paths', {})
                
                # Scatter plots
                if 'scatter_actual_predicted' in viz_paths and os.path.exists(viz_paths['scatter_actual_predicted']):
                    st.markdown("##### Actual vs Predicted (Bayesian Model)")
                    st.image(viz_paths['scatter_actual_predicted'])
                
                if 'scatter_errorbars' in viz_paths and os.path.exists(viz_paths['scatter_errorbars']):
                    st.markdown("##### Test Data with Uncertainty Bars")
                    st.image(viz_paths['scatter_errorbars'])
                
                # SHAP plots for tree model
                st.markdown("#### SHAP Explainability - Tree Model")
                col1, col2 = st.columns(2)
                with col1:
                    if 'shap_bar_tree' in viz_paths and os.path.exists(viz_paths['shap_bar_tree']):
                        st.markdown("##### Feature Importance (Bar Plot)")
                        st.image(viz_paths['shap_bar_tree'])
                with col2:
                    if 'shap_scatter_tree' in viz_paths and os.path.exists(viz_paths['shap_scatter_tree']):
                        st.markdown("##### SHAP Values (Beeswarm Plot)")
                        st.image(viz_paths['shap_scatter_tree'])
                
                # SHAP plots for Bayesian model
                st.markdown("#### SHAP Explainability - Bayesian Model")
                col1, col2 = st.columns(2)
                with col1:
                    if 'shap_bar_bayesian' in viz_paths and os.path.exists(viz_paths['shap_bar_bayesian']):
                        st.markdown("##### Feature Importance (Bar Plot - Unscaled)")
                        st.image(viz_paths['shap_bar_bayesian'])
                with col2:
                    if 'shap_scatter_bayesian' in viz_paths and os.path.exists(viz_paths['shap_scatter_bayesian']):
                        st.markdown("##### SHAP Values (Beeswarm Plot - Unscaled)")
                        st.image(viz_paths['shap_scatter_bayesian'])
                
                # Model Files
                st.subheader("Saved Model Files")
                model_paths = result.get('model_paths', {})
                
                for model_name, model_path in model_paths.items():
                    if os.path.exists(model_path):
                        st.success(f"✓ {model_name}: `{model_path}`")
                        # Add download button
                        try:
                            with open(model_path, 'rb') as f:
                                file_extension = os.path.splitext(model_path)[1]
                                mime_type = "application/octet-stream"
                                if file_extension == ".pkl":
                                    mime_type = "application/octet-stream"
                                elif file_extension == ".pt":
                                    mime_type = "application/octet-stream"
                                
                                st.download_button(
                                    label=f"📥 Download {model_name}",
                                    data=f.read(),
                                    file_name=os.path.basename(model_path),
                                    mime=mime_type,
                                    key=f"download_{model_name}"
                                )
                        except Exception as e:
                            st.warning(f"Could not read {model_name}: {e}")
                    else:
                        st.warning(f"⚠ {model_name}: `{model_path}` (not found)")
            else:
                st.error("❌ ML Workflow failed")
                if 'error' in result:
                    st.error(f"Error: {result['error']}")


if __name__ == "__main__":
    main()

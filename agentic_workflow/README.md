# Agentic Workflow for Research Paper Pipeline

An agentic workflow system that automates the research paper collection pipeline using LLM (llama3-8b-instruct via vllm) to orchestrate tool calls for crossref search, deduplication, downloading, database creation, and XML organization.

## Features

- **Agentic Orchestration**: Uses LLM to intelligently orchestrate workflow steps
- **Parallel Processing**: Configurable parallelization for all CPU-intensive tasks
- **Multiple Interfaces**: Both CLI and Streamlit web interface
- **Modular Tools**: Separate tools for each task (search, deduplication, download, database, organization)

## Installation

Ensure you have the required dependencies:

```bash
pip install pandas requests beautifulsoup4 joblib p-tqdm tqdm streamlit
```

## Usage

### CLI Interface

Basic usage:

```bash
python -m agentic_workflow.cli \
  --query "carbide fracture" \
  --journals 0272-8842 \
  --start-date 2020-01 \
  --end-date 2024-12 \
  --email your.email@example.com \
  --api-keys "key1,key2" \
  --direct
```

With multiple queries and journals:

```bash
python -m agentic_workflow.cli \
  -q "query1" -q "query2" \
  --journals 0272-8842,0925-8388 \
  --start-date 2020-01 \
  --end-date 2024-12 \
  --email your.email@example.com \
  --api-keys "key1,key2" \
  --n-workers-search 8 \
  --n-workers-download 40 \
  --n-workers-database 20 \
  --direct
```

Using files for journals and API keys:

```bash
python -m agentic_workflow.cli \
  -q "query" \
  --journals journals.txt \
  --api-keys keys.txt \
  --start-date 2020-01 \
  --end-date 2024-12 \
  --email your.email@example.com \
  --direct
```

### Streamlit Interface

Launch the web interface:

```bash
streamlit run agentic_workflow/streamlit_app.py
```

Then open your browser to the URL shown (typically http://localhost:8501).

### Agent Mode (with LLM)

To use the agent with LLM orchestration, ensure you have a vllm server running:

```bash
# Start vllm server (example)
python -m vllm.entrypoints.openai.api_server \
  --model llama3-8b-instruct \
  --port 8000
```

Then run without the `--direct` flag:

```bash
python -m agentic_workflow.cli \
  --query "carbide fracture" \
  --journals 0272-8842 \
  --start-date 2020-01 \
  --end-date 2024-12 \
  --email your.email@example.com \
  --api-keys "key1,key2" \
  --vllm-url http://localhost:8000
```

## Configuration

Configuration can be set via:
1. Environment variables
2. Command-line arguments
3. Streamlit UI inputs

### Environment Variables

- `CROSSREF_EMAIL`: Email for CrossRef API
- `ELSEVIER_API_KEYS`: Comma-separated API keys
- `VLLM_URL`: URL of vLLM server (default: http://localhost:8000)
- `VLLM_MODEL`: Model name (default: llama3-8b-instruct)
- `N_WORKERS_SEARCH`: Workers for search (default: 4)
- `N_WORKERS_DOWNLOAD`: Workers for download (default: 20)
- `N_WORKERS_DATABASE`: Workers for database (default: 20)
- `OUTPUT_DIR`: Base output directory (default: ./output)

## Workflow Steps

1. **Crossref Search**: Search CrossRef API for papers matching queries, journals, and date ranges
2. **Deduplication**: Combine all search results and remove duplicates based on DOI
3. **Download**: Download article XMLs from DOIs using Elsevier API
4. **Database Building**: Extract metadata (DOI, title, abstract, PII) from downloaded XMLs
5. **Organization**: Move XMLs from journal-specific directories to combined_xmls directory

## Output Structure

```
output/
├── dois_elsevier_{query_name}_/
│   └── {Journal_Name}/
│       └── {ISSN}_{start}_{end}.csv
├── consolidated_{query_name}.csv
├── corpus_{query_name}/
│   ├── {Journal_Name}/
│   │   └── {PII}/
│   │       └── {PII}.xml
│   └── combined_xmls/
│       └── {PII}/
│           └── {PII}.xml
└── corpus_{query_name}.csv
```

## Tools

Each tool is registered and can be called by the agent:

- `crossref_search`: Search CrossRef API
- `combine_and_deduplicate`: Combine and deduplicate CSV results
- `download_articles`: Download articles from DOIs
- `build_database`: Build database from XMLs
- `organize_xmls`: Organize XMLs into combined directory

## License

See the main project license.


# Public Crossref Dashboard

A simple Streamlit dashboard that:
- accepts a natural-language research query,
- generates 20 editable search queries using OpenAI,
- accepts an optional ISSN list (or falls back to defaults),
- runs a **serial** Crossref search across query/ISSN/month combinations (one task at a time),
- logs DOIs found per (query, journal, month) with a progress bar,
- outputs a unified dataframe and CSV download.

## Folder Contents

- `app.py`: Streamlit UI and workflow orchestration
- `query_generator.py`: OpenAI query generation
- `crossref_client.py`: serial Crossref search client
- `defaults.py`: default ISSNs and constants
- `requirements.txt`: deploy dependencies

## Local Run

1. Open terminal in this folder:

```bash
cd public_crossref_dashboard
```

2. Create virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Start app:

```bash
streamlit run app.py
```

4. In the app:
- enter OpenAI API key,
- choose model and optionally edit system prompt,
- enter research query + date range,
- generate/edit query list,
- optionally provide ISSNs,
- run search and download CSV.

## Streamlit Community Cloud Deployment

1. Push this folder to your GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app and point to:
   - repository: your repo
   - file path: `public_crossref_dashboard/app.py`
4. Ensure dependencies are detected from:
   - `public_crossref_dashboard/requirements.txt`
5. Deploy.

You can enter OpenAI API key directly in the app UI at runtime. It is not persisted to disk by this app.

## Notes

- Crossref performance depends on query breadth and ISSN count.
- The search runs serially (one task at a time) to mirror the proven notebook loop
  in `example/example_notebook_share.ipynb` and avoid Crossref 429 (rate limit) errors.
- Tune `Pause between tasks` (default 5s) and `Pause between paginated requests`
  (default 1.2s) in the sidebar if you still see rate limits.
- The app de-duplicates by `(doi, query_used)`.

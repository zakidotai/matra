"""Defaults and reusable constants for the Public Crossref Dashboard."""

DEFAULT_ISSNS = [
    "0272-8842",
    "0925-8388",
    "0955-2219",
    "1879-0496",
    "0263-4368",
    "0257-8972",
    "1005-0302",
    "1385-8947",
    "1359-6462",
    "1359-8368",
    "1359-6454",
    "0921-5093",
    "0167-577X",
    "1875-5372",
    "0254-0584",
    "0010-938X",
    "0042-207X",
    "0026-0657",
    "0022-3115",
    "2468-0230",
    "0927-0256",
    "0043-1648",
    "1879-0062",
    "1873-1961",
    "0301-679X",
    "1873-5002",
    "1873-3891",
]

DEFAULT_TARGET_QUERY_COUNT = 20
DEFAULT_CROSSREF_ROWS = 1000
DEFAULT_CROSSREF_TIMEOUT_SECONDS = 60
DEFAULT_MAX_PAGES_PER_TASK = 20
DEFAULT_MAX_WORKERS = 12

DEFAULT_OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o",
]

DEFAULT_SYSTEM_PROMPT = (
    "You generate search-oriented keywords and short phrases for scientific literature "
    "discovery. Return exactly 20 unique items. Mix single words and 2-3 word phrases. "
    "Ensure terms are relevant, diverse, and practical for Crossref search."
)

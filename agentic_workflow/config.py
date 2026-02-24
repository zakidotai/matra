"""
Configuration management for the agentic workflow
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for the agentic workflow"""
    
    # API Configuration
    email: str = field(default_factory=lambda: os.getenv("CROSSREF_EMAIL", "mzaki4@jh.edu"))
    api_keys: List[str] = field(default_factory=lambda: os.getenv("ELSEVIER_API_KEYS", "").split(",") if os.getenv("ELSEVIER_API_KEYS") else [])
    vllm_url: str = field(default_factory=lambda: os.getenv("VLLM_URL", "http://localhost:8000"))
    model_name: str = field(default_factory=lambda: os.getenv("VLLM_MODEL", "llama3.1-8b"))
    
    # LLM Provider Configuration
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "local"))  # "local", "none", "olmo", "openai"
    openrouter_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    # Model for tool calls (workflow execution) - should be instruction-tuned, not reasoning
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "allenai/olmo-3-7b-instruct:free"))
    # Model for query generation - can use reasoning models
    openrouter_query_model: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_QUERY_MODEL", "allenai/olmo-3-32b-think:free"))
    # OpenAI provider configuration
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5-nano"))
    openai_query_model: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_QUERY_MODEL", "gpt-5-nano"))
    
    # Parallelization
    n_workers_search: int = field(default_factory=lambda: int(os.getenv("N_WORKERS_SEARCH", "4")))
    n_workers_download: int = field(default_factory=lambda: int(os.getenv("N_WORKERS_DOWNLOAD", "20")))
    n_workers_database: int = field(default_factory=lambda: int(os.getenv("N_WORKERS_DATABASE", "20")))
    
    # Output
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "./output"))
    
    # API Settings
    crossref_api_base: str = "https://api.crossref.org/works?query="
    elsevier_api_base: str = "https://api.elsevier.com/content/article/doi/"
    max_rows: int = 1000
    sleep_interval: int = 5
    request_timeout: int = 100
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.email:
            errors.append("Email is required for CrossRef API")
        
        if not self.api_keys or not any(self.api_keys):
            errors.append("At least one Elsevier API key is required")
        
        if self.llm_provider == "olmo" and not self.openrouter_api_key:
            errors.append("OpenRouter API key is required when using Olmo model")
        
        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OpenAI API key is required when using OpenAI provider")
        
        if self.n_workers_search < 1:
            errors.append("n_workers_search must be at least 1")
        
        if self.n_workers_download < 1:
            errors.append("n_workers_download must be at least 1")
        
        if self.n_workers_database < 1:
            errors.append("n_workers_database must be at least 1")
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "email": self.email,
            "api_keys": self.api_keys,
            "vllm_url": self.vllm_url,
            "model_name": self.model_name,
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "openai_query_model": self.openai_query_model,
            "n_workers_search": self.n_workers_search,
            "n_workers_download": self.n_workers_download,
            "n_workers_database": self.n_workers_database,
            "output_dir": self.output_dir,
        }


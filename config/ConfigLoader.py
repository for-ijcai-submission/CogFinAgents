from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import yaml
import os
import re

# ---------- Pydantic Models ----------

class LLMConfig(BaseModel):
    api_base: str = ""
    api_key: str = ""
    model: str = "openai/gpt-4o"
    temperature: float = 0.3
    cache: bool = False

class ExecutionConfig(BaseModel):
    parallel_agents: bool = False
    max_workers: int = 8
    debug_llm_io: bool = False

class TrumpSocialConfig(BaseModel):
    enabled: bool = False
    path: str = "./data/global/processed_trump.csv"

class AblationConfig(BaseModel):
    memory_disabled: bool = False  # Ablation 1: Disable memory
    remove_social_news: bool = False  # Ablation 2: Remove social and news data

class DataDropRule(BaseModel):
    """
    General data masking rule: disable specified inputs by agent_type × asset × date range × data modules (drop_modules).
    When enabled=true, the rule's date_start/date_end and assets will override main experiment configuration.
    """
    name: str
    enabled: bool = False
    agents: List[str]  # agent_type list, e.g. ["D"] or ["C","D"]
    assets: List[str]  # asset list, e.g. ["GLD"] or ["BTC","DOGE"]
    date_start: str    # YYYY-MM-DD (unified with main experiment config field name)
    date_end: str      # YYYY-MM-DD (unified with main experiment config field name)
    drop_modules: List[str]  # e.g. ["macro_data","on_chain_data","company_data","social_data","news_data"]

    @field_validator('date_start', 'date_end')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"Invalid date format in DataDropRule: {v}. Expected YYYY-MM-DD")
        return v


class SimulationConfig(BaseModel):
    assets: List[str]
    date_start: str
    date_end: str
    assets_root: str = "./assets"
    result_dir: str = "./simulation_results"
    agent_type_count: Dict[str, int]
    llm: LLMConfig
    execution: Optional[ExecutionConfig] = ExecutionConfig()
    trump_social: Optional[TrumpSocialConfig] = TrumpSocialConfig()
    ablation: Optional[AblationConfig] = AblationConfig()
    # General data masking rules (both case study & ablation can be configured here)
    data_drop_rules: Optional[List[DataDropRule]] = None

    @field_validator('date_start', 'date_end')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD")
        return v

    @field_validator('agent_type_count')
    @classmethod
    def check_agent_types(cls, v: Dict[str, int]) -> Dict[str, int]:
        allowed = {'A','B','C','D'}
        for k, val in v.items():
            if k not in allowed:
                raise ValueError(f"Unknown agent type in agent_type_count: {k}")
            if val < 0:
                raise ValueError(f"Agent count must be non-negative: {k}={val}")
        return v

class MemorySpec(BaseModel):
    window_days: int = Field(ge=1, le=365)
    sort_key: str  # abs_return | abs_pnl | abs_error | recency
    top_k: int = Field(ge=1, le=50)

    @field_validator('sort_key')
    @classmethod
    def validate_sort_key(cls, v: str) -> str:
        allowed = {'abs_return','abs_pnl','abs_error','recency'}
        if v not in allowed:
            raise ValueError(f"Unsupported memory.sort_key: {v}")
        return v

class AgentProfile(BaseModel):
    name: str
    info_access: str  # basic | advanced
    memory: MemorySpec
    profile_text: str = ""  # Core behavioral description for LLM prompt

    @field_validator('info_access')
    @classmethod
    def validate_access(cls, v: str) -> str:
        if v not in {'basic','advanced'}:
            raise ValueError(f"info_access must be 'basic' or 'advanced', got {v}")
        return v

class AgentsConfig(BaseModel):
    profiles: Dict[str, AgentProfile]

    @field_validator('profiles')
    @classmethod
    def validate_profiles(cls, v: Dict[str, 'AgentProfile']) -> Dict[str, 'AgentProfile']:
        required = {'A','B','C','D'}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"Missing agent profiles: {missing}")
        return v

class DataSchemaEntry(BaseModel):
    basic: List[str]
    advanced: List[str]

class SocialTruncationConfig(BaseModel):
    max_rows: int = 20

class NewsTruncationConfig(BaseModel):
    max_rows: int = 15
    sentiment_distribution: Dict[str, int] = Field(default_factory=lambda: {"Positive": 6, "Negative": 6, "Neutral": 3})

class MacroTruncationConfig(BaseModel):
    max_text_chars: int = 5000

class TruncationConfig(BaseModel):
    social_data: Optional[SocialTruncationConfig] = None
    news_data: Optional[NewsTruncationConfig] = None
    macro_data: Optional[MacroTruncationConfig] = None

class DataSchemas(BaseModel):
    schemas: Dict[str, DataSchemaEntry]  # market_data, macro_data, ...
    applicability: Dict[str, List[str]]  # company_assets, onchain_assets
    truncation: Optional[TruncationConfig] = None

    @field_validator('schemas')
    @classmethod
    def validate_schemas(cls, v: Dict[str, 'DataSchemaEntry']) -> Dict[str, 'DataSchemaEntry']:
        for k, entry in v.items():
            if not isinstance(entry.basic, list) or not isinstance(entry.advanced, list):
                raise ValueError(f"{k} must define 'basic' and 'advanced' lists.")
        return v

    @field_validator('applicability')
    @classmethod
    def validate_applicability(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        required = {'company_assets','onchain_assets'}
        if not required.issubset(v.keys()):
            raise ValueError(f"applicability must include keys: {required}")
        return v

# ---------- Loader Functions ----------

def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_simulation_config(path: str) -> SimulationConfig:
    raw = _load_yaml(path)
    return SimulationConfig(**raw)

def load_agents_config(path: str) -> AgentsConfig:
    raw = _load_yaml(path)
    return AgentsConfig(**raw)

def load_data_schema(path: str) -> DataSchemas:
    raw = _load_yaml(path)
    return DataSchemas(**raw)
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", PROJECT_ROOT / "config.yaml"))


class OutputSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dir: str = "output/information_agent"
    json_file: str = "latest_entities.json"
    markdown_file: str = "latest_entities.md"


class ContentSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_content_chars: int = 12000


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = "duckduckgo"
    timeout_seconds: int = 5


class FetchSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_workers: int = 6
    timeout_seconds: int = 30
    cache_dir: str = ".cache/fetch_url"
    cache_ttl_seconds: int = 86400
    retrieval_passage_limit: int = 5
    retrieval_neighbor_radius: int = 1


class VertexCacheSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dir: str = ".cache/vertex_cache"
    ttl_seconds: int = 3600


class StandardModeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    search_result_limit: int = 8
    fetch_limit: int = 6


class DeepModeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    search_result_limit: int = 12
    fetch_limit: int = 3


class LightningModeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_search_queries: int = 3
    search_result_limit: int = 5
    search_timeout_seconds: int = 3
    fetch_url_limit: int = 3


class ModeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    standard: StandardModeSettings = Field(default_factory=StandardModeSettings)
    deep: DeepModeSettings = Field(default_factory=DeepModeSettings)
    lightning: LightningModeSettings = Field(default_factory=LightningModeSettings)


class RecursiveResearchSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_rounds: int = 2
    max_fetch_urls: int = 2
    standard_search_result_limit: int = 3
    deep_search_result_limit: int = 5


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    openai_default: str = "gpt-4.1-mini"
    vertex_default: str = "gemini-2.5-flash"
    intent: str | None = None


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    output: OutputSettings = Field(default_factory=OutputSettings)
    content: ContentSettings = Field(default_factory=ContentSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    vertex_cache: VertexCacheSettings = Field(default_factory=VertexCacheSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    modes: ModeSettings = Field(default_factory=ModeSettings)
    recursive_research: RecursiveResearchSettings = Field(
        default_factory=RecursiveResearchSettings
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _set_if_not_none(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _load_raw_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    output = dict(raw.get("output") or {})
    search = dict(raw.get("search") or {})
    fetch = dict(raw.get("fetch") or {})
    vertex_cache = dict(raw.get("vertex_cache") or {})
    models = dict(raw.get("models") or {})
    content = dict(raw.get("content") or {})
    modes = dict(raw.get("modes") or {})
    standard_mode = dict(modes.get("standard") or {})
    deep_mode = dict(modes.get("deep") or {})
    lightning_mode = dict(modes.get("lightning") or {})
    recursive_research = dict(raw.get("recursive_research") or {})

    _set_if_not_none(output, "dir", os.getenv("OUTPUT_DIR"))
    _set_if_not_none(output, "json_file", os.getenv("OUTPUT_JSON_FILE"))
    _set_if_not_none(output, "markdown_file", os.getenv("OUTPUT_MARKDOWN_FILE"))

    _set_if_not_none(content, "max_content_chars", _env_int("MAX_CONTENT_CHARS"))

    _set_if_not_none(search, "provider", os.getenv("SEARCH_PROVIDER"))
    _set_if_not_none(search, "timeout_seconds", _env_int("SEARCH_TIMEOUT_SECONDS"))

    _set_if_not_none(fetch, "max_workers", _env_int("FETCH_URL_MAX_WORKERS"))
    _set_if_not_none(fetch, "timeout_seconds", _env_int("FETCH_TIMEOUT_SECONDS"))
    _set_if_not_none(fetch, "cache_dir", os.getenv("FETCH_CACHE_DIR"))
    _set_if_not_none(fetch, "cache_ttl_seconds", _env_int("FETCH_CACHE_TTL_SECONDS"))
    _set_if_not_none(
        fetch,
        "retrieval_passage_limit",
        _env_int("FETCH_RETRIEVAL_PASSAGE_LIMIT"),
    )
    _set_if_not_none(
        fetch,
        "retrieval_neighbor_radius",
        _env_int("FETCH_RETRIEVAL_NEIGHBOR_RADIUS"),
    )

    _set_if_not_none(vertex_cache, "dir", os.getenv("VERTEX_CACHE_DIR"))
    _set_if_not_none(
        vertex_cache,
        "ttl_seconds",
        _env_int("VERTEX_CACHE_TTL_SECONDS"),
    )

    _set_if_not_none(models, "openai_default", os.getenv("DEFAULT_OPENAI_MODEL"))
    _set_if_not_none(models, "vertex_default", os.getenv("DEFAULT_VERTEX_MODEL"))
    _set_if_not_none(models, "intent", os.getenv("DEFAULT_INTENT_MODEL"))

    _set_if_not_none(
        standard_mode, "search_result_limit", _env_int("STANDARD_SEARCH_RESULT_LIMIT")
    )
    _set_if_not_none(standard_mode, "fetch_limit", _env_int("STANDARD_FETCH_LIMIT"))

    _set_if_not_none(deep_mode, "search_result_limit", _env_int("DEEP_SEARCH_RESULT_LIMIT"))
    _set_if_not_none(deep_mode, "fetch_limit", _env_int("DEEP_FETCH_LIMIT"))

    _set_if_not_none(
        lightning_mode, "max_search_queries", _env_int("LIGHTNING_MAX_SEARCH_QUERIES")
    )
    _set_if_not_none(
        lightning_mode, "search_result_limit", _env_int("LIGHTNING_SEARCH_RESULT_LIMIT")
    )
    _set_if_not_none(
        lightning_mode,
        "search_timeout_seconds",
        _env_int("LIGHTNING_SEARCH_TIMEOUT_SECONDS"),
    )
    _set_if_not_none(
        lightning_mode, "fetch_url_limit", _env_int("LIGHTNING_FETCH_URL_LIMIT")
    )

    _set_if_not_none(
        recursive_research, "max_rounds", _env_int("RECURSIVE_RESEARCH_MAX_ROUNDS")
    )
    _set_if_not_none(
        recursive_research,
        "max_fetch_urls",
        _env_int("RECURSIVE_RESEARCH_MAX_FETCH_URLS"),
    )
    _set_if_not_none(
        recursive_research,
        "standard_search_result_limit",
        _env_int("RECURSIVE_RESEARCH_STANDARD_SEARCH_RESULT_LIMIT"),
    )
    _set_if_not_none(
        recursive_research,
        "deep_search_result_limit",
        _env_int("RECURSIVE_RESEARCH_DEEP_SEARCH_RESULT_LIMIT"),
    )

    return {
        **raw,
        "output": output,
        "content": content,
        "search": search,
        "fetch": fetch,
        "vertex_cache": vertex_cache,
        "models": models,
        "modes": {
            **modes,
            "standard": standard_mode,
            "deep": deep_mode,
            "lightning": lightning_mode,
        },
        "recursive_research": recursive_research,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.model_validate(_apply_env_overrides(_load_raw_config()))


def env_flag(name: str, default: bool = False) -> bool:
    return _env_flag(name, default=default)


def use_vertex_ai() -> bool:
    return env_flag("GOOGLE_GENAI_USE_VERTEXAI", default=False)


def get_api_key() -> str:
    return os.getenv("PROVIDER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def get_base_url() -> str | None:
    return os.getenv("PROVIDER_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def get_model_name() -> str:
    if use_vertex_ai():
        default_model = get_settings().models.vertex_default
    else:
        default_model = get_settings().models.openai_default

    return (
        os.getenv("INFORMATION_AGENT_MODEL")
        or os.getenv("PROVIDER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or default_model
    )


def get_intent_model_name() -> str:
    return (
        os.getenv("INTENT_MODEL")
        or get_settings().models.intent
        or get_model_name()
    )


def get_vertex_project() -> str:
    return os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()


def get_vertex_location() -> str:
    return os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"


def get_output_dir() -> Path:
    return PROJECT_ROOT / get_settings().output.dir


def get_json_output_path() -> Path:
    return get_output_dir() / get_settings().output.json_file


def get_markdown_output_path() -> Path:
    return get_output_dir() / get_settings().output.markdown_file


def get_max_content_chars() -> int:
    return get_settings().content.max_content_chars


def get_search_provider() -> str:
    return get_settings().search.provider


def get_search_timeout_seconds() -> int:
    return get_settings().search.timeout_seconds


def get_search_result_limit(deep_research: bool) -> int:
    if deep_research:
        return get_settings().modes.deep.search_result_limit
    return get_settings().modes.standard.search_result_limit


def get_fetch_limit(deep_research: bool) -> int:
    if deep_research:
        return get_settings().modes.deep.fetch_limit
    return get_settings().modes.standard.fetch_limit


def get_fetch_url_max_workers() -> int:
    return get_settings().fetch.max_workers


def get_fetch_timeout_seconds() -> int:
    return get_settings().fetch.timeout_seconds


def get_fetch_cache_dir() -> Path:
    configured_path = Path(get_settings().fetch.cache_dir)
    if configured_path.is_absolute():
        return configured_path
    return PROJECT_ROOT / configured_path


def get_fetch_cache_ttl_seconds() -> int:
    return get_settings().fetch.cache_ttl_seconds


def get_vertex_cache_dir() -> Path:
    configured_path = Path(get_settings().vertex_cache.dir)
    if configured_path.is_absolute():
        return configured_path
    return PROJECT_ROOT / configured_path


def get_vertex_cache_ttl_seconds() -> int:
    return get_settings().vertex_cache.ttl_seconds


def get_fetch_retrieval_passage_limit() -> int:
    return get_settings().fetch.retrieval_passage_limit


def get_fetch_retrieval_neighbor_radius() -> int:
    return get_settings().fetch.retrieval_neighbor_radius


def get_recursive_research_max_rounds() -> int:
    return get_settings().recursive_research.max_rounds


def get_recursive_research_max_fetch_urls() -> int:
    return get_settings().recursive_research.max_fetch_urls


def get_recursive_search_result_limit(deep_research: bool) -> int:
    if deep_research:
        return get_settings().recursive_research.deep_search_result_limit
    return get_settings().recursive_research.standard_search_result_limit


def get_lightning_max_search_queries() -> int:
    return get_settings().modes.lightning.max_search_queries


def get_lightning_fetch_url_limit() -> int:
    return get_settings().modes.lightning.fetch_url_limit


def get_lightning_search_timeout_seconds() -> int:
    return get_settings().modes.lightning.search_timeout_seconds


def get_lightning_search_result_limit() -> int:
    return get_settings().modes.lightning.search_result_limit


MAX_CONTENT_CHARS = get_max_content_chars()
RECURSIVE_RESEARCH_MAX_ROUNDS = get_recursive_research_max_rounds()
RECURSIVE_RESEARCH_MAX_FETCH_URLS = get_recursive_research_max_fetch_urls()
OUTPUT_DIR = get_output_dir()
JSON_OUTPUT_PATH = get_json_output_path()
MARKDOWN_OUTPUT_PATH = get_markdown_output_path()
FETCH_CACHE_DIR = get_fetch_cache_dir()

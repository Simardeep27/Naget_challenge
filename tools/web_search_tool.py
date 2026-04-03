import os
import logging

import requests
from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pydantic import Field

from utils.config import get_search_provider, get_search_timeout_seconds

logger = logging.getLogger(__name__)


class SearchToolInput(BaseIOSchema):
    """Search tool input schema."""

    value: str = Field(..., description="Query to search")


class SearchToolOutput(BaseIOSchema):
    """Search tool output schema."""

    results: list[dict[str, object]] = Field(
        default_factory=list,
        description="Returned web search results",
    )
    error: str | None = Field(
        default=None,
        description="Handled provider or network error, if the search failed",
    )


class SearchToolConfig(BaseToolConfig):
    """Tool configuration options."""

    provider: str = Field(
        default_factory=get_search_provider
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SEARCH_TOOL_API_KEY")
        or os.getenv("BRAVE_API_KEY")
    )
    max_results: int = Field(default=10, description="Maximum results per search")
    timeout_seconds: int = Field(
        default_factory=get_search_timeout_seconds,
        description="Timeout per search request in seconds",
    )


class SearchTool(BaseTool[SearchToolInput, SearchToolOutput]):
    """Search the web using DuckDuckGo by default, or Brave when configured."""

    def __init__(self, config: SearchToolConfig = SearchToolConfig()):
        super().__init__(config)
        self.provider = config.provider.strip().lower()
        self.api_key = config.api_key
        self.max_results = config.max_results
        self.timeout_seconds = config.timeout_seconds

    def _search_brave(self, query: str) -> list[dict[str, object]]:
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY is required when SEARCH_PROVIDER=brave")

        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            params={"q": query, "count": self.max_results},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("web", {}).get("results", [])

        return [
            {
                "title": item.get("title", ""),
                "href": item.get("url", ""),
                "body": item.get("description", ""),
            }
            for item in raw_results
            if item.get("url") and item.get("title")
        ]

    def _search_duckduckgo(self, query: str) -> list[dict[str, object]]:
        with DDGS(timeout=self.timeout_seconds) as ddgs:
            raw_results = list(ddgs.text(query, max_results=self.max_results))

        return [
            {
                "title": item.get("title", ""),
                "href": item.get("href", "") or item.get("url", ""),
                "body": item.get("body", "") or item.get("snippet", ""),
            }
            for item in raw_results
            if (item.get("href") or item.get("url")) and item.get("title")
        ]

    def run(self, params: SearchToolInput) -> SearchToolOutput:
        query = params.value.strip()
        if not query:
            return SearchToolOutput(results=[])

        try:
            if self.provider == "brave":
                results = self._search_brave(query)
            else:
                results = self._search_duckduckgo(query)
        except (DDGSException, requests.RequestException, ValueError) as exc:
            error_message = (
                f"{self.provider} search failed for query {query!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            logger.warning(error_message)
            return SearchToolOutput(results=[], error=error_message)
        except Exception as exc:
            error_message = (
                f"{self.provider} search failed unexpectedly for query {query!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            logger.warning(error_message)
            return SearchToolOutput(results=[], error=error_message)

        return SearchToolOutput(results=results)

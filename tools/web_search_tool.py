import os

import requests
from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import Field

load_dotenv()


class SearchToolInput(BaseIOSchema):
    """Search tool input schema."""

    value: str = Field(..., description="Query to search")


class SearchToolOutput(BaseIOSchema):
    """Search tool output schema."""

    results: list[dict[str, object]] = Field(
        default_factory=list,
        description="Returned web search results",
    )


class SearchToolConfig(BaseToolConfig):
    """Tool configuration options."""

    provider: str = Field(
        default_factory=lambda: os.getenv("SEARCH_PROVIDER", "duckduckgo")
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SEARCH_TOOL_API_KEY")
        or os.getenv("BRAVE_API_KEY")
    )
    max_results: int = Field(default=10, description="Maximum results per search")


class SearchTool(BaseTool[SearchToolInput, SearchToolOutput]):
    """Search the web using DuckDuckGo by default, or Brave when configured."""

    def __init__(self, config: SearchToolConfig = SearchToolConfig()):
        super().__init__(config)
        self.provider = config.provider.strip().lower()
        self.api_key = config.api_key
        self.max_results = config.max_results

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
            timeout=30,
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
        with DDGS() as ddgs:
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

        if self.provider == "brave":
            results = self._search_brave(query)
        else:
            results = self._search_duckduckgo(query)

        return SearchToolOutput(results=results)

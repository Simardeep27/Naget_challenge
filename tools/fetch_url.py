import os
from concurrent.futures import ThreadPoolExecutor

import trafilatura
from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig
from pydantic import Field


class FetchURLToolInputSchema(BaseIOSchema):
    """Input payload for the fetch URL tool."""

    value: list[str] = Field(
        ..., description="List of URLs to fetch and extract text from"
    )


class FetchURLToolOutputSchema(BaseIOSchema):
    """Output payload for the fetch URL tool."""

    result: list[str] = Field(
        default_factory=list,
        description=(
            "Extracted markdown-like content per URL. "
            "Contains an error string if fetch or extraction failed."
        ),
    )


class FetchURLTool(BaseTool[FetchURLToolInputSchema, FetchURLToolOutputSchema]):
    """Fetch a URL and extract readable content using trafilatura."""

    def __init__(self, config: BaseToolConfig = BaseToolConfig()):
        super().__init__(config)
        self.max_workers = max(
            1,
            int(os.getenv("FETCH_URL_MAX_WORKERS", "6")),
        )

    def _fetch_one(self, url: str) -> str:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return f"[FETCH FAILED] Could not download: {url}"

            text = trafilatura.extract(
                downloaded,
                url=url,
                favor_recall=True,
                include_tables=True,
                include_images=False,
                output_format="markdown",
                deduplicate=True,
            )
            return text or f"[EXTRACT FAILED] No content extracted from: {url}"
        except Exception as exc:
            return f"[ERROR] {url} :: {type(exc).__name__}: {exc}"

    def run(self, params: FetchURLToolInputSchema) -> FetchURLToolOutputSchema:
        urls = params.value
        if not urls:
            return FetchURLToolOutputSchema(result=[])

        worker_count = min(self.max_workers, len(urls))
        if worker_count == 1:
            results = [self._fetch_one(url) for url in urls]
            return FetchURLToolOutputSchema(result=results)

        # executor.map preserves input order, so downstream URL/result zipping still works
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(self._fetch_one, urls))

        return FetchURLToolOutputSchema(result=results)

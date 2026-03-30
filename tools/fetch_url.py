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

    def run(self, params: FetchURLToolInputSchema) -> FetchURLToolOutputSchema:
        results: list[str] = []

        for url in params.value:
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded is None:
                    results.append(f"[FETCH FAILED] Could not download: {url}")
                    continue

                text = trafilatura.extract(
                    downloaded,
                    url=url,
                    favor_recall=True,
                    include_tables=True,
                    include_images=False,
                    output_format="markdown",
                    deduplicate=True,
                )
                results.append(
                    text or f"[EXTRACT FAILED] No content extracted from: {url}"
                )
            except Exception as exc:
                results.append(f"[ERROR] {url} :: {type(exc).__name__}: {exc}")

        return FetchURLToolOutputSchema(result=results)

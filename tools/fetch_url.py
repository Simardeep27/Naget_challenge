from concurrent.futures import ThreadPoolExecutor

import requests
import trafilatura
from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig
from pydantic import Field
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from utils.config import get_fetch_timeout_seconds, get_fetch_url_max_workers


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

    ERROR_PREFIXES = ("[FETCH FAILED]", "[EXTRACT FAILED]", "[ERROR]")

    def __init__(self, config: BaseToolConfig = BaseToolConfig()):
        super().__init__(config)
        self.max_workers = max(1, get_fetch_url_max_workers())
        self.timeout_seconds = max(1, get_fetch_timeout_seconds())

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        no_retry = Retry(
            total=0,
            connect=0,
            read=0,
            redirect=0,
            status=0,
            other=0,
            raise_on_redirect=False,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=no_retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; InfoAgent/1.0; "
                    "+https://github.com/openai/codex)"
                )
            }
        )
        return session

    def _fetch_one(self, url: str) -> str:
        try:
            with self._build_session() as session:
                response = session.get(
                    url,
                    timeout=self.timeout_seconds,
                    allow_redirects=True,
                )
                response.raise_for_status()
                if not response.content:
                    return f"[FETCH FAILED] Empty response body: {url}"

                if not response.encoding:
                    response.encoding = response.apparent_encoding or "utf-8"

                downloaded = response.text

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
        except requests.RequestException as exc:
            return f"[ERROR] {url} :: {type(exc).__name__}: {exc}"
        except Exception as exc:
            return f"[ERROR] {url} :: {type(exc).__name__}: {exc}"

    @classmethod
    def is_error_result(cls, content: str) -> bool:
        return content.startswith(cls.ERROR_PREFIXES)

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

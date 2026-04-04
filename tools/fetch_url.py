import hashlib
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
import trafilatura
from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig
from pydantic import Field
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from schema import PreviewFetchRecord
from utils.config import (
    get_fetch_cache_dir,
    get_fetch_cache_ttl_seconds,
    get_fetch_retrieval_neighbor_radius,
    get_fetch_retrieval_passage_limit,
    get_fetch_timeout_seconds,
    get_fetch_url_max_workers,
)
from utils.text_utils import compact_text

WORD_PATTERN = re.compile(r"[a-z0-9]{2,}")
QUOTED_PHRASE_PATTERN = re.compile(r'"([^"]+)"')
TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
META_DESCRIPTION_PATTERN = re.compile(
    r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
HEADING_PATTERN = re.compile(
    r"<h([1-3])[^>]*>(.*?)</h\1>",
    re.IGNORECASE | re.DOTALL,
)
JSON_LD_PATTERN = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
ANCHOR_PATTERN = re.compile(r"<a[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
TOC_HINT_TERMS = (
    "about",
    "booking",
    "compare",
    "comparison",
    "docs",
    "documentation",
    "benchmark",
    "benchmarks",
    "faq",
    "fixtures",
    "hours",
    "live",
    "location",
    "locations",
    "menu",
    "pricing",
    "profile",
    "reservation",
    "review",
    "schedule",
    "score",
    "scores",
    "standings",
    "status",
    "api",
)


class FetchURLToolInputSchema(BaseIOSchema):
    """Input payload for the fetch URL tool."""

    value: list[str] = Field(
        ..., description="List of URLs to fetch and extract text from"
    )
    focus_query: str | None = Field(
        default=None,
        description=(
            "Optional retrieval query used to select the most relevant passages "
            "from the cached page extraction."
        ),
    )
    full_document: bool = Field(
        default=False,
        description=(
            "Whether to return the full extracted document instead of selected passages."
        ),
    )


class FetchURLToolOutputSchema(BaseIOSchema):
    """Output payload for the fetch URL tool."""

    result: list[str] = Field(
        default_factory=list,
        description=(
            "Relevant extracted markdown-like passages per URL. "
            "Contains an error string if fetch or extraction failed."
        ),
    )


class PreviewURLToolInputSchema(BaseIOSchema):
    """Input payload for the lightweight preview stage."""

    value: list[str] = Field(
        ..., description="List of URLs to preview before a full fetch"
    )
    max_bytes: int = Field(
        default=24576,
        description="Maximum number of response bytes to inspect per page",
    )


class PreviewURLToolOutputSchema(BaseIOSchema):
    """Preview results used to decide whether a page deserves a full fetch."""

    result: list[PreviewFetchRecord] = Field(
        default_factory=list,
        description="Preview records for each requested URL",
    )


class FetchURLTool(BaseTool[FetchURLToolInputSchema, FetchURLToolOutputSchema]):
    """Fetch a URL, cache the extraction, and retrieve the most relevant passages."""

    ERROR_PREFIXES = ("[FETCH FAILED]", "[EXTRACT FAILED]", "[ERROR]")
    DEFAULT_CACHE_TTL_SECONDS = 86400
    MAX_BLOCK_CHARS = 900
    MAX_SELECTED_PASSAGES_FALLBACK = 3
    DEFAULT_PREVIEW_MAX_BYTES = 24576
    PREVIEW_TIMEOUT_SECONDS = 10

    def __init__(self, config: BaseToolConfig = BaseToolConfig()):
        super().__init__(config)
        self.max_workers = max(1, get_fetch_url_max_workers())
        self.timeout_seconds = max(1, get_fetch_timeout_seconds())
        self.cache_dir = get_fetch_cache_dir()
        self.cache_ttl_seconds = max(0, get_fetch_cache_ttl_seconds())
        self.retrieval_passage_limit = max(1, get_fetch_retrieval_passage_limit())
        self.retrieval_neighbor_radius = max(0, get_fetch_retrieval_neighbor_radius())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def _normalize_url(self, url: str) -> str:
        parts = urlsplit(url.strip())
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

    def _cache_path_for_url(self, url: str) -> Path:
        normalized_url = self._normalize_url(url)
        digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load_cache_entry(
        self,
        url: str,
        *,
        allow_stale: bool = False,
    ) -> dict[str, Any] | None:
        cache_path = self._cache_path_for_url(url)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        fetched_at = int(payload.get("fetched_at") or 0)
        is_fresh = (
            self.cache_ttl_seconds == 0
            or (time.time() - fetched_at) <= self.cache_ttl_seconds
        )
        if is_fresh or allow_stale:
            return payload
        return None

    def _write_cache_entry(self, url: str, payload: dict[str, Any]) -> None:
        for candidate_url in {
            self._normalize_url(url),
            self._normalize_url(str(payload.get("final_url") or url)),
        }:
            cache_path = self._cache_path_for_url(candidate_url)
            try:
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                continue

    def _extract_heading(self, block: str) -> str | None:
        stripped = block.strip()
        if not stripped:
            return None

        markdown_heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if markdown_heading:
            return compact_text(markdown_heading.group(1))
        return None

    def _split_long_block(self, block: str) -> list[str]:
        if len(block) <= self.MAX_BLOCK_CHARS:
            return [block]

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", block)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.MAX_BLOCK_CHARS:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = sentence
                continue

            for index in range(0, len(sentence), self.MAX_BLOCK_CHARS):
                piece = sentence[index : index + self.MAX_BLOCK_CHARS].strip()
                if piece:
                    chunks.append(piece)
            current = ""

        if current:
            chunks.append(current)
        return chunks or [block]

    def _extract_passages(self, extracted_text: str) -> list[dict[str, str | None]]:
        blocks = [
            block.strip()
            for block in re.split(r"\n\s*\n", extracted_text)
            if block and block.strip()
        ]
        passages: list[dict[str, str | None]] = []
        current_heading: str | None = None

        for block in blocks:
            heading = self._extract_heading(block)
            if heading and len(block.splitlines()) == 1:
                current_heading = heading
                continue

            normalized_block = "\n".join(
                line.rstrip() for line in block.splitlines() if line.strip()
            ).strip()
            if not normalized_block:
                continue

            for chunk in self._split_long_block(normalized_block):
                passages.append(
                    {
                        "heading": current_heading,
                        "text": chunk,
                    }
                )

        if passages:
            return passages

        fallback_text = extracted_text.strip()
        if not fallback_text:
            return []

        return [{"heading": None, "text": chunk} for chunk in self._split_long_block(fallback_text)]

    def _tokenize(self, value: str) -> list[str]:
        return WORD_PATTERN.findall(value.lower())

    def _score_passage(
        self,
        passage: dict[str, str | None],
        *,
        focus_query: str,
    ) -> float:
        heading = compact_text(passage.get("heading"))
        text = compact_text(passage.get("text"))
        if not text:
            return 0.0

        focus_terms = set(self._tokenize(focus_query))
        if not focus_terms:
            return 0.0

        text_tokens = set(self._tokenize(text))
        heading_tokens = set(self._tokenize(heading or ""))
        combined_normalized = " ".join(self._tokenize(f"{heading or ''} {text}"))

        score = 0.0
        score += float(len(focus_terms & text_tokens))
        score += float(2 * len(focus_terms & heading_tokens))

        for phrase in QUOTED_PHRASE_PATTERN.findall(focus_query):
            normalized_phrase = " ".join(self._tokenize(phrase))
            if not normalized_phrase:
                continue
            if normalized_phrase in combined_normalized:
                score += 6.0
            if heading and normalized_phrase in " ".join(self._tokenize(heading)):
                score += 4.0

        return score

    def _select_relevant_passages(
        self,
        passages: list[dict[str, str | None]],
        *,
        focus_query: str | None,
    ) -> list[dict[str, str | None]]:
        if not passages:
            return []

        if not compact_text(focus_query):
            return passages[: self.retrieval_passage_limit]

        scored_passages = [
            (self._score_passage(passage, focus_query=focus_query or ""), index)
            for index, passage in enumerate(passages)
        ]
        positive_matches = [
            (score, index) for score, index in scored_passages if score > 0
        ]

        if positive_matches:
            positive_matches.sort(key=lambda item: (-item[0], item[1]))
            seed_indices = [
                index
                for _, index in positive_matches[: self.retrieval_passage_limit]
            ]
        else:
            seed_indices = list(
                range(
                    min(
                        self.MAX_SELECTED_PASSAGES_FALLBACK,
                        self.retrieval_passage_limit,
                        len(passages),
                    )
                )
            )

        selected_indices: list[int] = []
        max_selected = min(
            len(passages),
            self.retrieval_passage_limit + (2 * self.retrieval_neighbor_radius),
        )

        for seed_index in seed_indices:
            start = max(0, seed_index - self.retrieval_neighbor_radius)
            end = min(len(passages), seed_index + self.retrieval_neighbor_radius + 1)
            for index in range(start, end):
                if index not in selected_indices:
                    selected_indices.append(index)
                if len(selected_indices) >= max_selected:
                    break
            if len(selected_indices) >= max_selected:
                break

        selected_indices.sort()
        return [passages[index] for index in selected_indices]

    def _render_passages(
        self,
        passages: list[dict[str, str | None]],
    ) -> str:
        rendered_sections: list[str] = []
        seen_sections: set[tuple[str | None, str]] = set()

        for passage in passages:
            heading = compact_text(passage.get("heading"))
            text = compact_text(passage.get("text"))
            if not text:
                continue

            section_key = (heading, text)
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)

            if heading:
                rendered_sections.append(f"## {heading}\n{text}")
            else:
                rendered_sections.append(text)

        return "\n\n".join(rendered_sections)

    def _build_payload(
        self,
        *,
        requested_url: str,
        final_url: str,
        extracted_text: str,
    ) -> dict[str, Any]:
        return {
            "requested_url": self._normalize_url(requested_url),
            "final_url": self._normalize_url(final_url),
            "fetched_at": int(time.time()),
            "content": extracted_text,
            "passages": self._extract_passages(extracted_text),
        }

    def _download_and_extract(self, url: str) -> dict[str, Any] | str:
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

            extracted_text = trafilatura.extract(
                downloaded,
                url=url,
                favor_recall=True,
                include_tables=True,
                include_images=False,
                output_format="markdown",
                deduplicate=True,
            )
            if not extracted_text:
                return f"[EXTRACT FAILED] No content extracted from: {url}"

            payload = self._build_payload(
                requested_url=url,
                final_url=response.url or url,
                extracted_text=extracted_text,
            )
            self._write_cache_entry(url, payload)
            return payload
        except requests.RequestException as exc:
            return f"[ERROR] {url} :: {type(exc).__name__}: {exc}"
        except Exception as exc:
            return f"[ERROR] {url} :: {type(exc).__name__}: {exc}"

    def _strip_tags(self, value: str) -> str:
        cleaned = TAG_PATTERN.sub(" ", value or "")
        return html.unescape(" ".join(cleaned.split()))

    def _extract_preview_title(self, html_text: str) -> str | None:
        match = TITLE_PATTERN.search(html_text)
        if not match:
            return None
        return compact_text(self._strip_tags(match.group(1)))

    def _extract_meta_description(self, html_text: str) -> str | None:
        match = META_DESCRIPTION_PATTERN.search(html_text)
        if not match:
            return None
        return compact_text(self._strip_tags(match.group(1)))

    def _extract_headings(self, html_text: str) -> list[str]:
        headings: list[str] = []
        seen: set[str] = set()
        for _, heading_html in HEADING_PATTERN.findall(html_text):
            heading = compact_text(self._strip_tags(heading_html))
            if not heading or heading.lower() in seen:
                continue
            headings.append(heading)
            seen.add(heading.lower())
            if len(headings) >= 8:
                break
        return headings

    def _extract_json_ld_types(self, html_text: str) -> list[str]:
        detected_types: list[str] = []
        seen: set[str] = set()
        for block in JSON_LD_PATTERN.findall(html_text):
            for raw_type in re.findall(r'"@type"\s*:\s*"([^"]+)"', block):
                normalized = compact_text(raw_type)
                if not normalized or normalized.lower() in seen:
                    continue
                detected_types.append(normalized)
                seen.add(normalized.lower())
                if len(detected_types) >= 8:
                    return detected_types
        return detected_types

    def _extract_toc_hints(self, html_text: str) -> list[str]:
        hints: list[str] = []
        seen: set[str] = set()
        for anchor_html in ANCHOR_PATTERN.findall(html_text):
            anchor_text = compact_text(self._strip_tags(anchor_html))
            if not anchor_text:
                continue
            lowered = anchor_text.lower()
            if not any(term in lowered for term in TOC_HINT_TERMS):
                continue
            if lowered in seen:
                continue
            hints.append(anchor_text)
            seen.add(lowered)
            if len(hints) >= 8:
                break
        return hints

    def _infer_page_type(
        self,
        *,
        url: str,
        title: str | None,
        headings: list[str],
        meta_description: str | None,
        preview_text: str | None,
    ) -> str | None:
        combined = " ".join(
            part
            for part in [url, title or "", meta_description or "", preview_text or "", *headings]
            if part
        ).lower()
        if any(term in combined for term in ("privacy policy", "terms of service", "cookie policy")):
            return "policy"
        if any(term in combined for term in ("sign in", "log in", "create account", "subscribe")):
            return "account"
        if any(term in combined for term in ("search results", "tag archive", "category archive", "author page")):
            return "search"
        if any(term in combined for term in ("pricing", "plans", "billing")):
            return "pricing"
        if any(term in combined for term in ("benchmark", "leaderboard", "latency", "evaluation")):
            return "comparison"
        if any(term in combined for term in ("menu", "hours", "reservations", "order online", "location")):
            return "business_detail"
        if any(term in combined for term in ("fixture", "fixtures", "schedule", "live score", "scores", "standings", "match center")):
            return "live_status"
        if any(term in combined for term in ("compare", "comparison", "review", "guide", "best", "top")):
            return "comparison"
        if any(term in combined for term in ("directory", "listing", "find", "browse")):
            return "directory"
        if any(term in combined for term in ("profile", "about us", "company", "team", "overview")):
            return "profile"
        if any(term in combined for term in ("readme", "repository", "docs", "documentation", "api reference", "quickstart", "faq")):
            return "reference"
        if any(term in combined for term in ("home", "homepage", "welcome")):
            return "landing"
        return None

    def _build_preview_record(
        self,
        *,
        url: str,
        final_url: str,
        status_code: int | None,
        content_type: str | None,
        html_text: str,
    ) -> PreviewFetchRecord:
        extracted_preview = trafilatura.extract(
            html_text,
            url=url,
            favor_recall=True,
            include_tables=True,
            include_images=False,
            output_format="markdown",
            deduplicate=True,
        )
        preview_text = compact_text(extracted_preview) or compact_text(
            self._strip_tags(html_text)[:2200]
        )
        title = self._extract_preview_title(html_text)
        meta_description = self._extract_meta_description(html_text)
        headings = self._extract_headings(html_text)
        json_ld_types = self._extract_json_ld_types(html_text)
        toc_hints = self._extract_toc_hints(html_text)
        page_type = self._infer_page_type(
            url=final_url,
            title=title,
            headings=headings,
            meta_description=meta_description,
            preview_text=preview_text,
        )
        return PreviewFetchRecord(
            url=self._normalize_url(url),
            final_url=self._normalize_url(final_url),
            status_code=status_code,
            content_type=compact_text(content_type),
            title=title,
            meta_description=meta_description,
            headings=headings,
            json_ld_types=json_ld_types,
            toc_hints=toc_hints,
            preview_text=preview_text,
            page_type=page_type,
        )

    def _preview_one(self, url: str, max_bytes: int) -> PreviewFetchRecord:
        status_code: int | None = None
        content_type: str | None = None
        try:
            with self._build_session() as session:
                try:
                    head_response = session.head(
                        url,
                        timeout=min(self.timeout_seconds, self.PREVIEW_TIMEOUT_SECONDS),
                        allow_redirects=True,
                    )
                    status_code = head_response.status_code
                    content_type = head_response.headers.get("Content-Type")
                except requests.RequestException:
                    pass

                response = session.get(
                    url,
                    timeout=min(self.timeout_seconds, self.PREVIEW_TIMEOUT_SECONDS),
                    allow_redirects=True,
                    stream=True,
                    headers={"Range": f"bytes=0-{max_bytes - 1}"},
                )
                response.raise_for_status()
                status_code = response.status_code
                content_type = response.headers.get("Content-Type") or content_type

                raw_chunks: list[bytes] = []
                total_bytes = 0
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                    remaining = max_bytes - total_bytes
                    if remaining <= 0:
                        break
                    raw_piece = chunk[:remaining]
                    raw_chunks.append(raw_piece)
                    total_bytes += len(raw_piece)
                    if total_bytes >= max_bytes:
                        break

                raw_bytes = b"".join(raw_chunks)
                if not raw_bytes:
                    return PreviewFetchRecord(
                        url=self._normalize_url(url),
                        final_url=self._normalize_url(response.url or url),
                        status_code=status_code,
                        content_type=compact_text(content_type),
                        error=f"Preview returned an empty body: {url}",
                    )

                encoding = response.encoding or response.apparent_encoding or "utf-8"
                html_text = raw_bytes.decode(encoding, errors="ignore")
                return self._build_preview_record(
                    url=url,
                    final_url=response.url or url,
                    status_code=status_code,
                    content_type=content_type,
                    html_text=html_text,
                )
        except requests.RequestException as exc:
            return PreviewFetchRecord(
                url=self._normalize_url(url),
                final_url=self._normalize_url(url),
                status_code=status_code,
                content_type=compact_text(content_type),
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            return PreviewFetchRecord(
                url=self._normalize_url(url),
                final_url=self._normalize_url(url),
                status_code=status_code,
                content_type=compact_text(content_type),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _fetch_one(
        self,
        url: str,
        focus_query: str | None = None,
        full_document: bool = False,
    ) -> str:
        cached_payload = self._load_cache_entry(url)
        payload: dict[str, Any] | None = cached_payload

        if payload is None:
            downloaded_payload = self._download_and_extract(url)
            if isinstance(downloaded_payload, str):
                stale_payload = self._load_cache_entry(url, allow_stale=True)
                if stale_payload is None:
                    return downloaded_payload
                payload = stale_payload
            else:
                payload = downloaded_payload

        extracted_text = compact_text(str(payload.get("content") or ""))
        if full_document and extracted_text:
            return extracted_text

        passages = payload.get("passages")
        if not isinstance(passages, list) or not passages:
            passages = self._extract_passages(str(payload.get("content") or ""))

        selected_passages = self._select_relevant_passages(
            passages,
            focus_query=focus_query,
        )
        rendered = self._render_passages(selected_passages)
        if rendered:
            return rendered

        if extracted_text:
            return extracted_text
        return f"[EXTRACT FAILED] No content extracted from: {url}"

    @classmethod
    def is_error_result(cls, content: str) -> bool:
        return content.startswith(cls.ERROR_PREFIXES)

    def run(self, params: FetchURLToolInputSchema) -> FetchURLToolOutputSchema:
        urls = params.value
        if not urls:
            return FetchURLToolOutputSchema(result=[])

        focus_query = compact_text(params.focus_query)
        full_document = params.full_document
        worker_count = min(self.max_workers, len(urls))
        if worker_count == 1:
            results = [
                self._fetch_one(
                    url,
                    focus_query=focus_query,
                    full_document=full_document,
                )
                for url in urls
            ]
            return FetchURLToolOutputSchema(result=results)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    lambda url: self._fetch_one(
                        url,
                        focus_query=focus_query,
                        full_document=full_document,
                    ),
                    urls,
                )
            )

        return FetchURLToolOutputSchema(result=results)

    def preview(self, params: PreviewURLToolInputSchema) -> PreviewURLToolOutputSchema:
        urls = params.value
        if not urls:
            return PreviewURLToolOutputSchema(result=[])

        max_bytes = max(2048, params.max_bytes or self.DEFAULT_PREVIEW_MAX_BYTES)
        worker_count = min(self.max_workers, len(urls))
        if worker_count == 1:
            return PreviewURLToolOutputSchema(
                result=[self._preview_one(url, max_bytes=max_bytes) for url in urls]
            )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    lambda url: self._preview_one(url, max_bytes=max_bytes),
                    urls,
                )
            )

        return PreviewURLToolOutputSchema(result=results)

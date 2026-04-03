import json
import re
import time
from typing import Callable

from schema import EntityRow, InformationAgentOutput, SourceCitation, StructuredEntityTable, TableCell, TableColumn
from tools.fetch_url import FetchURLTool, FetchURLToolInputSchema
from tools.intent_classificaiton import build_fallback_intent_decomposition
from tools.web_search_tool import SearchTool, SearchToolInput
from tools.write_to_file import WriteToFileTool, WriteToFileToolInputSchema
from utils.config import (
    JSON_OUTPUT_PATH,
    MARKDOWN_OUTPUT_PATH,
    get_lightning_deadline_seconds,
    get_lightning_fetch_url_limit,
    get_lightning_max_content_chars,
    get_lightning_max_search_queries,
    get_lightning_model_name,
    get_lightning_request_timeout_seconds,
    get_lightning_result_row_limit,
)
from utils.llm_utils import run_structured_generation
from utils.result_utils import normalize_result
from utils.text_utils import compact_text, normalize_key, render_markdown_document, truncate_content


DOCUMENT_SUFFIXES = (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx")
LIST_PAGE_HINTS = (
    "best",
    "top",
    "things to do",
    "guide",
    "itinerary",
    "activities",
    "attractions",
    "visit",
    "travel",
)
ENTITY_PATTERN = re.compile(
    r"(?P<name>[A-Z][A-Za-z0-9&'().,-]*(?:\s+(?:[A-Z][A-Za-z0-9&'().,-]*|of|the|and|for|at|in|on|de|la|&)){0,8})\s+"
    r"(?:is|are|was|were|offers|offer|features|feature|serves|serve|located|includes|include|provides|provide|houses|house|contains|contain)\b"
)
GENERIC_NAME_HINTS = {
    "home",
    "homepage",
    "guide",
    "travel guide",
    "visit",
    "welcome",
    "top things to do",
    "things to do",
    "visit amherst",
    "museums in amherst",
}
CATEGORY_KEYWORDS = {
    "museum": "Museum",
    "park": "Park",
    "restaurant": "Restaurant",
    "pizza": "Restaurant",
    "startup": "Startup",
    "company": "Company",
    "tool": "Tool",
    "database": "Database",
    "software": "Software",
    "brewery": "Brewery",
    "cafe": "Cafe",
    "hotel": "Hotel",
    "trail": "Trail",
    "beach": "Beach",
    "gallery": "Gallery",
    "theater": "Theater",
    "venue": "Venue",
    "tour": "Tour",
    "attraction": "Attraction",
}


def build_empty_table(query: str) -> StructuredEntityTable:
    return StructuredEntityTable(
        query=query,
        title=f"Discovered entities for {query}",
        columns=[],
        rows=[],
        sources=[],
    )


def remaining_time_seconds(start_time: float, budget_seconds: float) -> float:
    return max(0.0, budget_seconds - (time.perf_counter() - start_time))


def is_document_url(url: str) -> bool:
    lowered = url.lower()
    return any(lowered.endswith(suffix) for suffix in DOCUMENT_SUFFIXES)


def build_search_candidates(
    query: str,
    intent_decomposition,
) -> list[str]:
    candidates = [query]
    candidates.extend(
        search_request.query for search_request in intent_decomposition.search_requests
    )
    candidates.extend(
        [
            f"{query} official",
            f"{query} best top list",
        ]
    )
    return list(
        dict.fromkeys(
            candidate.strip()
            for candidate in candidates
            if isinstance(candidate, str) and candidate.strip()
        )
    )


def score_search_result(query: str, item: dict[str, object]) -> int:
    title = compact_text(str(item.get("title") or "")) or ""
    snippet = compact_text(str(item.get("body") or item.get("snippet") or "")) or ""
    url = compact_text(str(item.get("href") or item.get("url") or "")) or ""
    combined = f"{title} {snippet}".lower()
    normalized_terms = {
        normalize_key(term).replace("_", " ")
        for term in query.lower().replace("?", " ").split()
        if len(term) > 2
    }

    score = 0
    score += sum(2 for term in normalized_terms if term and term in combined)
    score += sum(3 for hint in LIST_PAGE_HINTS if hint in combined)
    if url and not is_document_url(url):
        score += 2
    if any(
        generic_home_hint in combined
        for generic_home_hint in ("homepage", "home page", "official site")
    ):
        score -= 1
    return score


def select_lightning_urls(
    query: str,
    search_results: list[dict[str, object]],
    max_urls: int,
) -> list[str]:
    seen_urls: set[str] = set()
    selected_urls: list[str] = []
    seen_domains: set[str] = set()

    ranked_results = sorted(
        search_results,
        key=lambda item: score_search_result(query, item),
        reverse=True,
    )

    deferred_urls: list[str] = []
    for item in ranked_results:
        url = compact_text(str(item.get("href") or item.get("url") or ""))
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        domain = url.split("/")[2] if "://" in url else url
        if is_document_url(url):
            deferred_urls.append(url)
            continue

        if domain in seen_domains and len(ranked_results) > max_urls:
            deferred_urls.append(url)
            continue

        seen_domains.add(domain)
        selected_urls.append(url)
        if len(selected_urls) >= max_urls:
            return selected_urls[:max_urls]

    for url in deferred_urls:
        if len(selected_urls) >= max_urls:
            break
        if url not in selected_urls:
            selected_urls.append(url)

    return selected_urls[:max_urls]


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "rate limit" in message
        or "too many requests" in message
    )


def run_lightning_structured_generation(
    *,
    response_schema,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    request_timeout_seconds: float,
    max_output_tokens: int,
    start_time: float,
    deadline_seconds: float,
) -> tuple[object, int]:
    attempts = 0
    last_exc: Exception | None = None
    for backoff_seconds in (0.0, 0.45):
        attempts += 1
        try:
            return (
                run_structured_generation(
                    response_schema=response_schema,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=model_name,
                    request_timeout_seconds=request_timeout_seconds,
                    max_output_tokens=max_output_tokens,
                ),
                attempts,
            )
        except Exception as exc:
            last_exc = exc
            if not is_rate_limit_error(exc):
                raise
            if backoff_seconds <= 0:
                continue
            if remaining_time_seconds(start_time, deadline_seconds) <= backoff_seconds + 0.6:
                break
            time.sleep(backoff_seconds)

    if last_exc is not None:
        raise last_exc
    raise ValueError("Structured generation failed without an exception.")


def infer_category(text: str) -> str | None:
    lowered = text.lower()
    for keyword, label in CATEGORY_KEYWORDS.items():
        if keyword in lowered:
            return label
    return None


def is_generic_candidate(name: str, query: str) -> bool:
    compact_name = compact_text(name) or ""
    lowered = compact_name.lower()
    if not compact_name:
        return True
    if lowered in GENERIC_NAME_HINTS:
        return True
    if any(hint in lowered for hint in ("things to do", "travel guide", "visit ", "top ", "best ")):
        return True
    if normalize_key(compact_name) == normalize_key(query):
        return True
    if len(compact_name.split()) > 8:
        return True
    return False


def extract_name_and_quote(line: str, query: str) -> tuple[str | None, str | None]:
    compact_line = compact_text(line)
    if not compact_line or len(compact_line) < 8:
        return None, None

    match = ENTITY_PATTERN.search(compact_line)
    if match:
        candidate = compact_text(match.group("name"))
        if candidate and not is_generic_candidate(candidate, query):
            return candidate, compact_line

    stripped = compact_line.lstrip("#*-0123456789. )(").strip()
    if (
        stripped
        and len(stripped.split()) <= 6
        and any(char.isupper() for char in stripped)
        and line.lstrip().startswith(("-", "*"))
        and not is_generic_candidate(stripped, query)
    ):
        return stripped, compact_line

    return None, None


def build_heuristic_fallback_table(
    query: str,
    fetched_records: list[dict[str, object]],
    source_registry: dict[str, dict[str, str | None]],
) -> StructuredEntityTable:
    columns = [
        TableColumn(
            key="name",
            label="Name",
            description="Name of the discovered entity",
        ),
        TableColumn(
            key="summary",
            label="Summary",
            description="Short source-backed description",
        ),
    ]
    rows: list[EntityRow] = []
    seen_names: set[str] = set()

    for record in fetched_records:
        source_id = compact_text(str(record.get("source_id") or ""))
        source_title = compact_text(str(record.get("title") or ""))
        source_url = compact_text(str(record.get("url") or ""))
        content = str(record.get("content") or "")
        if not source_id or not source_title or not source_url or not content:
            continue

        for raw_line in content.splitlines():
            candidate_name, quote = extract_name_and_quote(raw_line, query)
            if not candidate_name or not quote:
                continue

            normalized_name = normalize_key(candidate_name)
            if not normalized_name or normalized_name in seen_names:
                continue

            citation = SourceCitation(
                source_id=source_id,
                source_title=source_title,
                source_url=source_url,
                quote=quote[:220],
            )
            summary_value = compact_text(quote)
            if not summary_value:
                continue

            row_cells = {
                "name": TableCell(value=candidate_name, citations=[citation]),
                "summary": TableCell(value=summary_value, citations=[citation]),
            }
            category = infer_category(summary_value)
            if category:
                if not any(column.key == "category" for column in columns):
                    columns.append(
                        TableColumn(
                            key="category",
                            label="Category",
                            description="High-level category inferred from fetched content",
                        )
                    )
                row_cells["category"] = TableCell(value=category, citations=[citation])

            rows.append(
                EntityRow(
                    entity_id=normalized_name,
                    cells=row_cells,
                )
            )
            seen_names.add(normalized_name)
            if len(rows) >= get_lightning_result_row_limit():
                break

        if len(rows) >= get_lightning_result_row_limit():
            break

    heuristic_table = StructuredEntityTable(
        query=query,
        title=f"Discovered entities for {query}",
        columns=columns,
        rows=rows,
        sources=[],
    )
    return normalize_result(
        heuristic_table,
        query=query,
        source_registry=source_registry,
        require_complete=False,
    )


def build_lightning_extraction_prompts(
    query: str,
    fetched_records: list[dict[str, object]],
    allow_partial: bool = False,
) -> tuple[str, str]:
    row_limit = get_lightning_result_row_limit()
    if allow_partial:
        column_instruction = 'Choose 2 or 3 columns total, including "name".'
        completeness_instruction = (
            "If complete rows are hard to support, return the best partially complete "
            "citation-backed rows instead of returning an empty table."
        )
    else:
        column_instruction = 'Choose 3 or 4 columns total, including "name".'
        completeness_instruction = (
            "Reduce the number of columns if needed so that you can still return rows."
        )

    system_prompt = f"""
You are a lightning-fast entity extraction system.
Return valid JSON matching the StructuredEntityTable schema.
Optimize for speed and precision over coverage.
Use only the fetched source content below as grounding.
{column_instruction}
Return at most {row_limit} rows.
Prefer columns that are likely to be complete across multiple entities, such as summary, category, location, or website.
Every populated cell must include at least one citation with exact source_id, source_title, source_url, and a short verbatim quote from the fetched content.
{completeness_instruction}
For broad travel, local, or recommendation queries, entities may be attractions, neighborhoods, museums, parks, tours, restaurants, venues, or activities.
Do not invent values or citations.
""".strip()

    user_prompt = (
        f"Original topic query:\n{query}\n\n"
        f"Fetched sources:\n{json.dumps(fetched_records, ensure_ascii=False, indent=2)}\n"
    )
    return system_prompt, user_prompt


def run_lightning_research(
    information_request: str,
    search_tool: SearchTool,
    fetch_tool: FetchURLTool,
    write_tool: WriteToFileTool,
    progress_callback: Callable[[str], None] | None = None,
) -> InformationAgentOutput:
    progress_callback = progress_callback or (lambda _message: None)
    start_time = time.perf_counter()
    deadline_seconds = get_lightning_deadline_seconds()
    intent_decomposition = build_fallback_intent_decomposition(
        information_request,
        deep_research=False,
    )

    source_registry: dict[str, dict[str, str | None]] = {}
    url_to_source_id: dict[str, str] = {}
    queries_run: list[dict[str, object]] = []
    failed_fetches: list[dict[str, object]] = []
    extraction_error: str | None = None
    normalization_mode = "empty"
    selected_result_count = 0
    extraction_attempts = 0
    llm_rate_limited = False

    search_candidates = build_search_candidates(
        information_request,
        intent_decomposition,
    )[: get_lightning_max_search_queries()]

    aggregated_results: list[dict[str, object]] = []
    progress_callback("Lightning search")
    for query in search_candidates:
        if remaining_time_seconds(start_time, deadline_seconds) <= 2.0:
            break

        output = search_tool.run(SearchToolInput(value=query))
        if output.error:
            queries_run.append(
                {
                    "query": query,
                    "results": 0,
                    "error": output.error,
                }
            )
            continue

        normalized_results: list[dict[str, object]] = []
        for item in output.results:
            url = compact_text(str(item.get("href") or item.get("url") or ""))
            title = compact_text(str(item.get("title") or ""))
            snippet = compact_text(str(item.get("body") or item.get("snippet") or ""))
            if not url or not title:
                continue

            source_id = url_to_source_id.get(url)
            if not source_id:
                source_id = f"src_{len(url_to_source_id) + 1:03d}"
                url_to_source_id[url] = source_id

            source_registry[source_id] = {
                "source_id": source_id,
                "title": title,
                "url": url,
                "snippet": snippet,
            }
            normalized_results.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "href": url,
                    "body": snippet or "",
                }
            )

        queries_run.append(
            {
                "query": query,
                "results": len(normalized_results),
            }
        )
        aggregated_results.extend(normalized_results)
        if len(aggregated_results) >= get_lightning_fetch_url_limit() * 3:
            break

    urls_to_fetch = select_lightning_urls(
        information_request,
        aggregated_results,
        max_urls=get_lightning_fetch_url_limit(),
    )
    selected_result_count = len(aggregated_results)
    fetched_records: list[dict[str, object]] = []

    if urls_to_fetch and remaining_time_seconds(start_time, deadline_seconds) > 1.0:
        progress_callback(f"Lightning fetch ({len(urls_to_fetch)} page(s))")
        fetch_output = fetch_tool.run(FetchURLToolInputSchema(value=urls_to_fetch))
        max_content_chars = get_lightning_max_content_chars()

        for url, content in zip(urls_to_fetch, fetch_output.result):
            source_id = url_to_source_id.get(url)
            if not source_id:
                continue

            metadata = source_registry[source_id]
            if fetch_tool.is_error_result(content):
                failed_fetches.append(
                    {
                        "source_id": source_id,
                        "title": metadata.get("title") or url,
                        "url": url,
                        "error": content,
                    }
                )
                continue

            fetched_records.append(
                {
                    "source_id": source_id,
                    "title": metadata.get("title") or url,
                    "url": url,
                    "snippet": metadata.get("snippet"),
                    "content": truncate_content(content, max_chars=max_content_chars),
                }
            )

    final_table = build_empty_table(information_request)
    if fetched_records and remaining_time_seconds(start_time, deadline_seconds) > 0.5:
        progress_callback("Lightning extraction")
        system_prompt, user_prompt = build_lightning_extraction_prompts(
            information_request,
            fetched_records,
            allow_partial=False,
        )

        try:
            extracted_table, attempts_used = run_lightning_structured_generation(
                response_schema=StructuredEntityTable,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=get_lightning_model_name(),
                request_timeout_seconds=min(
                    get_lightning_request_timeout_seconds(),
                    max(
                        0.5,
                        remaining_time_seconds(start_time, deadline_seconds) - 0.25,
                    ),
                ),
                max_output_tokens=1800,
                start_time=start_time,
                deadline_seconds=deadline_seconds,
            )
            extraction_attempts += attempts_used
            strict_table = normalize_result(
                extracted_table,
                query=information_request,
                source_registry=source_registry,
                require_complete=True,
            )
            if strict_table.rows:
                final_table = strict_table
                normalization_mode = "strict"
            else:
                partial_table = normalize_result(
                    extracted_table,
                    query=information_request,
                    source_registry=source_registry,
                    require_complete=False,
                )
                if partial_table.rows:
                    final_table = partial_table
                    normalization_mode = "partial_fallback"
                elif remaining_time_seconds(start_time, deadline_seconds) > 1.0:
                    fallback_system_prompt, fallback_user_prompt = (
                        build_lightning_extraction_prompts(
                            information_request,
                            fetched_records,
                            allow_partial=True,
                        )
                    )
                    fallback_extracted_table, attempts_used = (
                        run_lightning_structured_generation(
                            response_schema=StructuredEntityTable,
                            system_prompt=fallback_system_prompt,
                            user_prompt=fallback_user_prompt,
                            model_name=get_lightning_model_name(),
                            request_timeout_seconds=min(
                                max(
                                    0.75,
                                    remaining_time_seconds(start_time, deadline_seconds) - 0.1,
                                ),
                                1.75,
                            ),
                            max_output_tokens=1200,
                            start_time=start_time,
                            deadline_seconds=deadline_seconds,
                        )
                    )
                    extraction_attempts += attempts_used
                    fallback_table = normalize_result(
                        fallback_extracted_table,
                        query=information_request,
                        source_registry=source_registry,
                        require_complete=False,
                    )
                    if fallback_table.rows:
                        final_table = fallback_table
                        normalization_mode = "secondary_partial_fallback"
        except Exception as exc:
            extraction_error = f"{type(exc).__name__}: {exc}"
            llm_rate_limited = is_rate_limit_error(exc)

    if not final_table.rows and fetched_records:
        heuristic_table = build_heuristic_fallback_table(
            query=information_request,
            fetched_records=fetched_records,
            source_registry=source_registry,
        )
        if heuristic_table.rows:
            final_table = heuristic_table
            normalization_mode = (
                "heuristic_after_rate_limit" if llm_rate_limited else "heuristic_fallback"
            )

    progress_callback("Writing output files")
    json_payload = final_table.model_dump_json(indent=2)
    markdown_payload = render_markdown_document(final_table)
    json_write = write_tool.run(
        WriteToFileToolInputSchema(
            path=str(JSON_OUTPUT_PATH),
            content=json_payload,
        )
    )
    markdown_write = write_tool.run(
        WriteToFileToolInputSchema(
            path=str(MARKDOWN_OUTPUT_PATH),
            content=markdown_payload,
        )
    )

    execution_time_ms = int((time.perf_counter() - start_time) * 1000)
    return InformationAgentOutput(
        status="success",
        json_file_path=json_write.absolute_path,
        markdown_file_path=markdown_write.absolute_path,
        result=final_table,
        meta={
            "mode": "lightning",
            "deep_research": False,
            "recursive_research": {"enabled": False, "skipped_for_mode": True},
            "intent_decomposition": intent_decomposition.model_dump(),
            "queries_run": queries_run,
            "search_candidates": search_candidates,
            "selected_result_count": selected_result_count,
            "fetch_calls": 1 if urls_to_fetch else 0,
            "fetch_limit": get_lightning_fetch_url_limit(),
            "fetched_records": len(fetched_records),
            "failed_fetches": failed_fetches,
            "deadline_seconds": deadline_seconds,
            "execution_time_ms": execution_time_ms,
            "model": get_lightning_model_name(),
            "extraction_error": extraction_error,
            "extraction_attempts": extraction_attempts,
            "llm_rate_limited": llm_rate_limited,
            "normalization_mode": normalization_mode,
            "rows_returned": len(final_table.rows),
        },
    )

import json
import logging
from typing import Any, Callable

from schema import EntityRow, RecursiveResearchFill, SourceCitation, StructuredEntityTable, TableCell, TableColumn
from tools.fetch_url import FetchURLTool, FetchURLToolInputSchema
from tools.web_search_tool import SearchTool, SearchToolInput
from utils.config import (
    get_recursive_research_max_fetch_urls,
    get_recursive_research_max_rounds,
    get_recursive_search_result_limit,
)
from utils.llm_utils import run_structured_generation
from utils.result_utils import collect_sources_from_rows
from utils.text_utils import compact_text, normalize_key, truncate_content

logger = logging.getLogger(__name__)


def get_missing_columns(row: EntityRow, columns: list[TableColumn]) -> list[TableColumn]:
    return [
        column
        for column in columns
        if column.key not in row.cells
        or not row.cells[column.key].value
        or not row.cells[column.key].citations
    ]


def build_recursive_query(
    entity_name: str,
    missing_columns: list[TableColumn],
    base_query: str,
    round_index: int,
) -> str:
    special_terms = {
        "website": "official website site about",
        "location": "location headquarters address about",
        "summary": "overview profile about",
        "category": "type category what it is",
    }

    query_terms: list[str] = []
    for column in missing_columns:
        query_terms.append(column.label)
        if column.key in special_terms:
            query_terms.append(special_terms[column.key])

    deduped_terms = list(dict.fromkeys(term for term in query_terms if term))
    if round_index == 1:
        return f"\"{entity_name}\" {base_query} {' '.join(deduped_terms[:4])}".strip()

    return f"\"{entity_name}\" {' '.join(deduped_terms)} official about details".strip()


def merge_recursive_fill(
    row: EntityRow,
    fill: RecursiveResearchFill,
    allowed_columns: list[TableColumn],
    source_registry: dict[str, dict[str, str | None]],
) -> int:
    allowed_keys = {column.key for column in allowed_columns}
    filled_count = 0

    for candidate in fill.filled_cells:
        key = normalize_key(candidate.key)
        if key not in allowed_keys:
            continue
        if key in row.cells and row.cells[key].value and row.cells[key].citations:
            continue

        value = compact_text(candidate.value)
        if not value:
            continue

        citations: list[SourceCitation] = []
        for citation in candidate.citations:
            source_id = compact_text(citation.source_id)
            source_meta = source_registry.get(source_id or "", {})
            source_title = compact_text(citation.source_title) or compact_text(
                source_meta.get("title")
            )
            source_url = compact_text(citation.source_url) or compact_text(
                source_meta.get("url")
            )
            quote = compact_text(citation.quote)

            if not source_id or not source_title or not source_url or not quote:
                continue

            citations.append(
                SourceCitation(
                    source_id=source_id,
                    source_title=source_title,
                    source_url=source_url,
                    quote=quote,
                )
            )

        if not citations:
            continue

        row.cells[key] = TableCell(value=value, citations=citations)
        filled_count += 1

    return filled_count


def run_recursive_research(
    result: StructuredEntityTable,
    original_query: str,
    deep_research: bool,
    search_tool: SearchTool,
    fetch_tool: FetchURLTool,
    source_registry: dict[str, dict[str, str | None]],
    url_to_source_id: dict[str, str],
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[StructuredEntityTable, dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": True,
        "rows_considered": 0,
        "queries_run": [],
        "cells_filled": 0,
        "rounds_attempted": 0,
    }
    max_rounds = get_recursive_research_max_rounds()
    max_fetch_urls = get_recursive_research_max_fetch_urls()

    for row in result.rows:
        missing_columns = get_missing_columns(row, result.columns)
        if not missing_columns:
            continue

        stats["rows_considered"] += 1
        entity_name = row.cells["name"].value or row.entity_id

        for round_index in range(1, max_rounds + 1):
            missing_columns = get_missing_columns(row, result.columns)
            if not missing_columns:
                break

            stats["rounds_attempted"] += 1
            query = build_recursive_query(
                entity_name=entity_name,
                missing_columns=missing_columns,
                base_query=original_query,
                round_index=round_index,
            )
            if progress_callback is not None:
                progress_callback(
                    f"Recursive research for {entity_name} ({round_index}/{max_rounds})"
                )
            search_output = search_tool.run(SearchToolInput(value=query))
            if search_output.error:
                stats["queries_run"].append(
                    {
                        "entity_id": row.entity_id,
                        "entity_name": entity_name,
                        "round": round_index,
                        "query": query,
                        "missing_keys": [column.key for column in missing_columns],
                        "search_results": 0,
                        "error": search_output.error,
                    }
                )
                logger.warning(
                    "recursive search failed for %s on round %d: %s",
                    entity_name,
                    round_index,
                    search_output.error,
                )
                continue
            selected_results = search_output.results[
                : get_recursive_search_result_limit(deep_research)
            ]
            stats["queries_run"].append(
                {
                    "entity_id": row.entity_id,
                    "entity_name": entity_name,
                    "round": round_index,
                    "query": query,
                    "missing_keys": [column.key for column in missing_columns],
                    "search_results": len(selected_results),
                }
            )

            enriched_results: list[dict[str, str | None]] = []
            for item in selected_results:
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
                enriched_results.append(
                    {
                        "source_id": source_id,
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
                )

            urls_to_fetch = [
                result_item["url"]
                for result_item in enriched_results[:max_fetch_urls]
                if result_item["url"]
            ]
            if not urls_to_fetch:
                continue

            if progress_callback is not None:
                progress_callback(
                    f"Fetching follow-up sources for {entity_name} ({len(urls_to_fetch)} page(s))"
                )
            fetch_output = fetch_tool.run(FetchURLToolInputSchema(value=urls_to_fetch))
            fetched_records: list[dict[str, str | None]] = []
            for url, content in zip(urls_to_fetch, fetch_output.result):
                if fetch_tool.is_error_result(content):
                    continue

                source_id = url_to_source_id.get(url)
                if not source_id:
                    continue

                source_meta = source_registry[source_id]
                fetched_records.append(
                    {
                        "source_id": source_id,
                        "title": compact_text(source_meta.get("title")) or url,
                        "url": url,
                        "content": truncate_content(content),
                    }
                )

            if not fetched_records:
                continue

            current_values = {
                column.key: row.cells[column.key].value
                for column in result.columns
                if column.key in row.cells and row.cells[column.key].value
            }
            missing_descriptions = [
                {
                    "key": column.key,
                    "label": column.label,
                    "description": column.description,
                }
                for column in missing_columns
            ]

            fill_prompt = """
You fill missing cells for one entity in a structured table.
Return valid JSON matching the provided schema.
Only fill keys that are currently missing.
Do not modify or repeat already-known values.
Use only the fetched source content below.
Every returned cell must include at least one citation with exact source_id, source_title, source_url, and a short verbatim quote copied from the fetched content.
If a missing field is still unresolved, omit it from the response.
"""
            fill_user_prompt = (
                f"Original query:\n{original_query}\n\n"
                f"Entity name:\n{entity_name}\n\n"
                f"Known values:\n{json.dumps(current_values, ensure_ascii=False, indent=2)}\n\n"
                f"Missing columns:\n{json.dumps(missing_descriptions, ensure_ascii=False, indent=2)}\n\n"
                f"Fetched sources:\n{json.dumps(fetched_records, ensure_ascii=False, indent=2)}\n"
            )

            try:
                if progress_callback is not None:
                    progress_callback(f"Filling missing cells for {entity_name}")
                fill_result: RecursiveResearchFill = run_structured_generation(
                    RecursiveResearchFill,
                    fill_prompt,
                    fill_user_prompt,
                )
            except Exception:
                continue

            filled_count = merge_recursive_fill(
                row=row,
                fill=fill_result,
                allowed_columns=missing_columns,
                source_registry=source_registry,
            )
            stats["cells_filled"] += filled_count
            if filled_count > 0:
                logger.info(
                    "recursive research filled %d cell(s) for %s",
                    filled_count,
                    entity_name,
                )

    result.sources = collect_sources_from_rows(result.rows, source_registry)
    return result, stats

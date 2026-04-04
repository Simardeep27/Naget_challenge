from __future__ import annotations

import json
from typing import Any, Callable

from atomic_agents import BaseIOSchema
from pydantic import Field

from schema import (
    EntityRow,
    ResearchPlan,
    ResearchSlot,
    SourceCitation,
    StructuredEntityTable,
    TableCell,
)
from utils.llm_utils import run_structured_generation
from utils.result_utils import collect_sources_from_rows
from utils.text_utils import compact_text, normalize_key

RECURSIVE_BACKFILL_MODEL_NAME ="gemini-3.1-pro-preview"
RECURSIVE_BACKFILL_BATCH_SIZE = 5
RECURSIVE_BACKFILL_MAX_BATCHES = 1


class GroundedBackfillCitation(BaseIOSchema):
    """One grounded citation returned by the batched recursive backfill step."""

    source_title: str | None = Field(default=None)
    source_url: str | None = Field(default=None)
    quote: str | None = Field(default=None)


class GroundedBackfillCell(BaseIOSchema):
    """One requested cell fill returned by the batched recursive backfill step."""

    slot_key: str = Field(..., description="Missing slot key to fill")
    value: str | None = Field(default=None, description="Grounded value for the slot")
    citations: list[GroundedBackfillCitation] = Field(default_factory=list)


class GroundedBackfillRow(BaseIOSchema):
    """Batched cell fills for one existing table row."""

    entity_id: str = Field(..., description="Existing row identifier")
    fills: list[GroundedBackfillCell] = Field(default_factory=list)


class GroundedBackfillBatch(BaseIOSchema):
    """Batched grounded answers for the table's missing cells."""

    rows: list[GroundedBackfillRow] = Field(default_factory=list)


def _chunk_rows(
    rows: list[dict[str, object]],
    *,
    chunk_size: int,
) -> list[list[dict[str, object]]]:
    size = max(1, chunk_size)
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _row_has_missing_target_slot(
    row: EntityRow,
    *,
    target_slots: list[ResearchSlot],
) -> bool:
    return any(_is_slot_missing(row, slot) for slot in target_slots)


def _is_slot_missing(row: EntityRow, slot: ResearchSlot) -> bool:
    key = normalize_key(slot.key or slot.label)
    if key == "name":
        return False
    cell = row.cells.get(key)
    return not cell or not compact_text(cell.value) or not cell.citations


def _target_slots(
    research_plan: ResearchPlan,
    *,
    include_optional_slots: bool,
) -> list[ResearchSlot]:
    slots = list(research_plan.required_slots)
    if include_optional_slots:
        slots.extend(research_plan.nice_to_have_slots)
    return slots


def _row_name(row: EntityRow) -> str:
    return compact_text(
        (row.cells.get("name").value if row.cells.get("name") else None) or row.entity_id
    )


def _known_row_context(row: EntityRow) -> dict[str, str]:
    context: dict[str, str] = {}
    for key, cell in row.cells.items():
        value = compact_text(cell.value)
        if value:
            context[key] = value
    return context


def _missing_row_requests(
    *,
    result: StructuredEntityTable,
    target_slots: list[ResearchSlot],
) -> tuple[list[dict[str, object]], dict[str, set[str]]]:
    requests: list[dict[str, object]] = []
    missing_keys_by_row: dict[str, set[str]] = {}

    for row in result.rows:
        entity_name = _row_name(row)
        if not entity_name:
            continue

        missing_slots = []
        for slot in target_slots:
            if not _is_slot_missing(row, slot):
                continue
            missing_slots.append(
                {
                    "slot_key": normalize_key(slot.key or slot.label),
                    "slot_label": slot.label,
                    "description": slot.description,
                }
            )

        if not missing_slots:
            continue

        requests.append(
            {
                "entity_id": row.entity_id,
                "entity_name": entity_name,
                "known_cells": _known_row_context(row),
                "missing_slots": missing_slots,
            }
        )
        missing_keys_by_row[row.entity_id] = {
            str(slot["slot_key"]) for slot in missing_slots if slot.get("slot_key")
        }

    return requests, missing_keys_by_row


def _register_citation(
    *,
    citation: GroundedBackfillCitation,
    source_registry: dict[str, dict[str, str | None]],
    url_to_source_id: dict[str, str],
) -> SourceCitation | None:
    source_url = compact_text(citation.source_url)
    quote = compact_text(citation.quote)
    source_title = compact_text(citation.source_title) or source_url
    if not source_url or not quote:
        return None

    source_id = url_to_source_id.get(source_url)
    if not source_id:
        source_id = f"src_{len(url_to_source_id) + 1:03d}"
        url_to_source_id[source_url] = source_id

    source_registry[source_id] = {
        "source_id": source_id,
        "title": source_title,
        "url": source_url,
        "snippet": None,
    }
    return SourceCitation(
        source_id=source_id,
        source_title=source_title,
        source_url=source_url,
        quote=quote,
    )


def _build_table_cell(
    *,
    item: GroundedBackfillCell,
    source_registry: dict[str, dict[str, str | None]],
    url_to_source_id: dict[str, str],
) -> TableCell | None:
    value = compact_text(item.value)
    if not value:
        return None

    citations: list[SourceCitation] = []
    seen_pairs: set[tuple[str, str]] = set()
    for citation in item.citations:
        normalized = _register_citation(
            citation=citation,
            source_registry=source_registry,
            url_to_source_id=url_to_source_id,
        )
        if normalized is None:
            continue
        pair = (normalized.source_id, normalized.quote)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        citations.append(normalized)

    if not citations:
        return None
    return TableCell(value=value, citations=citations)


def run_recursive_research(
    result: StructuredEntityTable,
    original_query: str,
    research_plan: ResearchPlan,
    source_registry: dict[str, dict[str, str | None]],
    url_to_source_id: dict[str, str],
    progress_callback: Callable[[str], None] | None = None,
    include_optional_slots: bool = False,
) -> tuple[StructuredEntityTable, dict[str, Any]]:
    target_slots = _target_slots(
        research_plan,
        include_optional_slots=include_optional_slots,
    )
    row_requests, missing_keys_by_row = _missing_row_requests(
        result=result,
        target_slots=target_slots,
    )
    max_rows_to_fix = RECURSIVE_BACKFILL_BATCH_SIZE * RECURSIVE_BACKFILL_MAX_BATCHES
    capped_row_requests = row_requests[:max_rows_to_fix]
    skipped_row_requests = row_requests[max_rows_to_fix:]
    stats: dict[str, Any] = {
        "enabled": True,
        "batched": True,
        "batch_size": RECURSIVE_BACKFILL_BATCH_SIZE,
        "max_batches": RECURSIVE_BACKFILL_MAX_BATCHES,
        "rows_considered": len(capped_row_requests),
        "rows_over_cap": len(skipped_row_requests),
        "slots_considered": sum(len(keys) for keys in missing_keys_by_row.values()),
        "cells_filled": 0,
        "llm_calls": 0,
        "queries_run": [],
        "chunk_errors": [],
    }
    if not capped_row_requests:
        if skipped_row_requests:
            skipped_ids = {
                compact_text(row_request.get("entity_id"))
                for row_request in skipped_row_requests
            }
            result.rows = [
                row for row in result.rows if compact_text(row.entity_id) not in skipped_ids
            ]
        result.sources = collect_sources_from_rows(result.rows, source_registry)
        return result, stats

    system_prompt = """
You repair missing cells in an existing structured table.
Use fresh Google Search inside this model call to find grounded answers.
Do not rely on the previously fetched page bundle from the earlier extraction stage.
Only fill the requested missing cells. Do not add rows, remove rows, or invent extra columns.
Use the row's known cells as context when searching.
Return one fill entry for every requested slot in every provided row.
If a slot cannot be grounded confidently, return value=null and citations=[] for that slot.
Every non-null value must include at least one citation with source_title, source_url, and a short verbatim quote.
Keep values concise and directly usable inside a table cell.
""".strip()
    row_map = {row.entity_id: row for row in result.rows}
    row_chunks = _chunk_rows(
        capped_row_requests,
        chunk_size=RECURSIVE_BACKFILL_BATCH_SIZE,
    )[:RECURSIVE_BACKFILL_MAX_BATCHES]
    total_chunks = len(row_chunks)

    for chunk_index, row_chunk in enumerate(row_chunks, start=1):
        if progress_callback is not None:
            progress_callback(
                f"Grounded LLM backfill chunk {chunk_index}/{total_chunks} ({len(row_chunk)} row(s))"
            )

        chunk_slot_count = 0
        for row_request in row_chunk:
            entity_id = compact_text(row_request.get("entity_id"))
            if entity_id:
                chunk_slot_count += len(missing_keys_by_row.get(entity_id, set()))

        try:
            rows_payload = json.dumps(
                row_chunk,
                ensure_ascii=False,
                indent=2,
            )
            user_prompt = (
                f"Original query:\n{original_query}\n\n"
                f"Research objective:\n{research_plan.objective}\n\n"
                f"Entity type:\n{research_plan.entity_type}\n\n"
                f"Constraints:\n{json.dumps(research_plan.constraints, ensure_ascii=False)}\n\n"
                f"Rows with missing cells to backfill:\n{rows_payload}\n"
            )
            last_error: Exception | None = None
            grounded_batch: GroundedBackfillBatch | None = None
            for attempt in range(2):
                try:
                    stats["llm_calls"] += 1
                    grounded_batch = run_structured_generation(
                        response_schema=GroundedBackfillBatch,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model_name=RECURSIVE_BACKFILL_MODEL_NAME,
                        request_timeout_seconds=90 if attempt == 0 else 120,
                        max_output_tokens=3000,
                        google_search=True,
                    )
                    break
                except Exception as exc:
                    last_error = exc
            if grounded_batch is None:
                raise last_error or RuntimeError("Grounded backfill failed without an error")

        except Exception as exc:
            error_message = str(exc)
            stats["chunk_errors"].append(
                {
                    "chunk_index": chunk_index,
                    "chunk_count": total_chunks,
                    "rows": len(row_chunk),
                    "error": error_message,
                }
            )
            stats["queries_run"].append(
                {
                    "type": "grounded_llm_backfill",
                    "chunk_index": chunk_index,
                    "chunk_count": total_chunks,
                    "rows": len(row_chunk),
                    "slots": chunk_slot_count,
                    "error": error_message,
                }
            )
            continue

        for row_result in grounded_batch.rows:
            row = row_map.get(row_result.entity_id)
            if row is None:
                continue
            allowed_keys = missing_keys_by_row.get(row_result.entity_id, set())
            for fill in row_result.fills:
                slot_key = normalize_key(fill.slot_key)
                if not slot_key or slot_key not in allowed_keys:
                    continue
                if not _is_slot_missing(
                    row,
                    ResearchSlot(
                        key=slot_key,
                        label=slot_key,
                        description="Backfill validation slot",
                        required=False,
                    ),
                ):
                    continue
                cell = _build_table_cell(
                    item=fill,
                    source_registry=source_registry,
                    url_to_source_id=url_to_source_id,
                )
                if cell is None:
                    continue
                row.cells[slot_key] = cell
                stats["cells_filled"] += 1

        stats["queries_run"].append(
            {
                "type": "grounded_llm_backfill",
                "chunk_index": chunk_index,
                "chunk_count": total_chunks,
                "rows": len(row_chunk),
                "slots": chunk_slot_count,
            }
        )

    before_filter_count = len(result.rows)
    result.rows = [
        row for row in result.rows if not _row_has_missing_target_slot(row, target_slots=target_slots)
    ]
    stats["rows_removed_with_missing_values"] = before_filter_count - len(result.rows)
    result.sources = collect_sources_from_rows(result.rows, source_registry)
    return result, stats

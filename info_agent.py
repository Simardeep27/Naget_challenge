from __future__ import annotations

"""Main orchestration for the deterministic research pipeline."""

import time
from dataclasses import dataclass, field
from typing import Callable

from atomic_agents import BaseToolConfig

from schema import (
    EntityRow,
    InformationAgentOutput,
    ResearchPlan,
    SearchCandidate,
    StructuredEntityTable,
)
from tools.fetch_url import FetchURLTool
from tools.research_planner import ResearchPlannerInput, ResearchPlannerTool
from tools.web_search_tool import SearchTool, SearchToolConfig
from tools.write_to_file import WriteToFileTool, WriteToFileToolInputSchema
from utils.config import (
    JSON_OUTPUT_PATH,
    MARKDOWN_OUTPUT_PATH,
    get_fetch_limit,
    get_lightning_fetch_url_limit,
    get_lightning_max_search_queries,
    get_lightning_search_result_limit,
    get_lightning_search_timeout_seconds,
    get_model_name,
    get_search_result_limit,
)
from utils.progress import CliProgressReporter
from utils.recursive_research import run_recursive_research
from utils.result_utils import normalize_result
from utils.text_utils import render_markdown_document
from utils.tiered_research import (
    apply_semantic_rerank,
    build_initial_search_queries,
    extract_table_from_frontier,
    fetch_frontier_records,
    format_research_plan,
    get_required_column_keys,
    merge_tables,
    run_candidate_search,
    select_frontier_candidates,
)


@dataclass(frozen=True)
class PipelineBudgets:
    """Execution budgets for one research run."""

    mode: str
    initial_query_budget: int
    frontier_size: int
    max_frontier_batches: int


@dataclass(frozen=True)
class InformationTools:
    """Typed container for the concrete tools used by one run."""

    planner_tool: ResearchPlannerTool
    search_tool: SearchTool
    fetch_url_tool: FetchURLTool
    write_file_tool: WriteToFileTool


@dataclass
class FrontierState:
    """Mutable state accumulated while walking the frontier."""

    working_table: StructuredEntityTable
    remaining_candidates: list[SearchCandidate]
    source_registry: dict[str, dict[str, str | None]] = field(default_factory=dict)
    url_to_source_id: dict[str, str] = field(default_factory=dict)
    selected_frontier_urls: list[str] = field(default_factory=list)
    fetched_records: list[dict[str, object]] = field(default_factory=list)
    failed_fetches: list[dict[str, object]] = field(default_factory=list)
    frontier_batch_count: int = 0
    frontier_fetch_call_count: int = 0


def _mode_name(*, deep_research: bool, lightning: bool) -> str:
    if lightning:
        return "lightning"
    return "deep" if deep_research else "standard"


def _search_query_budget(*, deep_research: bool, lightning: bool) -> int:
    if lightning:
        return get_lightning_max_search_queries()
    return 4 if deep_research else 3


def _fetch_budget(*, deep_research: bool, lightning: bool) -> int:
    if lightning:
        return get_lightning_fetch_url_limit()
    if deep_research:
        return min(get_fetch_limit(True), 3)
    return get_fetch_limit(False)


def _frontier_size(*, deep_research: bool, lightning: bool) -> int:
    return min(
        _fetch_budget(deep_research=deep_research, lightning=lightning),
        5 if deep_research else 4,
    )


def _max_frontier_batches(*, deep_research: bool, lightning: bool) -> int:
    frontier_size = _frontier_size(deep_research=deep_research, lightning=lightning)
    fetch_budget = _fetch_budget(deep_research=deep_research, lightning=lightning)
    return max(1, -(-fetch_budget // max(1, frontier_size)))


def _extraction_chunk_size(*, deep_research: bool, lightning: bool) -> int:
    if lightning:
        return 2
    return 3 if deep_research else 2


def _empty_table(query: str) -> StructuredEntityTable:
    return StructuredEntityTable(
        query=query,
        title=f"Discovered entities for {query}",
        columns=[],
        rows=[],
        sources=[],
    )


def _flatten_query_history(
    initial_queries: list[dict[str, object]],
    *search_meta_objects: dict[str, object],
) -> list[dict[str, object]]:
    history = list(initial_queries)
    for meta in search_meta_objects:
        for item in meta.get("queries_run", []):
            if isinstance(item, dict):
                history.append(item)
    return history


def _build_pipeline_budgets(
    *,
    plan: ResearchPlan,
    deep_research: bool,
    lightning: bool,
) -> PipelineBudgets:
    return PipelineBudgets(
        mode=_mode_name(deep_research=deep_research, lightning=lightning),
        initial_query_budget=_search_query_budget(
            deep_research=deep_research,
            lightning=lightning,
        ),
        frontier_size=_frontier_size(
            deep_research=deep_research,
            lightning=lightning,
        ),
        max_frontier_batches=_max_frontier_batches(
            deep_research=deep_research,
            lightning=lightning,
        ),
    )


def create_information_tools(
    deep_research: bool = False,
    lightning: bool = False,
) -> InformationTools:
    if lightning:
        search_config = SearchToolConfig(
            max_results=get_lightning_search_result_limit(),
            timeout_seconds=get_lightning_search_timeout_seconds(),
        )
    else:
        search_config = SearchToolConfig(max_results=get_search_result_limit(deep_research))

    return InformationTools(
        planner_tool=ResearchPlannerTool(),
        search_tool=SearchTool(search_config),
        fetch_url_tool=FetchURLTool(BaseToolConfig()),
        write_file_tool=WriteToFileTool(BaseToolConfig()),
    )


def _finalize_table(
    *,
    table: StructuredEntityTable,
    query: str,
    plan: ResearchPlan,
    source_registry: dict[str, dict[str, str | None]],
) -> tuple[StructuredEntityTable, StructuredEntityTable]:
    required_keys = get_required_column_keys(plan)
    partial_table = normalize_result(
        table,
        query=query,
        source_registry=source_registry,
        require_complete=False,
        required_column_keys=required_keys,
    )
    complete_table = normalize_result(
        table,
        query=query,
        source_registry=source_registry,
        require_complete=True,
        required_column_keys=required_keys,
    )
    return complete_table, partial_table


def _prune_incomplete_columns(
    table: StructuredEntityTable,
) -> StructuredEntityTable:
    if not table.rows:
        return table

    keep_keys = {"name"}
    for column in table.columns:
        key = column.key
        if key == "name":
            continue
        if all(
            key in row.cells
            and row.cells[key].value
            and row.cells[key].citations
            for row in table.rows
        ):
            keep_keys.add(key)

    pruned_columns = [column for column in table.columns if column.key in keep_keys]
    pruned_rows = [
        EntityRow(
            entity_id=row.entity_id,
            cells={key: cell for key, cell in row.cells.items() if key in keep_keys},
        )
        for row in table.rows
    ]
    return StructuredEntityTable(
        query=table.query,
        title=table.title,
        columns=pruned_columns,
        rows=pruned_rows,
        sources=list(table.sources),
    )


def _merge_batch_table(
    *,
    current_table: StructuredEntityTable,
    batch_table: StructuredEntityTable,
    source_registry: dict[str, dict[str, str | None]],
) -> StructuredEntityTable:
    if not current_table.rows:
        return batch_table
    return merge_tables(
        base=current_table,
        incoming=batch_table,
        source_registry=source_registry,
    )


def _chunk_records(
    records: list[dict[str, object]],
    *,
    chunk_size: int,
) -> list[list[dict[str, object]]]:
    size = max(1, chunk_size)
    return [records[index : index + size] for index in range(0, len(records), size)]


def _extract_table_from_chunks(
    *,
    query: str,
    plan: ResearchPlan,
    fetched_records: list[dict[str, object]],
    source_registry: dict[str, dict[str, str | None]],
    allow_partial: bool,
    required_only: bool,
    chunk_size: int,
    progress_callback: Callable[[str], None] | None = None,
    progress_label: str | None = None,
) -> StructuredEntityTable:
    chunked_records = _chunk_records(fetched_records, chunk_size=chunk_size)
    if not chunked_records:
        return _empty_table(query)

    merged_table = _empty_table(query)
    total_chunks = len(chunked_records)

    for chunk_index, chunk in enumerate(chunked_records, start=1):
        if progress_callback is not None and progress_label is not None and total_chunks > 1:
            progress_callback(f"{progress_label} ({chunk_index}/{total_chunks})")
        chunk_table = extract_table_from_frontier(
            query=query,
            plan=plan,
            fetched_records=chunk,
            source_registry=source_registry,
            allow_partial=allow_partial,
            required_only=required_only,
        )
        merged_table = _merge_batch_table(
            current_table=merged_table,
            batch_table=chunk_table,
            source_registry=source_registry,
        )

    return merged_table


def _update_frontier_selection(
    *,
    state: FrontierState,
    frontier_candidates: list[SearchCandidate],
) -> None:
    selected_urls = {candidate.url for candidate in frontier_candidates}
    for url in selected_urls:
        if url not in state.selected_frontier_urls:
            state.selected_frontier_urls.append(url)
    state.remaining_candidates = [
        candidate
        for candidate in state.remaining_candidates
        if candidate.url not in selected_urls
    ]


def _is_forbidden_fetch_error(error: object) -> bool:
    message = str(error or "").lower()
    return "403" in message or "forbidden" in message


def _all_forbidden_fetch_failures(
    *,
    frontier_candidates: list[SearchCandidate],
    batch_records: list[dict[str, object]],
    batch_failed_fetches: list[dict[str, object]],
) -> bool:
    return (
        not batch_records
        and len(batch_failed_fetches) == len(frontier_candidates)
        and bool(frontier_candidates)
        and all(_is_forbidden_fetch_error(item.get("error")) for item in batch_failed_fetches)
    )


def _run_frontier_batches(
    *,
    query: str,
    plan: ResearchPlan,
    budgets: PipelineBudgets,
    state: FrontierState,
    fetch_tool: FetchURLTool,
    progress_callback: Callable[[str], None],
    extraction_chunk_size: int,
) -> None:
    for batch_index in range(1, budgets.max_frontier_batches + 1):
        state.frontier_batch_count = batch_index
        while state.remaining_candidates:
            frontier_candidates = select_frontier_candidates(
                candidates=state.remaining_candidates,
                frontier_size=budgets.frontier_size,
            )
            if not frontier_candidates:
                return

            _update_frontier_selection(state=state, frontier_candidates=frontier_candidates)

            progress_callback(
                f"Fetching frontier batch {batch_index} ({len(frontier_candidates)} page(s))"
            )
            state.frontier_fetch_call_count += 1
            batch_records, batch_failed_fetches = fetch_frontier_records(
                frontier_candidates=frontier_candidates,
                fetch_tool=fetch_tool,
                focus_query=query,
                source_registry=state.source_registry,
                url_to_source_id=state.url_to_source_id,
                full_document=True,
            )
            state.fetched_records.extend(batch_records)
            state.failed_fetches.extend(batch_failed_fetches)

            if _all_forbidden_fetch_failures(
                frontier_candidates=frontier_candidates,
                batch_records=batch_records,
                batch_failed_fetches=batch_failed_fetches,
            ):
                if state.remaining_candidates:
                    progress_callback(
                        "All selected frontier pages were forbidden; trying the next candidates"
                    )
                    continue
                return

            if not batch_records:
                break

            progress_callback(f"Extracting slot-driven table from batch {batch_index}")
            batch_table = _extract_table_from_chunks(
                query=query,
                plan=plan,
                fetched_records=batch_records,
                source_registry=state.source_registry,
                allow_partial=True,
                required_only=False,
                chunk_size=extraction_chunk_size,
                progress_callback=progress_callback,
                progress_label=f"Extracting slot-driven table from batch {batch_index}",
            )
            state.working_table = _merge_batch_table(
                current_table=state.working_table,
                batch_table=batch_table,
                source_registry=state.source_registry,
            )
            break


def _build_result_meta(
    *,
    budgets: PipelineBudgets,
    deep_research: bool,
    completion_mode: str,
    research_plan: ResearchPlan,
    initial_queries: list[str],
    combined_query_history: list[dict[str, object]],
    total_results: int,
    retrieved_candidate_count: int,
    ranked_candidates: list[SearchCandidate],
    state: FrontierState,
    complete_table: StructuredEntityTable,
    recursive_research_meta: dict[str, object],
    execution_time_ms: int,
) -> dict[str, object]:
    fetch_calls = state.frontier_fetch_call_count + int(recursive_research_meta.get("fetch_calls", 0))
    return {
        "mode": budgets.mode,
        "deep_research": deep_research,
        "recursive_research": recursive_research_meta,
        "research_plan": research_plan.model_dump(),
        "initial_queries": initial_queries,
        "queries_run": combined_query_history,
        "total_results": total_results,
        "candidate_counts": {
            "retrieved": retrieved_candidate_count,
            "ranked": len(ranked_candidates),
            "frontier": len(state.selected_frontier_urls),
            "fetched": len(state.fetched_records),
        },
        "frontier_urls": state.selected_frontier_urls,
        "frontier_batches": state.frontier_batch_count,
        "failed_fetches": state.failed_fetches,
        "fetch_calls": fetch_calls,
        "fetch_limit": budgets.frontier_size,
        "rows_with_required_slots": len(complete_table.rows),
        "completion_mode": completion_mode,
        "execution_time_ms": execution_time_ms,
        "model": get_model_name(),
    }


def run_information_agent(
    information_request: str,
    deep_research: bool = False,
    recursive_research: bool = False,
    lightning: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    detail_callback: Callable[[str], None] | None = None,
) -> str:
    """Run the tiered slot-driven research pipeline and return a JSON string."""

    progress_callback = progress_callback or (lambda _message: None)
    detail_callback = detail_callback or (lambda _message: None)
    start_time = time.perf_counter()

    mode = _mode_name(deep_research=deep_research, lightning=lightning)
    progress_callback(f"Initializing {mode} pipeline")
    tools = create_information_tools(deep_research=deep_research, lightning=lightning)
    planner_tool = tools.planner_tool
    search_tool = tools.search_tool
    fetch_tool = tools.fetch_url_tool
    write_tool = tools.write_file_tool

    progress_callback("Building research plan")
    research_plan = planner_tool.run(
        ResearchPlannerInput(
            user_request=information_request,
            deep_research=deep_research,
            lightning=lightning,
        )
    )
    detail_callback(format_research_plan(research_plan))
    budgets = _build_pipeline_budgets(
        plan=research_plan,
        deep_research=deep_research,
        lightning=lightning,
    )
    extraction_chunk_size = _extraction_chunk_size(
        deep_research=deep_research,
        lightning=lightning,
    )

    initial_queries = build_initial_search_queries(
        research_plan,
        max_queries=budgets.initial_query_budget,
    )

    progress_callback("Retrieving candidate pages")
    candidates, queries_run, total_results = run_candidate_search(
        search_tool=search_tool,
        queries=initial_queries,
    )

    progress_callback("Reranking candidate pages")
    ranked_candidates = apply_semantic_rerank(
        candidates=candidates,
        plan=research_plan,
    )

    state = FrontierState(
        working_table=_empty_table(information_request),
        remaining_candidates=list(ranked_candidates),
    )

    _run_frontier_batches(
        query=information_request,
        plan=research_plan,
        budgets=budgets,
        state=state,
        fetch_tool=fetch_tool,
        progress_callback=progress_callback,
        extraction_chunk_size=extraction_chunk_size,
    )
    recursive_research_meta: dict[str, object] = {
        "enabled": recursive_research,
        "queries_run": [],
        "fetch_calls": 0,
    }
    if recursive_research and state.working_table.rows:
        progress_callback("Backfilling missing cells")
        state.working_table, recursive_research_meta = run_recursive_research(
            result=state.working_table,
            original_query=information_request,
            research_plan=research_plan,
            source_registry=state.source_registry,
            url_to_source_id=state.url_to_source_id,
            progress_callback=progress_callback,
            include_optional_slots=False,
        )

    complete_table, partial_table = _finalize_table(
        table=state.working_table,
        query=information_request,
        plan=research_plan,
        source_registry=state.source_registry,
    )
    if complete_table.rows:
        final_table = _prune_incomplete_columns(complete_table)
        completion_mode = "required_slots_complete"
    elif partial_table.rows:
        final_table = _prune_incomplete_columns(partial_table)
        completion_mode = "partial_rows_returned"
    else:
        final_table = _empty_table(information_request)
        completion_mode = "empty"

    progress_callback("Writing output files")
    json_payload = final_table.model_dump_json(indent=2)
    markdown_payload = render_markdown_document(final_table)
    json_write = write_tool.run(
        WriteToFileToolInputSchema(path=str(JSON_OUTPUT_PATH), content=json_payload)
    )
    markdown_write = write_tool.run(
        WriteToFileToolInputSchema(
            path=str(MARKDOWN_OUTPUT_PATH),
            content=markdown_payload,
        )
    )

    combined_query_history = _flatten_query_history(
        queries_run,
        recursive_research_meta,
    )
    execution_time_ms = int((time.perf_counter() - start_time) * 1000)
    status = "success" if final_table.rows or state.fetched_records else "error"

    result = InformationAgentOutput(
        status=status,
        json_file_path=json_write.absolute_path,
        markdown_file_path=markdown_write.absolute_path,
        result=final_table,
        meta=_build_result_meta(
            budgets=budgets,
            deep_research=deep_research,
            completion_mode=completion_mode,
            research_plan=research_plan,
            initial_queries=initial_queries,
            combined_query_history=combined_query_history,
            total_results=total_results,
            retrieved_candidate_count=len(candidates),
            ranked_candidates=ranked_candidates,
            state=state,
            complete_table=complete_table,
            recursive_research_meta=recursive_research_meta,
            execution_time_ms=execution_time_ms,
        ),
    )
    progress_callback("Completed")
    return result.model_dump_json(indent=2)


def resolve_deep_research_choice(explicit_choice: bool | None = None) -> bool:
    if explicit_choice is not None:
        return explicit_choice

    answer = input("Run in deep-research mode? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Discover entities for a topic query using the info agent."
    )
    parser.add_argument("query", help="Topic query to search and structure")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--deep-research",
        dest="deep_research",
        action="store_true",
        default=None,
        help="Run a broader and more exhaustive research pass",
    )
    group.add_argument(
        "--no-deep-research",
        dest="deep_research",
        action="store_false",
        help="Run the standard research pass",
    )
    parser.add_argument(
        "--recursive-research",
        action="store_true",
        help="Expand nice-to-have slots after required slots are filled",
    )
    parser.add_argument(
        "--lightning",
        action="store_true",
        help="Run a speed-optimized version of the slot-driven pipeline",
    )
    args = parser.parse_args()

    with CliProgressReporter() as progress:
        result = run_information_agent(
            args.query,
            deep_research=(
                False
                if args.lightning
                else resolve_deep_research_choice(args.deep_research)
            ),
            recursive_research=(False if args.lightning else args.recursive_research),
            lightning=args.lightning,
            progress_callback=progress.update,
            detail_callback=progress.log,
        )
        progress.complete("Completed")

    print(result)

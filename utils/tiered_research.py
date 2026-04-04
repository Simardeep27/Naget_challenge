from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from atomic_agents import BaseIOSchema
from pydantic import Field

from schema import (
    EntityRow,
    PreviewFetchRecord,
    ResearchPlan,
    ResearchSlot,
    SearchCandidate,
    SourceCitation,
    StructuredEntityTable,
    TableCell,
    TableColumn,
)
from tools.fetch_url import FetchURLTool, FetchURLToolInputSchema, PreviewURLToolInputSchema
from tools.web_search_tool import SearchTool, SearchToolInput
from utils.llm_utils import run_structured_generation
from utils.result_utils import collect_sources_from_rows, normalize_result
from utils.text_utils import compact_text, normalize_key, slugify, truncate_content

logger = logging.getLogger(__name__)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "top",
    "use",
    "with",
}
EVIDENCE_PATH_HINTS = (
    "about",
    "api",
    "availability",
    "benchmark",
    "compare",
    "comparison",
    "directory",
    "docs",
    "documentation",
    "faq",
    "fixtures",
    "guide",
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
)
NON_EVIDENCE_PATH_HINTS = (
    "account",
    "amp",
    "author",
    "category",
    "checkout",
    "cookie",
    "feed",
    "login",
    "privacy",
    "search",
    "share",
    "signup",
    "subscribe",
    "tag",
    "terms",
)
NON_EVIDENCE_TEXT_HINTS = (
    "privacy policy",
    "terms of service",
    "sign in",
    "log in",
    "create account",
    "cookie policy",
    "search results",
    "tag archive",
    "category archive",
)
STRONG_PREVIEW_TYPES = {
    "business_detail",
    "comparison",
    "directory",
    "live_status",
    "pricing",
    "profile",
    "reference",
}
WEAK_PREVIEW_TYPES = {"account", "landing", "policy", "search"}


class SemanticCandidateScore(BaseIOSchema):
    """One optional learned rerank score for a top candidate."""

    candidate_id: str = Field(..., description="Candidate identifier")
    score: float = Field(..., ge=0, le=1, description="Relevance score from 0 to 1")


class SemanticCandidateScoreList(BaseIOSchema):
    """Semantic rerank response for the top candidates."""

    scores: list[SemanticCandidateScore] = Field(default_factory=list)


def format_research_plan(plan: ResearchPlan) -> str:
    lines = [
        "Research plan:",
        f"Mode: {plan.research_depth}",
        f"Objective: {plan.objective}",
        f"Entity type: {plan.entity_type}",
    ]
    if plan.required_slots:
        lines.append("Required slots:")
        for slot in plan.required_slots:
            lines.append(f"- {slot.label}: {slot.description}")
    if plan.nice_to_have_slots:
        lines.append("Nice-to-have slots:")
        for slot in plan.nice_to_have_slots:
            lines.append(f"- {slot.label}: {slot.description}")
    if plan.constraints:
        lines.append("Constraints: " + ", ".join(plan.constraints))
    if plan.evidence_types:
        lines.append("Evidence types: " + ", ".join(plan.evidence_types))
    return "\n".join(lines)


def get_slot_list(plan: ResearchPlan, include_optional: bool = True) -> list[ResearchSlot]:
    slots = list(plan.required_slots)
    if include_optional:
        slots.extend(plan.nice_to_have_slots)
    deduped: list[ResearchSlot] = []
    seen: set[str] = set()
    for slot in slots:
        key = normalize_key(slot.key or slot.label)
        if not key or key in seen:
            continue
        slot.key = key
        deduped.append(slot)
        seen.add(key)
    return deduped


def get_required_column_keys(plan: ResearchPlan) -> set[str]:
    return {"name", *[normalize_key(slot.key or slot.label) for slot in plan.required_slots]}


def _tokenize(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", value.lower())
        if token not in STOPWORDS
    ]


def _normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _extract_domain(url: str) -> str:
    return (urlsplit(url).netloc or "").lower()


def _extract_path(url: str) -> str:
    return (urlsplit(url).path or "/").rstrip("/") or "/"


def _empty_table(query: str) -> StructuredEntityTable:
    return StructuredEntityTable(
        query=query,
        title=f"Discovered entities for {query}",
        columns=[],
        rows=[],
        sources=[],
    )


def _slot_payloads(slots: Iterable[ResearchSlot]) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    for slot in slots:
        key = normalize_key(slot.key or slot.label)
        if not key or key == "name":
            continue
        payloads.append(
            {
                "key": key,
                "label": slot.label,
                "description": slot.description,
            }
        )
    return payloads


def _preview_for_candidate(
    preview_map: dict[str, PreviewFetchRecord],
    candidate: SearchCandidate,
) -> PreviewFetchRecord | None:
    return preview_map.get(_normalize_url(candidate.url)) or preview_map.get(candidate.url)


def _build_fetched_record(
    *,
    source_id: str,
    title: str,
    url: str,
    snippet: str | None,
    content: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "snippet": snippet,
        "content": truncate_content(content),
    }


def _build_failed_fetch(
    *,
    source_id: str,
    title: str,
    url: str,
    error: object,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "error": error,
    }


def build_initial_search_queries(
    plan: ResearchPlan,
    *,
    max_queries: int,
) -> list[str]:
    slot_labels = [slot.label for slot in plan.required_slots if normalize_key(slot.key) != "name"]
    evidence = " ".join(plan.evidence_types[:2]).strip()
    constraints = " ".join(plan.constraints[:2]).strip()
    queries = [
        plan.user_query,
        compact_text(f"{plan.user_query} {evidence}") or plan.user_query,
        compact_text(f"{plan.user_query} {constraints}") or plan.user_query,
        compact_text(f"{plan.user_query} {' '.join(slot_labels[:2])}") or plan.user_query,
    ]
    deduped = [
        query
        for query in dict.fromkeys(query for query in queries if compact_text(query))
    ]
    return deduped[: max(1, max_queries)]


def build_slot_followup_query(
    *,
    plan: ResearchPlan,
    entity_name: str,
    slot: ResearchSlot,
    domain: str | None = None,
) -> str:
    slot_terms = " ".join([slot.label, *slot.search_hints[:3]]).strip()
    evidence_terms = " ".join(plan.evidence_types[:2]).strip()
    prefix = f'site:{domain} ' if domain else ""
    return compact_text(
        f'{prefix}"{entity_name}" {plan.user_query} {slot_terms} {evidence_terms}'
    ) or f'"{entity_name}" {slot.label}'


def run_candidate_search(
    *,
    search_tool: SearchTool,
    queries: list[str],
) -> tuple[list[SearchCandidate], list[dict[str, object]], int]:
    candidates: list[SearchCandidate] = []
    queries_run: list[dict[str, object]] = []
    total_results = 0

    for query in queries:
        output = search_tool.run(SearchToolInput(value=query))
        if output.error:
            queries_run.append({"query": query, "results": 0, "error": output.error})
            continue

        normalized_count = 0
        for rank, item in enumerate(output.results, start=1):
            url = compact_text(str(item.get("href") or item.get("url") or ""))
            title = compact_text(str(item.get("title") or ""))
            snippet = compact_text(str(item.get("body") or item.get("snippet") or ""))
            if not url or not title:
                continue
            candidate_id = f"cand_{len(candidates) + 1:03d}"
            candidates.append(
                SearchCandidate(
                    candidate_id=candidate_id,
                    query=query,
                    seen_in_queries=[query],
                    title=title,
                    snippet=snippet,
                    url=url,
                    domain=_extract_domain(url),
                    path=_extract_path(url),
                    search_rank=rank,
                )
            )
            normalized_count += 1

        queries_run.append({"query": query, "results": normalized_count})
        total_results += normalized_count

    return candidates, queries_run, total_results


def _search_rank_score(rank: int) -> float:
    return 1.0 / (1.0 + math.log2(max(rank, 1)))


def _domain_prior(candidate: SearchCandidate) -> float:
    domain = candidate.domain.lower()
    path = candidate.path.lower()
    metadata = " ".join(
        [candidate.title.lower(), (candidate.snippet or "").lower(), path]
    )
    path_segments = [segment for segment in path.split("/") if segment]
    domain_labels = [label for label in domain.split(".") if label]

    score = 0.5

    if any(hint in path for hint in EVIDENCE_PATH_HINTS):
        score += 0.08
    if any(hint in path for hint in NON_EVIDENCE_PATH_HINTS):
        score -= 0.18
    if any(hint in metadata for hint in NON_EVIDENCE_TEXT_HINTS):
        score -= 0.12

    if len(path_segments) <= 3:
        score += 0.05
    elif len(path_segments) >= 7:
        score -= 0.08

    if len(domain_labels) <= 3:
        score += 0.03
    elif len(domain_labels) >= 5:
        score -= 0.05

    if "?" in candidate.url or "&" in candidate.url:
        score -= 0.04
    if path in {"", "/"}:
        score += 0.02

    return max(0.0, min(1.0, score))


def _title_signature(title: str) -> tuple[str, ...]:
    return tuple(sorted(set(_tokenize(title)) - STOPWORDS))


def cluster_and_score_candidates(
    candidates: list[SearchCandidate],
    plan: ResearchPlan,
) -> list[SearchCandidate]:
    clustered: dict[str, SearchCandidate] = {}
    title_clusters: dict[tuple[str, str], str] = {}

    for candidate in candidates:
        normalized_url = _normalize_url(candidate.url)
        title_signature = _title_signature(candidate.title)
        cluster_lookup_key = normalized_url
        if normalized_url not in clustered and title_signature:
            title_key = (candidate.domain, " ".join(title_signature[:8]))
            if title_key in title_clusters:
                cluster_lookup_key = title_clusters[title_key]
            else:
                title_clusters[title_key] = normalized_url

        candidate.cluster_id = slugify(cluster_lookup_key)
        candidate.search_rank_score = _search_rank_score(candidate.search_rank)
        candidate.domain_prior = _domain_prior(candidate)
        candidate.final_score = (
            0.55 * candidate.search_rank_score
            + 0.45 * candidate.domain_prior
        )

        existing = clustered.get(cluster_lookup_key)
        if existing is None or candidate.final_score > existing.final_score:
            clustered[cluster_lookup_key] = candidate
        else:
            for query in candidate.seen_in_queries:
                if query not in existing.seen_in_queries:
                    existing.seen_in_queries.append(query)
            existing.search_rank = min(existing.search_rank, candidate.search_rank)
            existing.search_rank_score = max(existing.search_rank_score, candidate.search_rank_score)
            existing.domain_prior = max(existing.domain_prior, candidate.domain_prior)
            existing.final_score = max(existing.final_score, candidate.final_score)

    return sorted(clustered.values(), key=lambda item: item.final_score, reverse=True)


def apply_semantic_rerank(
    *,
    candidates: list[SearchCandidate],
    plan: ResearchPlan,
    top_k: int = 10,
) -> list[SearchCandidate]:
    if not candidates:
        return []

    candidate_slice = candidates[: min(top_k, len(candidates))]
    system_prompt = """
You score search candidates for a slot-driven research agent.
Use only the candidate metadata.
Prefer candidates that are likely to fill the required slots and satisfy the query constraints.
Return valid JSON matching the provided schema.
""".strip()
    user_prompt = (
        f"Objective:\n{plan.objective}\n\n"
        f"Required slots:\n{[slot.label for slot in plan.required_slots]}\n\n"
        f"Constraints:\n{plan.constraints}\n\n"
        f"Evidence types:\n{plan.evidence_types}\n\n"
        f"Candidates:\n"
        + "\n".join(
            [
                (
                    f"- {candidate.candidate_id}: title={candidate.title!r}; "
                    f"snippet={candidate.snippet!r}; url={candidate.url!r}; "
                    f"domain={candidate.domain!r}; path={candidate.path!r}"
                )
                for candidate in candidate_slice
            ]
        )
    )

    try:
        result: SemanticCandidateScoreList = run_structured_generation(
            response_schema=SemanticCandidateScoreList,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception:
        return candidates

    score_map = {item.candidate_id: item.score for item in result.scores}
    for candidate in candidate_slice:
        semantic_score = max(0.0, min(1.0, score_map.get(candidate.candidate_id, 0.0)))
        candidate.semantic_rerank_score = semantic_score
        candidate.final_score = (0.65 * candidate.final_score) + (0.35 * semantic_score)

    return sorted(candidates, key=lambda item: item.final_score, reverse=True)


def _preview_relevance(preview: PreviewFetchRecord) -> float:
    if preview.error:
        return 0.0

    score = 0.35
    if preview.page_type in STRONG_PREVIEW_TYPES:
        score += 0.15
    if preview.page_type in WEAK_PREVIEW_TYPES:
        score -= 0.12
    if preview.toc_hints:
        score += min(0.15, 0.03 * len(preview.toc_hints))
    if preview.headings:
        score += min(0.08, 0.02 * len(preview.headings))
    if preview.meta_description:
        score += 0.05
    if preview.preview_text and len(preview.preview_text.split()) >= 35:
        score += 0.08
    return max(0.0, min(1.0, score))


def apply_preview_semantic_rerank(
    *,
    candidates: list[SearchCandidate],
    preview_map: dict[str, PreviewFetchRecord],
    plan: ResearchPlan,
    top_k: int,
) -> None:
    candidate_slice = [
        candidate
        for candidate in candidates[: min(top_k, len(candidates))]
        if preview_map.get(_normalize_url(candidate.url)) or preview_map.get(candidate.url)
    ]
    if not candidate_slice:
        return

    system_prompt = """
You score preview-stage search candidates for a slot-driven research agent.
Use only the preview metadata that was extracted from the page head and first content chunk.
Prefer pages that are likely to provide concrete evidence for the requested slots.
Return valid JSON matching the provided schema.
""".strip()
    user_prompt = (
        f"Objective:\n{plan.objective}\n\n"
        f"Required slots:\n{[slot.label for slot in plan.required_slots]}\n\n"
        f"Constraints:\n{plan.constraints}\n\n"
        f"Evidence types:\n{plan.evidence_types}\n\n"
        f"Preview candidates:\n"
        + "\n".join(
            [
                (
                    f"- {candidate.candidate_id}: title={preview.title!r}; "
                    f"meta_description={preview.meta_description!r}; "
                    f"headings={preview.headings!r}; toc_hints={preview.toc_hints!r}; "
                    f"page_type={preview.page_type!r}; url={candidate.url!r}"
                )
                for candidate in candidate_slice
                for preview in [
                    preview_map.get(_normalize_url(candidate.url))
                    or preview_map.get(candidate.url)
                ]
                if preview is not None
            ]
        )
    )

    try:
        result: SemanticCandidateScoreList = run_structured_generation(
            response_schema=SemanticCandidateScoreList,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception:
        return

    score_map = {item.candidate_id: item.score for item in result.scores}
    for candidate in candidate_slice:
        semantic_score = max(0.0, min(1.0, score_map.get(candidate.candidate_id, 0.0)))
        if semantic_score <= 0:
            continue
        candidate.preview_relevance = (
            0.45 * candidate.preview_relevance + 0.55 * semantic_score
        )


def preview_candidates(
    *,
    fetch_tool: FetchURLTool,
    candidates: list[SearchCandidate],
    preview_limit: int,
    plan: ResearchPlan,
) -> dict[str, PreviewFetchRecord]:
    selected = candidates[: max(0, preview_limit)]
    if not selected:
        return {}

    preview_output = fetch_tool.preview(
        PreviewURLToolInputSchema(value=[candidate.url for candidate in selected])
    )
    preview_map = {record.url: record for record in preview_output.result}
    for candidate in selected:
        preview = _preview_for_candidate(preview_map, candidate)
        if not preview:
            continue
        candidate.preview_relevance = _preview_relevance(preview)

    apply_preview_semantic_rerank(
        candidates=selected,
        preview_map=preview_map,
        plan=plan,
        top_k=preview_limit,
    )

    for candidate in selected:
        preview = _preview_for_candidate(preview_map, candidate)
        if not preview:
            continue
        candidate.final_score = (
            0.65 * candidate.final_score
            + 0.35 * candidate.preview_relevance
        )
    return preview_map


def select_frontier_candidates(
    *,
    candidates: list[SearchCandidate],
    frontier_size: int,
) -> list[SearchCandidate]:
    frontier: list[SearchCandidate] = []
    seen_domains: set[str] = set()
    ranked_candidates = sorted(candidates, key=lambda item: item.final_score, reverse=True)

    for candidate in ranked_candidates:
        if candidate.domain not in seen_domains or len(ranked_candidates) <= frontier_size:
            frontier.append(candidate)
            seen_domains.add(candidate.domain)
        if len(frontier) >= frontier_size:
            return frontier

    for candidate in ranked_candidates:
        if candidate in frontier:
            continue
        frontier.append(candidate)
        if len(frontier) >= frontier_size:
            break

    return frontier[:frontier_size]


def register_candidate_sources(
    *,
    candidates: Iterable[SearchCandidate],
    source_registry: dict[str, dict[str, str | None]],
    url_to_source_id: dict[str, str],
) -> None:
    for candidate in candidates:
        source_id = url_to_source_id.get(candidate.url)
        if not source_id:
            source_id = f"src_{len(url_to_source_id) + 1:03d}"
            url_to_source_id[candidate.url] = source_id
        source_registry[source_id] = {
            "source_id": source_id,
            "title": candidate.title,
            "url": candidate.url,
            "snippet": candidate.snippet,
        }


def fetch_frontier_records(
    *,
    frontier_candidates: list[SearchCandidate],
    fetch_tool: FetchURLTool,
    focus_query: str,
    source_registry: dict[str, dict[str, str | None]],
    url_to_source_id: dict[str, str],
    full_document: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    register_candidate_sources(
        candidates=frontier_candidates,
        source_registry=source_registry,
        url_to_source_id=url_to_source_id,
    )
    urls_to_fetch = [candidate.url for candidate in frontier_candidates]
    if not urls_to_fetch:
        return [], []

    fetch_output = fetch_tool.run(
        FetchURLToolInputSchema(
            value=urls_to_fetch,
            focus_query=focus_query,
            full_document=full_document,
        )
    )
    fetched_records: list[dict[str, object]] = []
    failed_fetches: list[dict[str, object]] = []

    for candidate, content in zip(frontier_candidates, fetch_output.result):
        source_id = url_to_source_id.get(candidate.url)
        if not source_id:
            continue
        metadata = source_registry[source_id]
        if fetch_tool.is_error_result(content):
            failed_fetches.append(
                _build_failed_fetch(
                    source_id=source_id,
                    title=metadata.get("title") or candidate.title,
                    url=candidate.url,
                    error=content,
                )
            )
            continue
        fetched_records.append(
            _build_fetched_record(
                source_id=source_id,
                title=metadata.get("title") or candidate.title,
                url=candidate.url,
                snippet=metadata.get("snippet"),
                content=content,
            )
        )

    return fetched_records, failed_fetches


def extract_table_from_frontier(
    *,
    query: str,
    plan: ResearchPlan,
    fetched_records: list[dict[str, object]],
    source_registry: dict[str, dict[str, str | None]],
    allow_partial: bool,
    required_only: bool = False,
) -> StructuredEntityTable:
    if not fetched_records:
        return _empty_table(query)

    required_slots = _slot_payloads(plan.required_slots)
    nice_to_have_slots = [] if required_only else _slot_payloads(plan.nice_to_have_slots)
    completeness_instruction = (
        "Every returned row must include all required slots with citations."
        if not allow_partial
        else "Prefer fully populated rows, but partial rows are acceptable if a required slot still needs another source."
    )
    optional_instruction = (
        "Do not include optional columns in the output."
        if required_only
        else "Include optional columns only when they are directly supported."
    )

    system_prompt = f"""
You extract a comparison table from frontier pages for a slot-driven research agent.
Return valid JSON matching the StructuredEntityTable schema.
Use only the fields requested by the research plan.
Do not create generic page summaries or extra columns beyond the requested slots.
Always include a "name" column.
Required slots: {[slot["key"] for slot in required_slots]}.
Optional slots: {[slot["key"] for slot in nice_to_have_slots]}.
{completeness_instruction}
Return as many well-supported rows as the evidence supports.
{optional_instruction}
Prefer more rows with complete required slots over fewer rows with extra optional detail.
If a fetched page contains a long ranked or directory-style list, extract multiple distinct entities from it rather than stopping after one or two.
Every populated cell must include at least one citation with exact source_id, source_title, source_url, and a short verbatim quote copied from the fetched content.
Only include entities that clearly satisfy the query constraints.
""".strip()
    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Objective:\n{plan.objective}\n\n"
        f"Constraints:\n{plan.constraints}\n\n"
        f"Required slots:\n{required_slots}\n\n"
        f"Nice-to-have slots:\n{nice_to_have_slots}\n\n"
        f"Evidence types:\n{plan.evidence_types}\n\n"
        f"Fetched frontier pages:\n{fetched_records}\n"
    )
    cache_key = hashlib.sha256(
        f"{system_prompt}\n\n{user_prompt}".encode("utf-8")
    ).hexdigest()

    try:
        extracted_table: StructuredEntityTable = run_structured_generation(
            response_schema=StructuredEntityTable,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            vertex_cache_key=f"extract_table:{cache_key}",
        )
    except Exception:
        logger.exception("Structured extraction failed for query %r", query)
        return StructuredEntityTable(
            query=query,
            title=_empty_table(query).title,
            columns=[TableColumn(key="name", label="Name", description="Name of the entity")],
            rows=[],
            sources=[],
        )

    return normalize_result(
        extracted_table,
        query=query,
        source_registry=source_registry,
        require_complete=False,
        required_column_keys=get_required_column_keys(plan),
    )


def merge_tables(
    *,
    base: StructuredEntityTable,
    incoming: StructuredEntityTable,
    source_registry: dict[str, dict[str, str | None]],
) -> StructuredEntityTable:
    column_map: dict[str, TableColumn] = {
        normalize_key(column.key): column for column in base.columns
    }
    for column in incoming.columns:
        key = normalize_key(column.key or column.label)
        if not key or key in column_map:
            continue
        column_map[key] = TableColumn(
            key=key,
            label=column.label,
            description=column.description,
        )

    row_map: dict[str, EntityRow] = {}
    for table in (base, incoming):
        for row in table.rows:
            name_cell = row.cells.get("name")
            entity_key = normalize_key((name_cell.value if name_cell else None) or row.entity_id)
            if not entity_key:
                continue
            existing = row_map.get(entity_key)
            if existing is None:
                cloned_cells = {
                    key: TableCell(value=cell.value, citations=list(cell.citations))
                    for key, cell in row.cells.items()
                }
                row_map[entity_key] = EntityRow(entity_id=entity_key, cells=cloned_cells)
                continue

            for key, cell in row.cells.items():
                current = existing.cells.get(key)
                if current is None or not compact_text(current.value):
                    existing.cells[key] = TableCell(
                        value=cell.value,
                        citations=list(cell.citations),
                    )
                    continue
                if normalize_key(str(current.value or "")) == normalize_key(str(cell.value or "")):
                    citation_pairs = {
                        (citation.source_id, citation.quote) for citation in current.citations
                    }
                    for citation in cell.citations:
                        if (citation.source_id, citation.quote) in citation_pairs:
                            continue
                        current.citations.append(citation)

    merged = StructuredEntityTable(
        query=base.query or incoming.query,
        title=base.title or incoming.title,
        columns=list(column_map.values()),
        rows=list(row_map.values()),
        sources=[],
    )
    merged.sources = collect_sources_from_rows(merged.rows, source_registry)
    return merged


def fill_slot_from_records(
    *,
    query: str,
    plan: ResearchPlan,
    entity_name: str,
    slot: ResearchSlot,
    records: list[dict[str, object]],
) -> TableCell | None:
    if not records:
        return None

    class SlotOnlyCell(BaseIOSchema):
        """One slot-fill result for a single entity."""

        value: str | None = Field(default=None)
        citations: list[SourceCitation] = Field(default_factory=list)

    system_prompt = """
You fill one missing slot for one entity.
Return valid JSON matching the provided schema.
Use only the fetched records below.
If the slot cannot be grounded, return value=null and an empty citations list.
Every citation must include exact source_id, source_title, source_url, and a short verbatim quote from the fetched content.
""".strip()
    user_prompt = (
        f"Original query:\n{query}\n\n"
        f"Objective:\n{plan.objective}\n\n"
        f"Entity name:\n{entity_name}\n\n"
        f"Slot to fill:\n{slot.model_dump()}\n\n"
        f"Fetched records:\n{records}\n"
    )
    try:
        result: SlotOnlyCell = run_structured_generation(
            response_schema=SlotOnlyCell,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception:
        return None

    if not compact_text(result.value) or not result.citations:
        return None
    return TableCell(value=compact_text(result.value), citations=result.citations)

from typing import Literal

from atomic_agents import BaseIOSchema
from pydantic import Field


class SourceCitation(BaseIOSchema):
    """Traceability metadata for a populated table cell."""

    source_id: str = Field(..., description="Stable source identifier, e.g. src_001")
    source_title: str = Field(..., description="Human-readable title of the source")
    source_url: str = Field(..., description="Canonical source URL")
    quote: str = Field(
        ..., description="Short verbatim quote from the fetched source content"
    )


class TableCell(BaseIOSchema):
    """A table cell value plus the citations backing it."""

    value: str | None = Field(None, description="Cell value")
    citations: list[SourceCitation] = Field(
        default_factory=list,
        description="One or more citations supporting this value",
    )


class TableColumn(BaseIOSchema):
    """A column in the final discovered-entities table."""

    key: str = Field(..., description="Snake_case column identifier")
    label: str = Field(..., description="Display label for the column")
    description: str = Field(..., description="What the column represents")


class EntityRow(BaseIOSchema):
    """A discovered entity represented as row cells keyed by column key."""

    entity_id: str = Field(..., description="Stable row identifier")
    cells: dict[str, TableCell] = Field(
        default_factory=dict,
        description="Cell values keyed by column key",
    )


class SourceRecord(BaseIOSchema):
    """A source that contributed evidence to the final result."""

    source_id: str = Field(..., description="Stable source identifier")
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    snippet: str | None = Field(None, description="Search snippet if available")


class StructuredEntityTable(BaseIOSchema):
    """The final structured output for a topic query."""

    query: str = Field(..., description="Original topic query")
    title: str = Field(..., description="Descriptive title for the entity table")
    columns: list[TableColumn] = Field(
        default_factory=list,
        description="Ordered columns for the final table",
    )
    rows: list[EntityRow] = Field(
        default_factory=list,
        description="Discovered entity rows",
    )
    sources: list[SourceRecord] = Field(
        default_factory=list,
        description="Sources cited by the final table",
    )


class ResearchSlot(BaseIOSchema):
    """One comparison slot the research pipeline should try to fill."""

    key: str = Field(..., description="Snake_case slot identifier")
    label: str = Field(..., description="Display label for the slot")
    description: str = Field(..., description="What the slot captures")
    required: bool = Field(
        default=True,
        description="Whether the slot is required for a row to count as complete",
    )
    search_hints: list[str] = Field(
        default_factory=list,
        description="Helpful terms for retrieval and extraction prompts",
    )


class ResearchPlan(BaseIOSchema):
    """Slot-driven research plan derived from the user's query."""

    user_query: str = Field(..., description="Original user query")
    research_depth: Literal["standard", "deep", "lightning"] = Field(
        ...,
        description="Selected research mode",
    )
    objective: str = Field(..., description="Core comparison objective")
    entity_type: str = Field(..., description="Likely entity type to compare")
    required_slots: list[ResearchSlot] = Field(
        default_factory=list,
        description="Must-have comparison slots",
    )
    nice_to_have_slots: list[ResearchSlot] = Field(
        default_factory=list,
        description="Helpful but optional comparison slots",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Important filters or constraints implied by the query",
    )
    evidence_types: list[str] = Field(
        default_factory=list,
        description="Evidence categories to prioritize during retrieval",
    )


class SearchCandidate(BaseIOSchema):
    """Metadata-only search candidate used before preview and full fetch."""

    candidate_id: str = Field(..., description="Stable candidate identifier")
    query: str = Field(..., description="Search query that produced this candidate")
    seen_in_queries: list[str] = Field(
        default_factory=list,
        description="All queries that surfaced this candidate",
    )
    title: str = Field(..., description="Search result title")
    snippet: str | None = Field(None, description="Search result snippet")
    url: str = Field(..., description="Candidate URL")
    domain: str = Field(..., description="Candidate domain")
    path: str = Field(..., description="Candidate path")
    search_rank: int = Field(..., description="Rank within its originating query")
    cluster_id: str | None = Field(
        default=None,
        description="Duplicate cluster identifier after deduplication",
    )
    search_rank_score: float = Field(
        default=0.0,
        description="Normalized score based on search rank",
    )
    domain_prior: float = Field(
        default=0.0,
        description="Heuristic domain-quality prior",
    )
    semantic_rerank_score: float = Field(
        default=0.0,
        description="Optional learned rerank signal on the top candidates",
    )
    preview_relevance: float = Field(
        default=0.0,
        description="Preview-stage relevance score before full fetch",
    )
    final_score: float = Field(
        default=0.0,
        description="Combined score used to select frontier pages",
    )


class PreviewFetchRecord(BaseIOSchema):
    """Lightweight preview for deciding whether a page deserves a full fetch."""

    url: str = Field(..., description="Requested URL")
    final_url: str = Field(..., description="Resolved final URL")
    status_code: int | None = Field(None, description="HTTP status code if available")
    content_type: str | None = Field(
        None, description="Content-Type header if available"
    )
    title: str | None = Field(None, description="Preview title")
    meta_description: str | None = Field(
        None, description="Meta description if found"
    )
    headings: list[str] = Field(
        default_factory=list,
        description="Previewed headings from the page",
    )
    json_ld_types: list[str] = Field(
        default_factory=list,
        description="Detected JSON-LD @type values",
    )
    toc_hints: list[str] = Field(
        default_factory=list,
        description="Detected navigational hints such as menu, schedule, pricing, docs, or reviews",
    )
    preview_text: str | None = Field(
        None,
        description="Short extracted preview text from the initial HTML bytes",
    )
    page_type: str | None = Field(
        None,
        description="Heuristic page category such as profile, comparison, live_status, pricing, or landing",
    )
    error: str | None = Field(
        default=None,
        description="Handled preview error if the page could not be previewed",
    )


class SlotGapAction(BaseIOSchema):
    """Next action to close one missing slot for one entity."""

    entity_id: str = Field(..., description="Target entity identifier")
    entity_name: str = Field(..., description="Target entity name")
    slot_key: str = Field(..., description="Missing slot key")
    required: bool = Field(..., description="Whether the slot is required")
    action_type: Literal["infer", "in_domain_expand", "new_web_search", "skip"] = (
        Field(..., description="Cheapest next action for this slot")
    )
    rationale: str = Field(..., description="Why this action was chosen")
    domain: str | None = Field(
        default=None,
        description="Domain to expand inside when action_type=in_domain_expand",
    )
    query: str | None = Field(
        default=None,
        description="Search query to run when a search action is needed",
    )


class InformationAgentOutput(BaseIOSchema):
    """Output payload for the information agent."""

    status: str = Field(..., description="success or error")
    json_file_path: str = Field(
        ..., description="Path to the written JSON output file"
    )
    markdown_file_path: str = Field(
        ..., description="Path to the written markdown output file"
    )
    result: StructuredEntityTable = Field(
        ..., description="Structured entity-discovery result"
    )
    meta: dict = Field(default_factory=dict, description="Execution metadata")

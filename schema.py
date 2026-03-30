from typing import Literal, Optional

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


class InformationAgentInput(BaseIOSchema):
    """Input payload for the information agent."""

    information_request: str = Field(
        ..., description="The topic query to research and structure"
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


class AgentAction(BaseIOSchema):
    """
    The agent's next step decision, returned on every agent.run() iteration.
    The orchestrator reads action_type and dispatches to the appropriate tool.
    """

    reasoning: str = Field(
        ..., description="Brief explanation of why this action is taken"
    )
    action_type: Literal["search_web", "fetch_url", "finish"] = Field(
        ...,
        description="Which action to perform next",
    )
    search_query: Optional[str] = Field(
        None, description="Search query string (required for search_web)"
    )
    urls_to_fetch: Optional[list[str]] = Field(
        None,
        description=(
            "URLs to fetch (required for fetch_url). "
            "Only include URLs whose search snippets indicate relevance."
        ),
    )
    final_result: Optional[StructuredEntityTable] = Field(
        None,
        description=(
            "Final structured entity table (required for finish). "
            "Each populated cell must include citations."
        ),
    )


class ToolResult(BaseIOSchema):
    """Tool execution result injected into agent.history between loop iterations."""

    tool_name: str = Field(..., description="Name of the tool that was executed")
    result: str = Field(..., description="JSON-encoded output from the tool")

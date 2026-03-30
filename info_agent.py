import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import instructor
from atomic_agents import AgentConfig, AtomicAgent, BaseToolConfig
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from dotenv import load_dotenv
from openai import OpenAI

from prompts.information_agent_prompt import get_information_agent_system_prompt
from schema import (
    AgentAction,
    EntityRow,
    InformationAgentInput,
    InformationAgentOutput,
    IntentDecomposition,
    SourceCitation,
    SourceRecord,
    StructuredEntityTable,
    TableCell,
    TableColumn,
    ToolResult,
)
from tools.fetch_url import FetchURLTool, FetchURLToolInputSchema
from tools.intent_classificaiton import (
    IntentClassificationInput,
    IntentClassificationTool,
    IntentClassificationToolConfig,
)
from tools.web_search_tool import SearchTool, SearchToolConfig, SearchToolInput
from tools.write_to_file import WriteToFileTool, WriteToFileToolInputSchema

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

MAX_CONTENT_CHARS = 12000
OUTPUT_DIR = Path("output/information_agent")
JSON_OUTPUT_PATH = OUTPUT_DIR / "latest_entities.json"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "latest_entities.md"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_api_key() -> str:
    return os.getenv("PROVIDER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def _get_base_url() -> str | None:
    return os.getenv("PROVIDER_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def _get_model_name() -> str:
    default_model = "gemini-2.5-flash" if _use_vertex_ai() else "gpt-4.1-mini"
    return (
        os.getenv("INFORMATION_AGENT_MODEL")
        or os.getenv("PROVIDER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or default_model
    )


def _use_vertex_ai() -> bool:
    return _env_flag("GOOGLE_GENAI_USE_VERTEXAI", default=False)


def _get_vertex_project() -> str:
    return os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()


def _get_vertex_location() -> str:
    return os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"


def _get_search_result_limit(deep_research: bool) -> int:
    return 12 if deep_research else 8


def _get_fetch_limit(deep_research: bool) -> int:
    return 12 if deep_research else 6


def _get_loop_limit(deep_research: bool) -> int:
    return 24 if deep_research else 16


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "entity"


def _compact_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def _truncate_content(value: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    trimmed = value[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{trimmed}\n\n[...truncated]"


def _escape_markdown_cell(value: str) -> str:
    compact = " ".join(value.split())
    if len(compact) > 100:
        compact = f"{compact[:97].rstrip()}..."
    return compact.replace("|", "\\|")


def _render_table_markdown(result: StructuredEntityTable) -> str:
    if not result.rows:
        return "_No entities extracted._"

    headers = [column.label for column in result.columns]
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]

    for row in result.rows:
        normalized_cells = {
            _normalize_key(key): cell for key, cell in (row.cells or {}).items()
        }
        values = []
        for column in result.columns:
            cell = normalized_cells.get(column.key)
            values.append(_escape_markdown_cell(cell.value) if cell and cell.value else "")
        lines.append(f"| {' | '.join(values)} |")

    return "\n".join(lines)


def _render_markdown_document(result: StructuredEntityTable) -> str:
    lines = [
        f"# {result.title}",
        "",
        f"Query: `{result.query}`",
        "",
        "## Entity Table",
        _render_table_markdown(result),
        "",
        "## Sources",
    ]

    for source in result.sources:
        lines.append(f"- `{source.source_id}`: [{source.title}]({source.url})")

    return "\n".join(lines)


def _normalize_result(
    result: StructuredEntityTable,
    query: str,
    source_registry: dict[str, dict[str, str | None]],
) -> StructuredEntityTable:
    normalized_columns: list[TableColumn] = []
    seen_column_keys: set[str] = set()

    for column in result.columns:
        key = _normalize_key(column.key or column.label)
        if not key or key in seen_column_keys:
            continue
        normalized_columns.append(
            TableColumn(
                key=key,
                label=_compact_text(column.label) or key.replace("_", " ").title(),
                description=_compact_text(column.description) or "",
            )
        )
        seen_column_keys.add(key)

    if "name" not in seen_column_keys:
        normalized_columns.insert(
            0,
            TableColumn(
                key="name",
                label="Name",
                description="Name of the discovered entity",
            ),
        )
        seen_column_keys.add("name")

    normalized_rows: list[EntityRow] = []
    used_source_ids: set[str] = set()

    for row in result.rows:
        raw_cells = {_normalize_key(key): cell for key, cell in (row.cells or {}).items()}
        cleaned_cells: dict[str, TableCell] = {}

        for column in normalized_columns:
            cell = raw_cells.get(column.key)
            if not cell or not _compact_text(cell.value):
                continue

            citations: list[SourceCitation] = []
            for citation in cell.citations or []:
                source_id = _compact_text(citation.source_id)
                source_meta = source_registry.get(source_id or "", {})
                source_title = _compact_text(citation.source_title) or _compact_text(
                    source_meta.get("title")
                )
                source_url = _compact_text(citation.source_url) or _compact_text(
                    source_meta.get("url")
                )
                quote = _compact_text(citation.quote)

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
                used_source_ids.add(source_id)

            if not citations:
                continue

            cleaned_cells[column.key] = TableCell(
                value=_compact_text(cell.value),
                citations=citations,
            )

        name_cell = cleaned_cells.get("name")
        if not name_cell or not name_cell.value:
            continue

        normalized_rows.append(
            EntityRow(
                entity_id=_slugify(name_cell.value),
                cells=cleaned_cells,
            )
        )

    normalized_sources = [
        SourceRecord(
            source_id=source_id,
            title=_compact_text(source_registry[source_id].get("title")) or source_id,
            url=_compact_text(source_registry[source_id].get("url")) or "",
            snippet=_compact_text(source_registry[source_id].get("snippet")),
        )
        for source_id in sorted(used_source_ids)
        if source_id in source_registry and _compact_text(source_registry[source_id].get("url"))
    ]

    return StructuredEntityTable(
        query=query,
        title=_compact_text(result.title) or f"Discovered entities for {query}",
        columns=normalized_columns,
        rows=normalized_rows,
        sources=normalized_sources,
    )


class SimpleChatHistory:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: Any) -> None:
        if hasattr(content, "model_dump_json"):
            serialized = content.model_dump_json(indent=2)
        elif isinstance(content, str):
            serialized = content
        else:
            serialized = json.dumps(content, ensure_ascii=False, indent=2)

        self.messages.append({"role": role, "content": serialized})


class VertexInformationAgent:
    """A minimal agent wrapper that preserves the current loop for Vertex AI."""

    def __init__(self, model: str, system_prompt: str, project: str, location: str):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version="v1"),
        )
        self.model = model
        self.system_prompt = system_prompt
        self.history = SimpleChatHistory()
        self.initial_input: InformationAgentInput | None = None

    def _build_prompt(self) -> str:
        if self.initial_input is None:
            raise ValueError("InformationAgentInput is required for the first run.")

        parts = [
            "Original topic query:",
            self.initial_input.information_request,
            f"Deep research requested: {self.initial_input.deep_research}",
        ]

        if self.initial_input.intent_decomposition is not None:
            parts.extend(
                [
                    "Intent decomposition:",
                    self.initial_input.intent_decomposition.model_dump_json(indent=2),
                ]
            )

        if self.history.messages:
            parts.append(
                "Tool execution history so far. Use these results as grounding for the next AgentAction."
            )
            for index, message in enumerate(self.history.messages, start=1):
                parts.append(
                    f"History item {index} ({message['role']}):\n{message['content']}"
                )

        return "\n\n".join(parts)

    def run(self, payload: InformationAgentInput | None = None) -> AgentAction:
        if payload is not None:
            self.initial_input = payload

        response = self.client.models.generate_content(
            model=self.model,
            contents=self._build_prompt(),
            config=self._types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0,
                response_mime_type="application/json",
                response_schema=AgentAction,
            ),
        )

        usage = getattr(response, "usage_metadata", None)
        if usage:
            logger.debug(
                "tokens - prompt=%s, completion=%s, total=%s",
                getattr(usage, "prompt_token_count", 0),
                getattr(usage, "candidates_token_count", 0),
                getattr(usage, "total_token_count", 0),
            )

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, AgentAction):
            return parsed
        if parsed is not None:
            return AgentAction.model_validate(parsed)
        if response.text:
            return AgentAction.model_validate_json(response.text)

        raise ValueError("Vertex AI returned no parseable AgentAction.")


def create_information_agent(deep_research: bool = False):
    """Create the agent and tools for the agentic research loop."""
    if _use_vertex_ai():
        project = _get_vertex_project()
        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT must be set when GOOGLE_GENAI_USE_VERTEXAI=True."
            )

        agent = VertexInformationAgent(
            model=_get_model_name(),
            system_prompt=get_information_agent_system_prompt(),
            project=project,
            location=_get_vertex_location(),
        )
    else:
        api_key = _get_api_key()
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY or PROVIDER_API_KEY, or set "
                "GOOGLE_GENAI_USE_VERTEXAI=True with GOOGLE_CLOUD_PROJECT."
            )

        openai_client = OpenAI(
            api_key=api_key,
            base_url=_get_base_url(),
        )
        client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)

        agent = AtomicAgent[InformationAgentInput, AgentAction](
            config=AgentConfig(
                client=client,
                model=_get_model_name(),
                history=ChatHistory(),
                system_prompt_generator=SystemPromptGenerator(
                    background=[get_information_agent_system_prompt()]
                ),
            )  # type: ignore[arg-type]
        )

        def _on_completion_response(response) -> None:
            usage = getattr(response, "usage", None)
            if usage:
                logger.debug(
                    "tokens - prompt=%d, completion=%d, total=%d",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                )

        agent.register_hook("completion:response", _on_completion_response)

    tools = {
        "intent_tool": IntentClassificationTool(IntentClassificationToolConfig()),
        "search_tool": SearchTool(
            SearchToolConfig(max_results=_get_search_result_limit(deep_research))
        ),
        "fetch_url_tool": FetchURLTool(BaseToolConfig()),
        "write_file_tool": WriteToFileTool(BaseToolConfig()),
    }

    return agent, tools


def run_information_agent(
    information_request: str, deep_research: bool = False
) -> str:
    """
    Run the agentic entity-discovery loop and return a JSON string.
    """
    start_time = time.perf_counter()
    agent, tools = create_information_agent(deep_research=deep_research)

    intent_tool = tools["intent_tool"]
    search_tool = tools["search_tool"]
    fetch_tool = tools["fetch_url_tool"]
    write_tool = tools["write_file_tool"]

    fetch_limit = _get_fetch_limit(deep_research)
    loop_limit = _get_loop_limit(deep_research)
    fetch_count = 0
    queries_run: list[dict[str, object]] = []
    total_results = 0
    json_file_path = ""
    markdown_file_path = ""
    final_table: StructuredEntityTable | None = None
    intent_decomposition: IntentDecomposition = intent_tool.run(
        IntentClassificationInput(
            user_request=information_request,
            deep_research=deep_research,
        )
    )

    source_registry: dict[str, dict[str, str | None]] = {}
    url_to_source_id: dict[str, str] = {}

    logger.info("starting entity discovery for topic query")

    logger.info(
        "intent decomposition ready: depth=%s, search_requests=%d",
        intent_decomposition.research_depth,
        len(intent_decomposition.search_requests),
    )

    action: AgentAction = agent.run(
        InformationAgentInput(
            information_request=information_request,
            deep_research=deep_research,
            intent_decomposition=intent_decomposition,
        )
    )

    for iteration in range(1, loop_limit + 1):
        logger.info(
            "[iter %d] agent -> %s | %s",
            iteration,
            action.action_type,
            action.reasoning,
        )

        if action.action_type == "search_web":
            query = (action.search_query or "").strip()
            if not query:
                tool_result_str = json.dumps(
                    {"error": "search_query was empty"}, ensure_ascii=False
                )
            else:
                logger.info("  search_web: querying %r", query)
                output = search_tool.run(SearchToolInput(value=query))

                enriched_results: list[dict[str, object]] = []
                for item in output.results:
                    url = _compact_text(str(item.get("href") or item.get("url") or ""))
                    title = _compact_text(str(item.get("title") or ""))
                    snippet = _compact_text(str(item.get("body") or item.get("snippet") or ""))
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
                            "href": url,
                            "body": snippet or "",
                        }
                    )

                queries_run.append({"query": query, "results": len(enriched_results)})
                total_results += len(enriched_results)
                tool_result_str = json.dumps(enriched_results, ensure_ascii=False)

        elif action.action_type == "fetch_url":
            if fetch_count >= fetch_limit:
                tool_result_str = json.dumps(
                    {
                        "error": (
                            f"fetch_url limit reached ({fetch_limit} calls). "
                            "Use the fetched evidence you already have and finish."
                        )
                    },
                    ensure_ascii=False,
                )
            else:
                urls = [url.strip() for url in (action.urls_to_fetch or []) if url.strip()]
                deduped_urls = list(dict.fromkeys(urls))
                if not deduped_urls:
                    tool_result_str = json.dumps(
                        {"error": "urls_to_fetch was empty"}, ensure_ascii=False
                    )
                else:
                    fetch_count += 1
                    logger.info(
                        "  fetch_url: fetching %d URL(s) [call %d/%d]",
                        len(deduped_urls),
                        fetch_count,
                        fetch_limit,
                    )
                    output = fetch_tool.run(FetchURLToolInputSchema(value=deduped_urls))

                    fetched_records: list[dict[str, object]] = []
                    for url, content in zip(deduped_urls, output.result):
                        source_id = url_to_source_id.get(url)
                        if not source_id:
                            source_id = f"src_{len(url_to_source_id) + 1:03d}"
                            url_to_source_id[url] = source_id
                            source_registry[source_id] = {
                                "source_id": source_id,
                                "title": url,
                                "url": url,
                                "snippet": None,
                            }

                        metadata = source_registry[source_id]
                        fetched_records.append(
                            {
                                "source_id": source_id,
                                "title": metadata.get("title") or url,
                                "url": url,
                                "snippet": metadata.get("snippet"),
                                "content": _truncate_content(content),
                            }
                        )

                    tool_result_str = json.dumps(fetched_records, ensure_ascii=False)

        elif action.action_type == "finish":
            if action.final_result is None:
                raise ValueError("Agent returned finish without final_result.")

            final_table = _normalize_result(
                action.final_result,
                query=information_request,
                source_registry=source_registry,
            )
            json_payload = final_table.model_dump_json(indent=2)
            markdown_payload = _render_markdown_document(final_table)

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
            json_file_path = json_write.absolute_path
            markdown_file_path = markdown_write.absolute_path
            logger.info("  finish: wrote JSON -> %s", json_file_path)
            logger.info("  finish: wrote markdown -> %s", markdown_file_path)
            break

        else:
            raise ValueError(f"Unknown action_type: {action.action_type}")

        agent.history.add_message(
            "user",
            ToolResult(tool_name=action.action_type, result=tool_result_str),
        )
        action = agent.run()
    else:
        logger.warning(
            "agent loop exhausted (%d iterations) without finishing",
            loop_limit,
        )

    execution_time_ms = int((time.perf_counter() - start_time) * 1000)
    status = "success" if final_table and json_file_path and markdown_file_path else "error"

    if final_table is None:
        final_table = StructuredEntityTable(
            query=information_request,
            title=f"Discovered entities for {information_request}",
            columns=[],
            rows=[],
            sources=[],
        )

    result = InformationAgentOutput(
        status=status,
        json_file_path=json_file_path,
        markdown_file_path=markdown_file_path,
        result=final_table,
        meta={
            "deep_research": deep_research,
            "intent_decomposition": intent_decomposition.model_dump(),
            "queries_run": queries_run,
            "total_results": total_results,
            "fetch_calls": fetch_count,
            "fetch_limit": fetch_limit,
            "execution_time_ms": execution_time_ms,
            "model": _get_model_name(),
        },
    )
    return result.model_dump_json(indent=2)


def resolve_deep_research_choice(explicit_choice: bool | None = None) -> bool:
    if explicit_choice is not None:
        return explicit_choice

    answer = input("Run in deep-research mode? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


if __name__ == "__main__":
    import argparse
    import time

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
    args = parser.parse_args()
    time_st = time.time()
    print(
        run_information_agent(
            args.query,
            deep_research=resolve_deep_research_choice(args.deep_research),
        )
    )
    time_end = time.time()
    print("Processing time:", time_end - time_st)

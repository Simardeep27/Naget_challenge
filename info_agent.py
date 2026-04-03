import json
import logging
import time
from typing import Callable

import instructor
from atomic_agents import AgentConfig, AtomicAgent, BaseToolConfig
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from openai import OpenAI

from prompts.information_agent_prompt import get_information_agent_system_prompt
from schema import AgentAction, InformationAgentInput, InformationAgentOutput, IntentDecomposition, StructuredEntityTable, ToolResult
from tools.fetch_url import FetchURLTool, FetchURLToolInputSchema
from tools.intent_classificaiton import (
    IntentClassificationInput,
    IntentClassificationTool,
    IntentClassificationToolConfig,
)
from tools.web_search_tool import SearchTool, SearchToolConfig, SearchToolInput
from tools.write_to_file import WriteToFileTool, WriteToFileToolInputSchema
from utils.config import (
    JSON_OUTPUT_PATH,
    MARKDOWN_OUTPUT_PATH,
    get_api_key,
    get_base_url,
    get_fetch_limit,
    get_lightning_search_result_limit,
    get_lightning_search_timeout_seconds,
    get_loop_limit,
    get_model_name,
    get_search_result_limit,
    get_vertex_location,
    get_vertex_project,
    use_vertex_ai,
)
from utils.lightning import run_lightning_research
from utils.llm_utils import VertexInformationAgent
from utils.progress import CliProgressReporter
from utils.recursive_research import run_recursive_research
from utils.result_utils import normalize_result
from utils.text_utils import compact_text, render_markdown_document, truncate_content

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_information_tools(
    deep_research: bool = False,
    lightning: bool = False,
):
    if lightning:
        search_config = SearchToolConfig(
            max_results=get_lightning_search_result_limit(),
            timeout_seconds=get_lightning_search_timeout_seconds(),
        )
    else:
        search_config = SearchToolConfig(
            max_results=get_search_result_limit(deep_research)
        )

    tools = {
        "search_tool": SearchTool(search_config),
        "fetch_url_tool": FetchURLTool(BaseToolConfig()),
        "write_file_tool": WriteToFileTool(BaseToolConfig()),
    }

    if not lightning:
        tools["intent_tool"] = IntentClassificationTool(IntentClassificationToolConfig())

    return tools


def create_information_agent(deep_research: bool = False):
    """Create the agent and tools for the agentic research loop."""
    if use_vertex_ai():
        from google.genai import types

        project = get_vertex_project()
        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT must be set when GOOGLE_GENAI_USE_VERTEXAI=True."
            )

        agent = VertexInformationAgent(
            model=get_model_name(),
            system_prompt=get_information_agent_system_prompt(),
            project=project,
            location=get_vertex_location(),
            http_options=types.HttpOptions(
                api_version="v1",
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    else:
        api_key = get_api_key()
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY or PROVIDER_API_KEY, or set "
                "GOOGLE_GENAI_USE_VERTEXAI=True with GOOGLE_CLOUD_PROJECT."
            )

        openai_client = OpenAI(
            api_key=api_key,
            base_url=get_base_url(),
            max_retries=0,
        )
        client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)

        agent = AtomicAgent[InformationAgentInput, AgentAction](
            config=AgentConfig(
                client=client,
                model=get_model_name(),
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

    tools = create_information_tools(deep_research=deep_research)

    return agent, tools


def run_information_agent(
    information_request: str,
    deep_research: bool = False,
    recursive_research: bool = False,
    lightning: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Run the agentic entity-discovery loop and return a JSON string."""
    progress_callback = progress_callback or (lambda _message: None)
    if lightning:
        progress_callback("Initializing lightning mode")
        tools = create_information_tools(lightning=True)
        result = run_lightning_research(
            information_request=information_request,
            search_tool=tools["search_tool"],
            fetch_tool=tools["fetch_url_tool"],
            write_tool=tools["write_file_tool"],
            progress_callback=progress_callback,
        )
        progress_callback("Completed")
        return result.model_dump_json(indent=2)

    start_time = time.perf_counter()
    progress_callback("Initializing agent")
    agent, tools = create_information_agent(deep_research=deep_research)

    intent_tool = tools["intent_tool"]
    search_tool = tools["search_tool"]
    fetch_tool = tools["fetch_url_tool"]
    write_tool = tools["write_file_tool"]

    fetch_limit = get_fetch_limit(deep_research)
    loop_limit = get_loop_limit(deep_research)
    fetch_count = 0
    queries_run: list[dict[str, object]] = []
    total_results = 0
    json_file_path = ""
    markdown_file_path = ""
    final_table: StructuredEntityTable | None = None
    recursive_research_meta: dict[str, object] = {"enabled": recursive_research}
    progress_callback("Decomposing intent")
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

    progress_callback("Planning research")
    action: AgentAction = agent.run(
        InformationAgentInput(
            information_request=information_request,
            deep_research=deep_research,
            recursive_research=recursive_research,
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
                progress_callback(f"Searching web (iteration {iteration})")
                logger.info("  search_web: querying %r", query)
                output = search_tool.run(SearchToolInput(value=query))

                if output.error:
                    queries_run.append(
                        {
                            "query": query,
                            "results": 0,
                            "error": output.error,
                        }
                    )
                    tool_result_str = json.dumps(
                        {
                            "query": query,
                            "results": [],
                            "error": output.error,
                        },
                        ensure_ascii=False,
                    )
                    logger.warning("  search_web failed: %s", output.error)
                    agent.history.add_message(
                        "user",
                        ToolResult(tool_name=action.action_type, result=tool_result_str),
                    )
                    progress_callback(f"Thinking about next step (iteration {iteration})")
                    action = agent.run()
                    continue

                enriched_results: list[dict[str, object]] = []
                for item in output.results:
                    url = compact_text(str(item.get("href") or item.get("url") or ""))
                    title = compact_text(str(item.get("title") or ""))
                    snippet = compact_text(
                        str(item.get("body") or item.get("snippet") or "")
                    )
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
                    progress_callback(
                        f"Fetching {len(deduped_urls)} page(s) ({fetch_count}/{fetch_limit})"
                    )
                    logger.info(
                        "  fetch_url: fetching %d URL(s) [call %d/%d]",
                        len(deduped_urls),
                        fetch_count,
                        fetch_limit,
                    )
                    output = fetch_tool.run(FetchURLToolInputSchema(value=deduped_urls))

                    fetched_records: list[dict[str, object]] = []
                    failed_fetches: list[dict[str, object]] = []
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
                        if fetch_tool.is_error_result(content):
                            failed_fetches.append(
                                {
                                    "source_id": source_id,
                                    "title": metadata.get("title") or url,
                                    "url": url,
                                    "snippet": metadata.get("snippet"),
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
                                "content": truncate_content(content),
                            }
                        )

                    tool_result_str = json.dumps(
                        {
                            "fetched_records": fetched_records,
                            "failed_fetches": failed_fetches,
                        },
                        ensure_ascii=False,
                    )

        elif action.action_type == "finish":
            if action.final_result is None:
                raise ValueError("Agent returned finish without final_result.")

            progress_callback("Normalizing results")
            final_table = normalize_result(
                action.final_result,
                query=information_request,
                source_registry=source_registry,
                require_complete=not recursive_research,
            )
            if recursive_research:
                progress_callback("Running recursive research")
                final_table, recursive_research_meta = run_recursive_research(
                    result=final_table,
                    original_query=information_request,
                    deep_research=deep_research,
                    search_tool=search_tool,
                    fetch_tool=fetch_tool,
                    source_registry=source_registry,
                    url_to_source_id=url_to_source_id,
                    progress_callback=progress_callback,
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
        progress_callback(f"Thinking about next step (iteration {iteration})")
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
            "recursive_research": recursive_research_meta,
            "intent_decomposition": intent_decomposition.model_dump(),
            "queries_run": queries_run,
            "total_results": total_results,
            "fetch_calls": fetch_count,
            "fetch_limit": fetch_limit,
            "execution_time_ms": execution_time_ms,
            "model": get_model_name(),
        },
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
        help=(
            "Run a targeted follow-up search pass that backfills missing "
            "attribute/value pairs into the original table"
        ),
    )
    parser.add_argument(
        "--lightning",
        action="store_true",
        help=(
            "Run a speed-optimized pass that uses one compact search/fetch/extract "
            "cycle and aims to finish in under 10 seconds"
        ),
    )
    args = parser.parse_args()
    if args.lightning and args.deep_research is True:
        parser.error("--lightning cannot be combined with --deep-research")
    if args.lightning and args.recursive_research:
        parser.error("--lightning cannot be combined with --recursive-research")

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
        )
        progress.complete("Completed")
    print(result)

from __future__ import annotations

import json
from typing import Callable

from schema import InformationAgentOutput
from tools.fetch_url import FetchURLTool
from tools.web_search_tool import SearchTool
from tools.write_to_file import WriteToFileTool


def run_lightning_research(
    information_request: str,
    search_tool: SearchTool,
    fetch_tool: FetchURLTool,
    write_tool: WriteToFileTool,
    progress_callback: Callable[[str], None] | None = None,
) -> InformationAgentOutput:
    """Compatibility wrapper for the old lightning entry point.

    Lightning mode now runs through the same deterministic slot-driven pipeline
    as standard and deep research, just with smaller budgets.
    """

    del search_tool, fetch_tool, write_tool

    from info_agent import run_information_agent

    raw_result = run_information_agent(
        information_request=information_request,
        deep_research=False,
        recursive_research=False,
        lightning=True,
        progress_callback=progress_callback,
    )
    return InformationAgentOutput.model_validate(json.loads(raw_result))

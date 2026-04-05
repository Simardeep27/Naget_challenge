# Architecture Notes

This document gives a quick mental model for how Soogle is organized and how a
request moves through the system.

## High-Level Flow

### CLI flow

```text
main.py
  -> run_information_agent(...)
  -> write JSON / markdown artifacts
  -> print final JSON payload
```

### API flow

```text
frontend
  -> POST /api/research/stream
  -> api_service.py
  -> run_information_agent(...)
  -> streamed progress events
  -> final structured result
```

## Core Modules

### [info_agent.py](/Users/ssethi/Documents/task/info_agent.py)

Owns the pipeline orchestration:

- choose mode budgets
- build the research plan
- run candidate search
- rerank candidates
- fetch frontier pages
- extract structured rows
- optionally run recursive backfill
- finalize artifacts and metadata

This file should stay focused on sequencing and policy, not on low-level search
or fetch implementation details.

### [schema.py](/Users/ssethi/Documents/task/schema.py)

Defines the shared data model:

- research plan schema
- candidate schema
- preview/fetch schemas
- final structured table
- output envelope

Keeping the schemas centralized makes the pipeline easier to debug and keeps the
frontend/backend contract explicit.

### [tools/](/Users/ssethi/Documents/task/tools)

These files wrap individual external actions:

- [research_planner.py](/Users/ssethi/Documents/task/tools/research_planner.py)
- [web_search_tool.py](/Users/ssethi/Documents/task/tools/web_search_tool.py)
- [fetch_url.py](/Users/ssethi/Documents/task/tools/fetch_url.py)
- [write_to_file.py](/Users/ssethi/Documents/task/tools/write_to_file.py)

The orchestration layer calls tools, but the tools should not own pipeline
policy.

### [utils/](/Users/ssethi/Documents/task/utils)

Shared helpers are grouped by concern:

- [config.py](/Users/ssethi/Documents/task/utils/config.py): env and runtime settings
- [llm_utils.py](/Users/ssethi/Documents/task/utils/llm_utils.py): structured generation helpers
- [tiered_research.py](/Users/ssethi/Documents/task/utils/tiered_research.py): candidate ranking and extraction helpers
- [recursive_research.py](/Users/ssethi/Documents/task/utils/recursive_research.py): backfill logic
- [result_utils.py](/Users/ssethi/Documents/task/utils/result_utils.py): normalization and citation shaping
- [progress.py](/Users/ssethi/Documents/task/utils/progress.py): CLI progress display
- [text_utils.py](/Users/ssethi/Documents/task/utils/text_utils.py): formatting helpers

## Frontend Structure

The frontend is intentionally thin:

- [page.tsx](/Users/ssethi/Documents/task/frontend/app/page.tsx): app entrypoint
- [search-workspace.tsx](/Users/ssethi/Documents/task/frontend/components/search-workspace.tsx): request submission and stream parsing
- [results-panel.tsx](/Users/ssethi/Documents/task/frontend/components/results-panel.tsx): results and progress rendering
- [types.ts](/Users/ssethi/Documents/task/frontend/lib/types.ts): typed API response shapes

The frontend should remain a renderer for backend progress and results, not a
second source of research logic.

## Separation Of Responsibilities

The intended boundaries are:

- orchestration in `info_agent.py`
- IO wrappers in `tools/`
- shared helpers in `utils/`
- schemas in `schema.py`
- transport in `api_service.py`
- UI rendering in `frontend/`

This separation keeps the codebase easier to review and reduces the chance that
logic gets duplicated across the CLI, API, and frontend.

## Current Tradeoffs

- The pipeline favors quality and traceability over speed.
- Model-backed planning and extraction improve structure, but introduce external dependency risk.
- Serverless deployments need explicit care around credentials, timeouts, and filesystem writes.

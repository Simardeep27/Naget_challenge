# Entity Discovery System

This project builds a topic-driven entity discovery workflow around [`info_agent.py`].


# RUNS INFORMATION

| Runs | Latency(sec) | 
|---|---|
| Normal Search (v1) | 130 | 
| DeepSearch (v1) | 230 | 
| Normal Search (v2) | 52.3 | 
| DeepSearch (v2) | 132.04 | 



Given a query like:

- `AI startups in healthcare`
- `top pizza places in Brooklyn`
- `open source database tools`

the pipeline will:

1. Build a slot-driven research plan around one core objective.
2. Retrieve cheap search candidates using metadata only.
3. Rerank and cluster candidates before fetching any full pages.
4. Preview the top pages using partial fetches to identify frontier pages.
5. Full-fetch only the strongest frontier pages.
6. Run schema-aware extraction driven by required and optional slots.
7. Fill missing slots with infer -> in-domain expand -> new web search follow-up.
8. Return JSON output where each populated cell includes source citations.
9. Write both JSON and markdown outputs to `output/information_agent/`.

This is now a deterministic pipeline. The repo no longer uses an LLM agent loop
where the model chooses action types like `search_web`, `fetch_url`, or `finish`.

## Main Files

- [`info_agent.py`](/Users/ssethi/Documents/task/info_agent.py): main static pipeline orchestration
- [`schema.py`](/Users/ssethi/Documents/task/schema.py): Pydantic schemas for research plans, candidates, previews, and structured output
- [`tools/research_planner.py`](/Users/ssethi/Documents/task/tools/research_planner.py): slot-driven planner
- [`tools/web_search_tool.py`](/Users/ssethi/Documents/task/tools/web_search_tool.py): search tool
- [`tools/fetch_url.py`](/Users/ssethi/Documents/task/tools/fetch_url.py): preview + full-fetch tool
- [`tools/write_to_file.py`](/Users/ssethi/Documents/task/tools/write_to_file.py): file writer
- [`utils/tiered_research.py`](/Users/ssethi/Documents/task/utils/tiered_research.py): candidate retrieval, reranking, previewing, and slot-aware extraction helpers
- [`utils/recursive_research.py`](/Users/ssethi/Documents/task/utils/recursive_research.py): slot-level follow-up logic

## Install

Using `pip`:

```bash
python3 -m pip install -r requirements.txt
```

Using `uv`:

```bash
uv sync
```

## Environment

Secrets and deployment-specific settings should go in `.env`.

For Google Cloud Vertex AI, set these values there:

```bash
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True
INFORMATION_AGENT_MODEL=gemini-2.5-flash
```

OpenAI-compatible variables are still supported as a fallback when `GOOGLE_GENAI_USE_VERTEXAI` is not enabled.

## Configuration

Non-secret runtime tuning now lives in [`config.yaml`](/Users/ssethi/Documents/task/config.yaml), including:

- output paths
- standard/deep/lightning mode limits
- recursive-research limits
- default search provider and timeout
- default model fallbacks

Example:

```yaml
search:
  provider: duckduckgo
  timeout_seconds: 5

modes:
  standard:
    search_result_limit: 8
    fetch_limit: 6
  lightning:
    max_search_queries: 3
    fetch_url_limit: 2
```

Environment variables can still override these values when needed, but `config.yaml` is now the primary place to tune the system.

You also need Google Cloud credentials available locally, typically:

```bash
gcloud auth application-default login
```

## Run

Run the main agent directly:

```bash
uv run python info_agent.py "AI startups in healthcare"
```

Or use the small wrapper:

```bash
uv run python main.py "top pizza places in Brooklyn"
```

## Web Frontend

A Vercel-ready Next.js frontend now lives in [`frontend/`](/Users/ssethi/Documents/task/frontend).
It provides:

- a chat-style research composer
- selectable methods: standard, deep, lightning
- an optional recursive-fill toggle
- tabular result rendering with sources and execution metadata

The frontend expects a same-origin API at `/api/research`.
Live progress updates stream over `/api/research/stream`.

## Vercel Services Setup

This repo includes a root [`vercel.json`](/Users/ssethi/Documents/task/vercel.json)
configured for Vercel Services:

- `web` -> [`frontend/`](/Users/ssethi/Documents/task/frontend)
- `agent` -> [`api_service.py`](/Users/ssethi/Documents/task/api_service.py)

The Python service exposes:

- `GET /api/health`
- `POST /api/research`
- `POST /api/research/stream`

For local multi-service development, use:

```bash
vercel dev -L
```

If you only want to run the frontend shell locally, install frontend deps and run:

```bash
cd frontend
npm install
NEXT_PUBLIC_AGENT_URL=http://127.0.0.1:8000 npm run dev
```

To connect that local frontend directly to the Python backend without `vercel dev`,
run the API service separately on port `8000`:

```bash
uv run uvicorn api_service:app --host 127.0.0.1 --port 8000
```

If you do not pass a research-depth flag, the CLI will ask whether you want deep research.

You can also set it explicitly:

```bash
uv run python info_agent.py "open source database tools" --deep-research
uv run python info_agent.py "open source database tools" --no-deep-research
uv run python info_agent.py "open source database tools" --recursive-research
uv run python info_agent.py "open source database tools" --lightning
```

## Output

The run returns an `InformationAgentOutput` JSON object with:

- `status`
- `json_file_path`
- `markdown_file_path`
- `result`
- `meta`

The `meta` object includes the selected research depth, the slot-driven research plan, candidate/frontier counts, and slot-follow-up stats when the pipeline had to backfill missing required or optional slots.

Inside `result`:

- `query`
- `title`
- `columns`
- `rows`
- `sources`

Each populated cell looks like:

```json
{
  "value": "Tempus",
  "citations": [
    {
      "source_id": "src_001",
      "source_title": "Example Source",
      "source_url": "https://example.com/article",
      "quote": "Tempus is a technology company advancing precision medicine."
    }
  ]
}
```

This satisfies the traceability requirement: every non-empty cell is backed by one or more fetched sources.

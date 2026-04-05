# Soogle (https://naget-challenge.vercel.app?_vercel_share=S3s4WzDoGPbs62K2WiwmqLYGxYUhTs7Z)

Soogle is a structured web-research system that turns an open-ended query into a
citation-backed table. The project combines a Python research pipeline with a
minimal Next.js frontend and a FastAPI streaming API.

## What It Does

Given a query such as:

- `AI startups in healthcare`
- `top pizza places in Brooklyn`
- `open source database tools`

Soogle:

1. Builds a slot-driven research plan for the query.
2. Generates a small set of retrieval queries.
3. Retrieves and reranks candidate pages.
4. Fetches only the strongest frontier pages.
5. Extracts structured rows with source-backed citations.
6. Optionally runs recursive follow-up research to fill missing slots.
7. Writes both JSON and markdown artifacts.

The system is intentionally deterministic at the orchestration layer. Instead of
letting the model choose arbitrary actions in a loop, the pipeline follows a
fixed sequence of planning, retrieval, frontier selection, extraction, and
optional backfilling.

## Approach

### Pipeline

```text
User Query
   -> Research Planner
   -> Search Query Generation
   -> Candidate Search
   -> Semantic Rerank
   -> Frontier Fetch
   -> Structured Extraction
   -> Optional Recursive Backfill
   -> JSON + Markdown Output
```

### Research Modes

- `standard`: balanced search and fetch budgets
- `deep`: broader retrieval and more exhaustive extraction
- `lightning`: smaller budgets for faster turnaround
- `recursive research`: deep mode plus targeted backfilling of missing cells

### Design Goals

- Keep the orchestration understandable and debuggable.
- Optimize for traceable, citation-backed structured output.
- Separate retrieval, extraction, and rendering concerns.
- Surface progress in both the CLI and the frontend.

## Repository Layout

The codebase is intentionally split by responsibility:

```text
.
├── api_service.py              # FastAPI service and streaming endpoints
├── info_agent.py               # Main pipeline orchestration
├── main.py                     # CLI entry point
├── schema.py                   # Shared Pydantic schemas
├── tools/
│   ├── research_planner.py     # Query -> ResearchPlan
│   ├── web_search_tool.py      # Search provider wrapper
│   ├── fetch_url.py            # Page fetch + extraction cache
│   └── write_to_file.py        # Output artifact writing
├── utils/
│   ├── config.py               # Settings and environment resolution
│   ├── llm_utils.py            # Structured LLM / Vertex helpers
│   ├── tiered_research.py      # Candidate ranking + table extraction helpers
│   ├── recursive_research.py   # Follow-up slot backfilling
│   ├── result_utils.py         # Normalization and citation shaping
│   ├── progress.py             # CLI progress reporting
│   └── text_utils.py           # Shared text-formatting helpers
├── frontend/
│   ├── app/                    # Next.js app router entrypoints
│   ├── components/             # Search page and results UI
│   └── lib/types.ts            # Frontend response types
├── config.yaml                 # Runtime tuning
└── vercel.json                 # Vercel Services configuration
```

A longer architecture walkthrough lives in
[docs/architecture.md](/Users/ssethi/Documents/task/docs/architecture.md).

## Setup

### 1. Install dependencies

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
python3 -m pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file with deployment-specific secrets.

Vertex AI example:

```bash
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
INFORMATION_AGENT_MODEL=gemini-2.5-flash
```

OpenAI-compatible example:

```bash
GOOGLE_GENAI_USE_VERTEXAI=false
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4.1-mini
```

### 3. Authenticate

For local development with Vertex AI:

```bash
gcloud auth application-default login
```

For Vercel or another serverless deployment, local ADC is not available.
Instead, provide one of:

```bash
GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
```

or

```bash
GOOGLE_SERVICE_ACCOUNT_BASE64=base64-encoded-service-account-json
```

The service account needs Vertex AI access for both standard generation and
cached-content calls.

### 4. Tune runtime settings

Non-secret runtime settings live in [config.yaml](/Users/ssethi/Documents/task/config.yaml).
This file controls:

- output paths
- search provider and timeout
- standard, deep, and lightning mode budgets
- recursive-research limits
- cache TTLs
- default model selection

## Running The Project

### CLI

Run the main pipeline directly:

```bash
uv run python info_agent.py "AI startups in healthcare"
```

Or use the small wrapper:

```bash
uv run python main.py "top pizza places in Brooklyn"
```

Explicit mode examples:

```bash
uv run python info_agent.py "open source database tools" --deep-research
uv run python info_agent.py "open source database tools" --no-deep-research
uv run python info_agent.py "open source database tools" --recursive-research
uv run python info_agent.py "open source database tools" --lightning
```

### Local API

Run the backend service:

```bash
uv run uvicorn api_service:app --host 127.0.0.1 --port 8000
```

Available endpoints:

- `GET /health`
- `POST /research`
- `POST /research/stream`

### Local Frontend

Run the frontend against the local API:

```bash
cd frontend
npm install
NEXT_PUBLIC_AGENT_URL=http://127.0.0.1:8000 npm run dev
```

### Vercel

This repository is configured for Vercel Services via
[vercel.json](/Users/ssethi/Documents/task/vercel.json):

- `web` -> `frontend/`
- `agent` -> `api_service.py`

High-level deploy flow:

1. Import the repo into Vercel from the repository root.
2. Add the required environment variables.
3. Redeploy after every env-var change.
4. Verify `GET /api/health`.
5. Verify a full streamed request against `POST /api/research/stream`.

## Frontend Overview

The frontend is intentionally small:

- a centered search-first landing page
- a mode picker for deep and recursive research
- live progress rendering from the streaming API
- a results area for tables, sources, and execution metadata

The frontend does not implement research logic itself. It only submits requests
to the backend and renders streamed progress plus the final structured result.

## Output Format

The pipeline returns an `InformationAgentOutput` object with:

- `status`
- `json_file_path`
- `markdown_file_path`
- `result`
- `meta`

`result` contains:

- `query`
- `title`
- `columns`
- `rows`
- `sources`

Each populated cell is citation-backed. Example:

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

This keeps the output auditable and easier to debug.

## Code Structure Notes

The codebase is organized around a few clear boundaries:

- `info_agent.py` owns pipeline orchestration only.
- `tools/` owns external actions such as search, fetch, and write.
- `utils/` owns cross-cutting logic like config, normalization, prompting, and progress.
- `schema.py` keeps the shared data model in one place.
- `frontend/` is isolated from backend internals and consumes typed API responses.

The repo still has room for further cleanup, but the current structure keeps the
main responsibilities separated and makes it possible to reason about each stage
in isolation.

## Known Limitations

- Deep and recursive research can be slow because the system prioritizes result quality over latency.
- Search quality depends on third-party providers and their rate limits.
- Some sites block scraping or return `403` responses, which can reduce coverage.
- Structured extraction can still degrade when upstream model responses are malformed or incomplete.
- Vertex AI deployment is more complex on Vercel because serverless environments need explicit credentials instead of local ADC.
- The frontend currently focuses on a single research workflow rather than full conversational state/history.

## Future Improvements

- Better observability around extraction-stage failures.
- More explicit retry and fallback behavior for model calls.
- Smarter frontier selection for local/business queries.
- More test coverage around normalization and streaming edge cases.
- Cleaner separation between local-dev helpers and deployment-specific behavior.

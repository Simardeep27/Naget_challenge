# Soogle

**Simardeep’s Google** — a web-search agent that converts open-ended queries into citation-backed structured information.

Soogle is a deterministic research pipeline for users who want structured, traceable outputs from noisy web queries. Instead of relying on a free-form agent loop that repeatedly decides what action to take next, the system builds an explicit research plan, retrieves candidate links, reranks them semantically, reads only the strongest frontier pages, and extracts structured results backed by citations.

The current focus of the system is **completion quality and result quality** rather than raw latency. Earlier versions were optimized more aggressively for speed, but the pipeline has evolved toward better retrieval decisions, stronger extraction, and more reliable structured outputs.

---

## What it does

Given a query such as:

- `AI startups in healthcare`
- `top pizza places in Brooklyn`
- `open source database tools`

Soogle:

1. Builds a structured research plan from the user query.
2. Breaks the query into retrieval-oriented intents and likely information needs.
3. Retrieves candidate links using web search.
4. Reranks results semantically before doing expensive page reads.
5. Fully loads only the top-N frontier pages based on the selected strategy.
6. Extracts relevant data into a structured schema.
7. Optionally performs recursive backfilling in batches for missing entries.
8. Returns citation-backed structured JSON and markdown outputs.

---

## Why this design

Earlier experiments used a more agentic loop where the model chose actions such as searching, fetching, and deciding when to stop. In practice, that approach was harder to control, more expensive, and less predictable.

Soogle instead uses a **deterministic slot-driven pipeline**. The design decisions were guided by a few goals:

- **Better completion quality** by explicitly modeling missing information.
- **Higher result quality** by reranking search results before full page reads.
- **Lower unnecessary fetch cost** by avoiding naive HTML loading for every retrieved result.
- **Traceability** by attaching source citations to populated output cells.
- **Controllability** through selectable research strategies such as standard, deep, and lightning.

This makes the pipeline easier to inspect, tune, and evaluate as a research system.

---

## Approach

### 1. Research plan construction
The user query is first transformed into a structured research plan. This plan captures:

- the main objective
- sub-intents
- probable supporting information
- likely fields needed to answer the query well

This gives the pipeline an explicit target structure before retrieval begins.

### 2. Candidate retrieval
The system retrieves initial links using web search. Earlier versions used a more naive pattern of reading many retrieved pages directly. The current pipeline instead treats search results as cheap candidates and delays expensive reads until after filtering.

### 3. Semantic reranking
Retrieved links are reranked using semantic relevance. This step is one of the key design changes in the system: rather than trusting search rank alone, Soogle reorders candidates according to how well they match the research plan and likely answer requirements.

### 4. Frontier selection and full-page reads
Only the strongest frontier pages are fully loaded. The number of pages fetched depends on the chosen strategy:

- **Lightning**: more aggressive pruning, lower cost, faster turnaround
- **Standard**: balanced retrieval depth
- **Deep research**: higher recall and more complete extraction at the cost of latency

### 5. Structured extraction
Once frontier pages are available, the pipeline uses the research plan and fetched content to extract structured results. The output is schema-aware and designed to produce citation-backed fields rather than free-form text summaries.

### 6. Recursive research for missing entries
If recursive research is enabled, the system inspects missing or incomplete entries and performs batched backfilling. This improves completion quality, but introduces additional latency. This mode is useful when the query requires broader coverage or when partial missingness matters.

---

## Architecture

```text
User Query
   ↓
Research Planner
   ↓
Intent-aware Search Query Generation
   ↓
Web Search Candidate Retrieval
   ↓
Semantic Reranking / Filtering
   ↓
Preview / Frontier Selection
   ↓
Top-N Full Page Fetch
   ↓
Schema-aware Extraction
   ↓
(Optional) Recursive Backfilling for Missing Entries
   ↓
Citation-backed JSON + Markdown Output
```


## Main Components

- **`info_agent.py`** — main static pipeline orchestration  
- **`schema.py`** — Pydantic schemas for research plans, candidates, previews, and structured output  
- **`tools/research_planner.py`** — slot-driven planner  
- **`tools/web_search_tool.py`** — search tool  
- **`tools/fetch_url.py`** — preview and full-fetch tool  
- **`tools/write_to_file.py`** — file writer  
- **`utils/tiered_research.py`** — candidate retrieval, reranking, previewing, and slot-aware extraction helpers  
- **`utils/recursive_research.py`** — slot-level follow-up logic  
- **`api_service.py`** — Python API service  
- **`frontend/`** — Vercel-ready Next.js frontend

## Demo UI

The project includes a demo-ready frontend with support for:

- a chat-style research composer  
- selectable methods: **standard**, **deep research**, and **lightning**  
- optional recursive filling  
- tabular structured result rendering  
- execution metadata  
- search path visibility  
- live API-backed research execution  

<!-- ![Soogle Home](./docs/soogle-home.png) -->
<!-- ![Soogle Results](./docs/soogle-results.png) -->

## Setup

### 1. Install dependencies

Using `uv`:

```bash
uv sync
```

Or using `pip`:

```bash
python3 -m pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file for secrets and deployment-specific settings.

For Vertex AI:

```bash
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True
INFORMATION_AGENT_MODEL=gemini-2.5-flash
```

OpenAI-compatible variables can still be supported as a fallback when Vertex AI is not enabled.

### 3. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

For Vercel or other serverless deployments, local ADC is not available by default.
Set one of these environment variables instead:

```bash
GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
```

or

```bash
GOOGLE_SERVICE_ACCOUNT_BASE64=base64-encoded-service-account-json
```

The service account needs Vertex AI access for both content generation and cached-content operations.

### 4. Configure runtime settings

Non-secret runtime tuning lives in `config.yaml`. This includes:

- output paths
- search provider and timeout
- standard/deep/lightning mode limits
- recursive research limits
- model fallback behavior

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

## Running the Project

### Option 1: Vercel deployment (recommended)

This is the primary setup path for the project.

The repository includes a root `vercel.json` configured for Vercel Services:

- `web` → `frontend/`
- `agent` → `api_service.py`

The Python service exposes:

- `GET /api/health`
- `POST /api/research`
- `POST /api/research/stream`

For local multi-service development with Vercel:

```bash
vercel dev -L
```

### Option 2: CLI

Run the main pipeline directly:

```bash
uv run python info_agent.py "AI startups in healthcare"
```

Or use the wrapper:

```bash
uv run python main.py "top pizza places in Brooklyn"
```

You can also choose an explicit research mode:

```bash
uv run python info_agent.py "open source database tools" --deep-research
uv run python info_agent.py "open source database tools" --no-deep-research
uv run python info_agent.py "open source database tools" --recursive-research
uv run python info_agent.py "open source database tools" --lightning
```

### Option 3: Local API + frontend

Run the backend:

```bash
uv run uvicorn api_service:app --host 127.0.0.1 --port 8000
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend includes local `/api/*` proxy routes to forward requests to the backend.

## Output Format

The pipeline returns an `InformationAgentOutput` object with:

- `status`
- `json_file_path`
- `markdown_file_path`
- `result`
- `meta`

The `meta` section includes information such as:

- selected research depth
- research plan
- candidate counts
- frontier counts
- follow-up statistics for recursive slot filling

The `result` section contains:

- `query`
- `title`
- `columns`
- `rows`
- `sources`

Each populated field is citation-backed. Example:

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

This makes the output easier to verify, debug, and reuse downstream.

## Performance Note

The current pipeline prioritizes retrieval quality and completion quality more than raw speed.

- Earlier iterations focused more aggressively on latency.
- The current direction adds stronger reranking and optional recursive backfilling.
- A recent pipeline version takes roughly `~180 seconds` in the deep pipeline setting.

This is an active tradeoff rather than an accident: the system is currently optimized to improve answer completeness and structured output quality.

## Known Limitations

- The system can still be slow, especially under DDGS rate limits.
- Recursive research improves completeness, but increases latency.
- Search quality is still partly constrained by external search provider behavior.
- The current search stack may be further improved with a custom semantic scraper to reduce dependency on external limits and improve retrieval control.

## Future Direction

A major next step is replacing part of the current retrieval dependency with a more controllable semantic scraping layer. The goal is to reduce search bottlenecks, improve frontier selection, and lower the latency introduced by repeated external search constraints.

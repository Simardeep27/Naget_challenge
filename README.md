# Entity Discovery System

This project builds a topic-driven entity discovery workflow around [`info_agent.py`].

Given a query like:

- `AI startups in healthcare`
- `top pizza places in Brooklyn`
- `open source database tools`

the agent will:

1. Decompose the query to multiple intents
2. Search the web for relevant pages.
3. Fetch and extract readable content from those pages.
4. Use an LLM to assemble a structured entity table.
5. Return JSON output where each populated cell includes source citations.
6. Write both JSON and markdown outputs to `output/information_agent/`.

## Main Files

- [`info_agent.py`](/Users/ssethi/Documents/task/info_agent.py): main orchestration loop
- [`schema.py`](/Users/ssethi/Documents/task/schema.py): Pydantic schemas for actions and structured output
- [`prompts/information_agent_prompt.py`](/Users/ssethi/Documents/task/prompts/information_agent_prompt.py): system prompt
- [`tools/web_search_tool.py`](/Users/ssethi/Documents/task/tools/web_search_tool.py): search tool
- [`tools/fetch_url.py`](/Users/ssethi/Documents/task/tools/fetch_url.py): fetch-and-extract tool
- [`tools/write_to_file.py`](/Users/ssethi/Documents/task/tools/write_to_file.py): file writer

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

For Google Cloud Vertex AI, set these values in `.env`:

```bash
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True
INFORMATION_AGENT_MODEL=gemini-2.5-flash
```

Optional search settings:

```bash
SEARCH_PROVIDER=duckduckgo
BRAVE_API_KEY=your_brave_key
```

You also need Google Cloud credentials available locally, typically:

```bash
gcloud auth application-default login
```

OpenAI-compatible variables are still supported as a fallback when `GOOGLE_GENAI_USE_VERTEXAI` is not enabled.

## Run

Run the main agent directly:

```bash
uv run python info_agent.py "AI startups in healthcare"
```

Or use the small wrapper:

```bash
uv run python main.py "top pizza places in Brooklyn"
```

If you do not pass a research-depth flag, the CLI will ask whether you want deep research.

You can also set it explicitly:

```bash
uv run python info_agent.py "open source database tools" --deep-research
uv run python info_agent.py "open source database tools" --no-deep-research
```

## Output

The run returns an `InformationAgentOutput` JSON object with:

- `status`
- `json_file_path`
- `markdown_file_path`
- `result`
- `meta`

The `meta` object includes the selected research depth and the intent decomposition used to guide the search plan.

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

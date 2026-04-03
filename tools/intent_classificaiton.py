import json

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig
from openai import OpenAI
from pydantic import Field

from schema import IntentDecomposition, SearchIntent
from utils.config import (
    get_api_key,
    get_base_url,
    get_intent_model_name,
    get_vertex_location,
    get_vertex_project,
    use_vertex_ai,
)


class IntentClassificationInput(BaseIOSchema):
    """Input for intent decomposition."""

    user_request: str = Field(..., description="The user's topic query")
    deep_research: bool = Field(
        default=False,
        description="Whether the user wants a deep-research pass",
    )


class IntentClassificationToolConfig(BaseToolConfig):
    """LLM configuration for intent decomposition."""

    model_name: str = Field(default_factory=get_intent_model_name)


def build_fallback_intent_decomposition(
    user_request: str, deep_research: bool
) -> IntentDecomposition:
    base_queries = [
        SearchIntent(
            label="Core entities",
            query=user_request,
            purpose="Find the main entities directly related to the user query.",
        ),
        SearchIntent(
            label="Official sources",
            query=f"{user_request} official websites",
            purpose="Find official sites or canonical pages for the entities.",
        ),
        SearchIntent(
            label="Comparison pages",
            query=f"{user_request} best top list",
            purpose="Find curated lists or comparison sources with multiple entities.",
        ),
    ]

    if deep_research:
        base_queries.extend(
            [
                SearchIntent(
                    label="Reviews and analysis",
                    query=f"{user_request} reviews comparison",
                    purpose="Find second-order sources that compare or evaluate entities.",
                ),
                SearchIntent(
                    label="Directories",
                    query=f"{user_request} directory database",
                    purpose="Find directories or databases listing many matching entities.",
                ),
                SearchIntent(
                    label="Location and metadata",
                    query=f"{user_request} location pricing features",
                    purpose="Find pages that expose comparable attributes for each entity.",
                ),
            ]
        )

    return IntentDecomposition(
        user_query=user_request,
        research_depth="deep" if deep_research else "standard",
        entity_type="entity",
        intent_summary=(
            "Discover entities matching the topic query and compare them using "
            "source-backed attributes."
        ),
        comparison_axes=["name", "summary", "location", "website"],
        search_requests=base_queries,
    )


class IntentClassificationTool(
    BaseTool[IntentClassificationInput, IntentDecomposition]
):
    """Decompose a user query into research-ready sub-intents."""

    def __init__(
        self, config: IntentClassificationToolConfig = IntentClassificationToolConfig()
    ):
        super().__init__(config)
        self.model_name = config.model_name

    def _run_vertex(self, params: IntentClassificationInput) -> IntentDecomposition:
        from google import genai
        from google.genai import types

        project = get_vertex_project()
        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT must be set when GOOGLE_GENAI_USE_VERTEXAI=True."
            )

        client = genai.Client(
            vertexai=True,
            project=project,
            location=get_vertex_location(),
            http_options=types.HttpOptions(
                api_version="v1",
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

        system_prompt = (
            "You decompose a user topic query into a compact research plan for an "
            "entity-discovery pipeline. Return valid JSON matching the provided schema."
        )
        user_prompt = f"""
User query:
{params.user_request}

Deep research requested:
{params.deep_research}

Return an IntentDecomposition with:
- research_depth = "deep" if deep_research is true, otherwise "standard"
- 3 to 5 search_requests for standard research
- 6 to 8 search_requests for deep research
- non-overlapping search queries
- comparison_axes that help compare entities for this topic
- a specific entity_type and intent_summary
"""

        response = client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                response_mime_type="application/json",
                response_schema=IntentDecomposition,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, IntentDecomposition):
            return parsed
        if parsed is not None:
            return IntentDecomposition.model_validate(parsed)
        if response.text:
            return IntentDecomposition.model_validate_json(response.text)
        raise ValueError("Vertex AI returned no parseable IntentDecomposition.")

    def _run_openai(self, params: IntentClassificationInput) -> IntentDecomposition:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY or PROVIDER_API_KEY is required.")

        client = OpenAI(
            api_key=api_key,
            base_url=get_base_url(),
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decompose a user topic query into a compact research plan "
                        "for an entity-discovery pipeline. Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User query:\n{params.user_request}\n\n"
                        f"Deep research requested:\n{params.deep_research}\n\n"
                        "Return an IntentDecomposition with:\n"
                        '- research_depth = "deep" if deep_research is true, '
                        'otherwise "standard"\n'
                        "- 3 to 5 search_requests for standard research\n"
                        "- 6 to 8 search_requests for deep research\n"
                        "- non-overlapping search queries\n"
                        "- comparison_axes that help compare entities for this topic\n"
                        "- a specific entity_type and intent_summary\n"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        return IntentDecomposition.model_validate_json(content)

    def run(self, params: IntentClassificationInput) -> IntentDecomposition:
        try:
            if use_vertex_ai():
                return self._run_vertex(params)
            return self._run_openai(params)
        except Exception:
            return build_fallback_intent_decomposition(
                params.user_request,
                params.deep_research,
            )

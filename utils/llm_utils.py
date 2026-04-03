import json
import logging
from typing import Any

from openai import OpenAI

from schema import AgentAction, InformationAgentInput
from utils.config import (
    get_api_key,
    get_base_url,
    get_model_name,
    get_vertex_location,
    get_vertex_project,
    use_vertex_ai,
)

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        model: str,
        system_prompt: str,
        project: str,
        location: str,
        http_options=None,
    ):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=http_options
            or types.HttpOptions(
                api_version="v1",
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
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
            f"Recursive research enabled: {self.initial_input.recursive_research}",
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


def run_structured_generation(
    response_schema: type[Any],
    system_prompt: str,
    user_prompt: str,
    model_name: str | None = None,
    request_timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
) -> Any:
    resolved_model_name = model_name or get_model_name()

    if use_vertex_ai():
        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True,
            project=get_vertex_project(),
            location=get_vertex_location(),
            http_options=types.HttpOptions(
                api_version="v1",
                timeout=(
                    int(request_timeout_seconds * 1000)
                    if request_timeout_seconds is not None
                    else None
                ),
                retry_options=types.HttpRetryOptions(
                    attempts=1,
                ),
            ),
        )
        response = client.models.generate_content(
            model=resolved_model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                response_mime_type="application/json",
                response_schema=response_schema,
                max_output_tokens=max_output_tokens,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, response_schema):
            return parsed
        if parsed is not None:
            return response_schema.model_validate(parsed)
        if response.text:
            return response_schema.model_validate_json(response.text)
        raise ValueError(f"No parseable response returned for {response_schema.__name__}.")

    api_key = get_api_key()
    if not api_key:
        raise ValueError("No API key available for structured generation.")

    client = OpenAI(
        api_key=api_key,
        base_url=get_base_url(),
        timeout=request_timeout_seconds,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=resolved_model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return response_schema.model_validate_json(content)

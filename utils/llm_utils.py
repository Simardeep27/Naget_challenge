import hashlib
import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import OpenAI

from utils.config import (
    get_api_key,
    get_base_url,
    get_model_name,
    get_vertex_cache_dir,
    get_vertex_cache_ttl_seconds,
    get_vertex_location,
    get_vertex_project,
    use_vertex_ai,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_vertex_client(
    project: str,
    location: str,
    timeout_ms: int | None,
):
    from google import genai
    from google.genai import types

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(
            api_version="v1",
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


@lru_cache(maxsize=8)
def _get_openai_client(
    api_key: str,
    base_url: str | None,
) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
    )


def _vertex_cache_path(cache_key: str) -> Path:
    cache_dir = get_vertex_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _load_vertex_cache_entry(
    *,
    cache_key: str,
    model_name: str,
) -> dict[str, Any] | None:
    cache_path = _vertex_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if str(payload.get("model_name") or "") != model_name:
        return None

    expires_at = float(payload.get("expires_at") or 0)
    if expires_at and time.time() > expires_at:
        return None

    cache_name = str(payload.get("cache_name") or "")
    if not cache_name:
        return None

    return payload


def _write_vertex_cache_entry(
    *,
    cache_key: str,
    model_name: str,
    cache_name: str,
    ttl_seconds: int,
) -> None:
    cache_path = _vertex_cache_path(cache_key)
    payload = {
        "cache_name": cache_name,
        "model_name": model_name,
        "expires_at": time.time() + ttl_seconds,
    }
    try:
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return


def _delete_vertex_cache_entry(cache_key: str) -> None:
    cache_path = _vertex_cache_path(cache_key)
    try:
        cache_path.unlink(missing_ok=True)
    except OSError:
        return


_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _extract_json_text(raw_text: str) -> str:
    cleaned = _CONTROL_CHAR_PATTERN.sub("", raw_text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    first_object = cleaned.find("{")
    last_object = cleaned.rfind("}")
    if first_object != -1 and last_object != -1 and last_object > first_object:
        return cleaned[first_object : last_object + 1]

    first_array = cleaned.find("[")
    last_array = cleaned.rfind("]")
    if first_array != -1 and last_array != -1 and last_array > first_array:
        return cleaned[first_array : last_array + 1]

    return cleaned


def _repair_json_output(
    *,
    response_schema: type[Any],
    raw_text: str,
    model_name: str,
    request_timeout_seconds: float | None,
    max_output_tokens: int | None,
) -> Any:
    return run_structured_generation(
        response_schema=response_schema,
        system_prompt=(
            "Repair malformed JSON so it exactly matches the requested schema. "
            "Do not add new factual claims. Preserve only information already present in the input."
        ),
        user_prompt=f"Malformed JSON to repair:\n{raw_text}\n",
        model_name=model_name,
        request_timeout_seconds=min(request_timeout_seconds or 30, 30),
        max_output_tokens=max_output_tokens,
        google_search=False,
    )


def run_structured_generation(
    response_schema: type[Any],
    system_prompt: str,
    user_prompt: str,
    model_name: str | None = None,
    request_timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
    google_search: bool = False,
    vertex_cache_key: str | None = None,
    vertex_cache_ttl_seconds: int | None = None,
) -> Any:
    """Run one structured generation call for a deterministic pipeline stage."""

    resolved_model_name = model_name or get_model_name()
    vertex_enabled = use_vertex_ai()

    if google_search and not vertex_enabled:
        raise ValueError(
            "google_search grounded structured generation requires the Vertex AI backend."
        )

    if vertex_enabled:
        from google.genai import types

        client = _get_vertex_client(
            get_vertex_project(),
            get_vertex_location(),
            (
                int(request_timeout_seconds * 1000)
                if request_timeout_seconds is not None
                else None
            ),
        )
        direct_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=(None if google_search else response_schema),
            max_output_tokens=max_output_tokens,
            tools=(
                [types.Tool(google_search=types.GoogleSearch())]
                if google_search
                else None
            ),
        )

        def generate_without_cache():
            return client.models.generate_content(
                model=resolved_model_name,
                contents=user_prompt,
                config=direct_config,
            )

        if vertex_cache_key and not google_search:
            ttl_seconds = max(
                60,
                vertex_cache_ttl_seconds or get_vertex_cache_ttl_seconds(),
            )
            try:
                cache_entry = _load_vertex_cache_entry(
                    cache_key=vertex_cache_key,
                    model_name=resolved_model_name,
                )
                cache_name = str(cache_entry.get("cache_name") or "") if cache_entry else ""

                if not cache_name:
                    cached_content = client.caches.create(
                        model=resolved_model_name,
                        config=types.CreateCachedContentConfig(
                            display_name=(
                                "info-agent-"
                                f"{hashlib.sha256(vertex_cache_key.encode('utf-8')).hexdigest()[:16]}"
                            ),
                            system_instruction=system_prompt,
                            contents=user_prompt,
                            ttl=f"{ttl_seconds}s",
                        ),
                    )
                    cache_name = str(cached_content.name or "")
                    if cache_name:
                        _write_vertex_cache_entry(
                            cache_key=vertex_cache_key,
                            model_name=resolved_model_name,
                            cache_name=cache_name,
                            ttl_seconds=ttl_seconds,
                        )

                try:
                    response = client.models.generate_content(
                        model=resolved_model_name,
                        contents="Generate the requested JSON response from the cached context.",
                        config=types.GenerateContentConfig(
                            temperature=0,
                            response_mime_type="application/json",
                            response_schema=response_schema,
                            max_output_tokens=max_output_tokens,
                            cached_content=cache_name,
                        ),
                    )
                except Exception:
                    _delete_vertex_cache_entry(vertex_cache_key)
                    cached_content = client.caches.create(
                        model=resolved_model_name,
                        config=types.CreateCachedContentConfig(
                            display_name=(
                                "info-agent-"
                                f"{hashlib.sha256(vertex_cache_key.encode('utf-8')).hexdigest()[:16]}"
                            ),
                            system_instruction=system_prompt,
                            contents=user_prompt,
                            ttl=f"{ttl_seconds}s",
                        ),
                    )
                    cache_name = str(cached_content.name or "")
                    if cache_name:
                        _write_vertex_cache_entry(
                            cache_key=vertex_cache_key,
                            model_name=resolved_model_name,
                            cache_name=cache_name,
                            ttl_seconds=ttl_seconds,
                        )
                    response = client.models.generate_content(
                        model=resolved_model_name,
                        contents="Generate the requested JSON response from the cached context.",
                        config=types.GenerateContentConfig(
                            temperature=0,
                            response_mime_type="application/json",
                            response_schema=response_schema,
                            max_output_tokens=max_output_tokens,
                            cached_content=cache_name,
                        ),
                    )
            except Exception as exc:
                logger.warning(
                    "Vertex cached-content generation failed for %s; retrying without cache: %s: %s",
                    response_schema.__name__,
                    type(exc).__name__,
                    exc,
                )
                _delete_vertex_cache_entry(vertex_cache_key)
                response = generate_without_cache()
        else:
            response = generate_without_cache()

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, response_schema):
            return parsed
        if parsed is not None:
            return response_schema.model_validate(parsed)
        if response.text:
            raw_text = _extract_json_text(response.text)
            try:
                return response_schema.model_validate_json(raw_text)
            except Exception:
                if google_search:
                    return _repair_json_output(
                        response_schema=response_schema,
                        raw_text=raw_text,
                        model_name=resolved_model_name,
                        request_timeout_seconds=request_timeout_seconds,
                        max_output_tokens=max_output_tokens,
                    )
                raise
        raise ValueError(f"No parseable response returned for {response_schema.__name__}.")

    api_key = get_api_key()
    if not api_key:
        raise ValueError("No API key available for structured generation.")

    client = _get_openai_client(
        api_key=api_key,
        base_url=get_base_url(),
    )
    request_kwargs: dict[str, Any] = {
        "model": resolved_model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "timeout": request_timeout_seconds,
    }
    if max_output_tokens is not None:
        request_kwargs["max_completion_tokens"] = max_output_tokens

    response = client.chat.completions.create(
        **request_kwargs,
    )
    content = response.choices[0].message.content or "{}"
    return response_schema.model_validate_json(content)

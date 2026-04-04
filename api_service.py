from __future__ import annotations

import json
import os
from enum import Enum
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

os.environ.setdefault("OUTPUT_DIR", "/tmp/information_agent")

from info_agent import run_information_agent


class ResearchMethod(str, Enum):
    standard = "standard"
    deep = "deep"
    lightning = "lightning"


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User query to research")
    method: ResearchMethod = Field(
        default=ResearchMethod.deep,
        description="Research strategy to run",
    )
    recursive_research: bool = Field(
        default=False,
        description="Whether to run targeted backfill after the main pass",
    )


app = FastAPI(
    title="Information Agent API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _validate_research_request(payload: ResearchRequest) -> str:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    if payload.method == ResearchMethod.lightning and payload.recursive_research:
        raise HTTPException(
            status_code=400,
            detail="Lightning mode cannot be combined with recursive research.",
        )

    return query


def _run_research(
    *,
    query: str,
    payload: ResearchRequest,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    try:
        raw_result = run_information_agent(
            information_request=query,
            deep_research=(payload.method == ResearchMethod.deep),
            recursive_research=payload.recursive_research,
            lightning=(payload.method == ResearchMethod.lightning),
            progress_callback=progress_callback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Research run failed: {type(exc).__name__}: {exc}",
        ) from exc

    return json.loads(raw_result)


def _encode_stream_event(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def _stream_research(
    *,
    query: str,
    payload: ResearchRequest,
) -> Iterator[bytes]:
    event_queue: Queue[dict[str, Any] | None] = Queue()

    def emit(
        event_type: str,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {"type": event_type}
        if message is not None:
            event["message"] = message
        if data is not None:
            event["data"] = data
        event_queue.put(event)

    def worker() -> None:
        try:
            result = _run_research(
                query=query,
                payload=payload,
                progress_callback=lambda message: emit("progress", message=message),
            )
            emit("result", data=result)
        except HTTPException as exc:
            emit("error", message=str(exc.detail))
        except Exception as exc:  # pragma: no cover - defensive fallback
            emit("error", message=f"Research run failed: {type(exc).__name__}: {exc}")
        finally:
            event_queue.put(None)

    Thread(target=worker, daemon=True).start()

    while True:
        try:
            event = event_queue.get(timeout=10)
        except Empty:
            yield _encode_stream_event({"type": "heartbeat"})
            continue

        if event is None:
            yield _encode_stream_event({"type": "done"})
            break

        yield _encode_stream_event(event)


@app.post("/research")
def research(payload: ResearchRequest) -> dict[str, Any]:
    query = _validate_research_request(payload)
    return _run_research(query=query, payload=payload)


@app.post("/research/stream")
def research_stream(payload: ResearchRequest) -> StreamingResponse:
    query = _validate_research_request(payload)
    return StreamingResponse(
        _stream_research(query=query, payload=payload),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

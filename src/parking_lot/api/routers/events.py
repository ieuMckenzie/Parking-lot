"""Server-Sent Events endpoint for live detection streaming."""

import asyncio
import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from parking_lot.api.deps import get_engine
from parking_lot.engine.scanner import ScannerEngine

router = APIRouter(tags=["events"])


@router.get("/events")
async def event_stream(
    request: Request,
    last_n: int = Query(0, ge=0, le=50, description="Replay the last N events on connect"),
    engine: ScannerEngine = Depends(get_engine),
):
    """SSE endpoint. Streams detection events as they happen.

    Event format:
        event: detection
        data: {"camera_id": "cam0", "value": "ABC1234", ...}

    Connect with EventSource:
        const es = new EventSource("/api/v1/events");
        es.addEventListener("detection", (e) => console.log(JSON.parse(e.data)));
    """
    loop = asyncio.get_event_loop()
    sub_id, queue = engine.events.subscribe(loop)

    async def generate():
        try:
            # Replay recent events if requested
            if last_n > 0:
                for event in engine.events.get_recent(last_n):
                    yield _format_sse(event)

            # Stream live events
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield _format_sse(event)
                except asyncio.TimeoutError:
                    # Send keepalive comment to prevent proxy/browser timeouts
                    yield ": keepalive\n\n"
        finally:
            engine.events.unsubscribe(sub_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


def _format_sse(event: dict) -> str:
    """Format a dict as an SSE message."""
    event_type = event.get("type", "message")
    data = json.dumps(event)
    return f"event: {event_type}\ndata: {data}\n\n"

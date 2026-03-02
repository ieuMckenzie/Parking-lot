"""Thread-safe event bus for broadcasting detection events to SSE clients."""

import asyncio
import threading
import time


class EventBus:
    """Fan-out pub/sub: worker threads publish events, async SSE handlers consume them.

    Each SSE client gets its own asyncio.Queue. When an OCR worker publishes
    an event, it's pushed to every active subscriber queue.
    """

    def __init__(self, maxlen: int = 50):
        self._lock = threading.Lock()
        self._subscribers: dict[int, asyncio.Queue] = {}
        self._counter = 0
        self._history: list[dict] = []
        self._maxlen = maxlen

    def publish(self, event: dict):
        """Called from worker threads. Pushes event to all subscriber queues."""
        event.setdefault("timestamp", time.time())
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._maxlen:
                self._history = self._history[-self._maxlen:]
            for q in self._subscribers.values():
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # Slow consumer — drop oldest and push new
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(event)
                    except asyncio.QueueFull:
                        pass

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> tuple[int, asyncio.Queue]:
        """Register a new subscriber. Returns (subscriber_id, queue)."""
        q = asyncio.Queue(maxsize=100)
        with self._lock:
            self._counter += 1
            sub_id = self._counter
            self._subscribers[sub_id] = q
        return sub_id, q

    def unsubscribe(self, sub_id: int):
        """Remove a subscriber."""
        with self._lock:
            self._subscribers.pop(sub_id, None)

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get the last N events (for clients that just connected)."""
        with self._lock:
            return list(self._history[-n:])

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

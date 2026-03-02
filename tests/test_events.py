"""Tests for parking_lot.engine.events.EventBus."""

import asyncio
import threading
import time

import pytest

from parking_lot.engine.events import EventBus


@pytest.fixture
def bus():
    return EventBus(maxlen=10)


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestPublish:
    def test_publish_adds_timestamp(self, bus):
        bus.publish({"type": "detection", "value": "ABC123"})
        recent = bus.get_recent(1)
        assert len(recent) == 1
        assert "timestamp" in recent[0]

    def test_publish_preserves_existing_timestamp(self, bus):
        bus.publish({"type": "detection", "timestamp": 12345.0})
        recent = bus.get_recent(1)
        assert recent[0]["timestamp"] == 12345.0

    def test_history_respects_maxlen(self, bus):
        for i in range(20):
            bus.publish({"type": "detection", "value": f"PLATE{i}"})
        recent = bus.get_recent(50)
        assert len(recent) == 10  # maxlen=10
        assert recent[0]["value"] == "PLATE10"  # oldest kept
        assert recent[-1]["value"] == "PLATE19"  # newest


class TestSubscribe:
    def test_subscribe_returns_id_and_queue(self, bus, loop):
        sub_id, q = bus.subscribe(loop)
        assert isinstance(sub_id, int)
        assert isinstance(q, asyncio.Queue)

    def test_subscriber_count(self, bus, loop):
        assert bus.subscriber_count == 0
        id1, _ = bus.subscribe(loop)
        assert bus.subscriber_count == 1
        id2, _ = bus.subscribe(loop)
        assert bus.subscriber_count == 2
        bus.unsubscribe(id1)
        assert bus.subscriber_count == 1

    def test_unsubscribe_nonexistent_no_crash(self, bus):
        bus.unsubscribe(9999)  # should not raise


class TestFanOut:
    def test_event_reaches_subscriber(self, bus, loop):
        sub_id, q = bus.subscribe(loop)
        bus.publish({"type": "detection", "value": "ABC123"})
        assert not q.empty()
        event = q.get_nowait()
        assert event["value"] == "ABC123"

    def test_event_reaches_multiple_subscribers(self, bus, loop):
        _, q1 = bus.subscribe(loop)
        _, q2 = bus.subscribe(loop)
        bus.publish({"type": "detection", "value": "XYZ789"})
        e1 = q1.get_nowait()
        e2 = q2.get_nowait()
        assert e1["value"] == "XYZ789"
        assert e2["value"] == "XYZ789"

    def test_unsubscribed_client_stops_receiving(self, bus, loop):
        sub_id, q = bus.subscribe(loop)
        bus.unsubscribe(sub_id)
        bus.publish({"type": "detection", "value": "ABC123"})
        assert q.empty()

    def test_slow_consumer_drops_oldest(self, bus, loop):
        # Create a bus and subscriber with a small queue
        small_bus = EventBus(maxlen=200)
        _, q = small_bus.subscribe(loop)

        # Fill the queue to capacity (100)
        for i in range(100):
            small_bus.publish({"type": "detection", "value": f"OLD{i}"})

        # Publish one more — should drop oldest and add new
        small_bus.publish({"type": "detection", "value": "NEWEST"})

        # Queue should still be at capacity
        assert q.qsize() == 100

        # Drain and check the newest is there
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        assert events[-1]["value"] == "NEWEST"


class TestGetRecent:
    def test_empty_bus(self, bus):
        assert bus.get_recent(5) == []

    def test_get_fewer_than_available(self, bus):
        for i in range(5):
            bus.publish({"type": "detection", "value": f"P{i}"})
        recent = bus.get_recent(3)
        assert len(recent) == 3
        assert recent[0]["value"] == "P2"

    def test_get_more_than_available(self, bus):
        bus.publish({"type": "detection", "value": "ONLY"})
        recent = bus.get_recent(10)
        assert len(recent) == 1


class TestThreadSafety:
    def test_concurrent_publish_subscribe(self, bus, loop):
        errors = []

        def publisher():
            try:
                for i in range(100):
                    bus.publish({"type": "detection", "value": f"P{i}"})
            except Exception as e:
                errors.append(e)

        def subscriber():
            try:
                for _ in range(10):
                    sub_id, q = bus.subscribe(loop)
                    time.sleep(0.001)
                    bus.unsubscribe(sub_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=publisher),
            threading.Thread(target=publisher),
            threading.Thread(target=subscriber),
            threading.Thread(target=subscriber),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []

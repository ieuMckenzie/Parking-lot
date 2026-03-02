"""Tests for the FastAPI API layer using a mock engine."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from parking_lot.api.app import create_app
from parking_lot.api.deps import set_engine


@pytest.fixture
def client(mock_engine):
    app = create_app(mock_engine)
    # Override lifespan by setting engine directly
    set_engine(mock_engine)
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_fields(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "running"
        assert data["uptime_seconds"] == 42.5
        assert data["num_cameras"] == 1
        assert "yolo_queue_size" in data
        assert "ocr_queue_size" in data
        assert "sse_subscribers" in data


class TestCameras:
    def test_list_cameras(self, client):
        resp = client.get("/api/v1/cameras")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["cameras"]) == 1
        cam = data["cameras"][0]
        assert cam["id"] == "cam0"
        assert cam["connected"] is True
        assert cam["fps"] == 15.3

    def test_snapshot_returns_jpeg(self, client):
        resp = client.get("/api/v1/cameras/cam0/snapshot")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        # JPEG magic bytes
        assert resp.content[:2] == b"\xff\xd8"

    def test_snapshot_unknown_camera(self, client, mock_engine):
        mock_engine.get_snapshot.return_value = None
        resp = client.get("/api/v1/cameras/cam99/snapshot")
        assert resp.status_code == 404


class TestDetections:
    def test_current_detections(self, client):
        resp = client.get("/api/v1/detections")
        assert resp.status_code == 200
        data = resp.json()
        assert "cam0" in data["detections"]
        assert len(data["detections"]["cam0"]) == 1
        det = data["detections"]["cam0"][0]
        assert det["label"] == "licenseplate"
        assert det["conf"] == 0.92

    def test_detection_log(self, client):
        resp = client.get("/api/v1/detections/log")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["Value"] == "ABC1234"

    def test_detection_log_limit(self, client):
        resp = client.get("/api/v1/detections/log?limit=5")
        assert resp.status_code == 200

    def test_export_log(self, client):
        resp = client.get("/api/v1/detections/log/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]


class TestConfig:
    def test_get_config(self, client):
        resp = client.get("/api/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min_thresh"] == 0.5
        assert data["sr_enabled"] is False
        assert data["num_ocr_workers"] == 2

    def test_update_thresholds(self, client, mock_engine):
        resp = client.put("/api/v1/config/thresholds", json={
            "min_thresh": 0.7,
            "ocr_thresh": 0.5,
        })
        assert resp.status_code == 200
        assert "updated" in resp.json()["message"].lower()
        # Verify the config was actually modified
        assert mock_engine.cfg.min_thresh == 0.7
        assert mock_engine.cfg.ocr.thresh == 0.5

    def test_update_partial_thresholds(self, client, mock_engine):
        original_cooldown = mock_engine.cfg.detection.log_cooldown
        resp = client.put("/api/v1/config/thresholds", json={
            "min_thresh": 0.6,
        })
        assert resp.status_code == 200
        assert mock_engine.cfg.min_thresh == 0.6
        # Unset fields should remain unchanged
        assert mock_engine.cfg.detection.log_cooldown == original_cooldown


class TestAuthorized:
    def test_list_authorized(self, client):
        resp = client.get("/api/v1/config/authorized")
        assert resp.status_code == 200
        data = resp.json()
        assert "ABC1234" in data["plates"]

    def test_add_authorized(self, client):
        resp = client.post("/api/v1/config/authorized", json={"plate": "NEW1234"})
        assert resp.status_code == 200
        assert "added" in resp.json()["message"].lower()

        # Verify it shows in the list
        resp2 = client.get("/api/v1/config/authorized")
        assert "NEW1234" in resp2.json()["plates"]

    def test_add_duplicate(self, client):
        resp = client.post("/api/v1/config/authorized", json={"plate": "ABC1234"})
        assert resp.status_code == 200
        assert "already" in resp.json()["message"].lower()

    def test_add_empty_plate(self, client):
        resp = client.post("/api/v1/config/authorized", json={"plate": "  "})
        assert resp.status_code == 400

    def test_remove_authorized(self, client):
        resp = client.delete("/api/v1/config/authorized/ABC1234")
        assert resp.status_code == 200
        assert "removed" in resp.json()["message"].lower()

    def test_remove_nonexistent(self, client):
        resp = client.delete("/api/v1/config/authorized/DOESNOTEXIST")
        assert resp.status_code == 404


class TestSSE:
    """Test SSE format and route registration.

    Note: SSE streaming can't be reliably tested with FastAPI's TestClient
    because the ASGI transport doesn't cleanly cancel long-lived generators.
    The EventBus pub/sub logic is tested thoroughly in test_events.py.
    Here we test the format helper and verify the route exists.
    """

    def test_format_sse_detection(self):
        from parking_lot.api.routers.events import _format_sse

        event = {"type": "detection", "camera_id": "cam0", "value": "ABC123"}
        result = _format_sse(event)
        lines = result.strip().split("\n")
        assert lines[0] == "event: detection"
        assert lines[1].startswith("data: ")
        assert '"ABC123"' in lines[1]
        # Must end with double newline (SSE spec)
        assert result.endswith("\n\n")

    def test_format_sse_custom_type(self):
        from parking_lot.api.routers.events import _format_sse

        event = {"type": "alert", "message": "gate opened"}
        result = _format_sse(event)
        assert result.startswith("event: alert\n")

    def test_format_sse_default_type(self):
        from parking_lot.api.routers.events import _format_sse

        event = {"value": "no type field"}
        result = _format_sse(event)
        assert result.startswith("event: message\n")

    def test_events_route_registered(self, client):
        """The /events route should exist (not 404/405)."""
        # Use a regular GET — it will start streaming, but we just check
        # the route is reachable by verifying the OpenAPI schema includes it
        openapi = client.get("/openapi.json").json()
        paths = openapi["paths"]
        assert "/api/v1/events" in paths
        assert "get" in paths["/api/v1/events"]

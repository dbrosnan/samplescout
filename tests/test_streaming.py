"""Tests for streaming microphone recording (WebSocket + waveform helpers)."""

import json

import numpy as np
import pytest


class TestConcatTail:
    """Tests for the _concat_tail buffer helper."""

    def test_single_chunk(self):
        """Return entire chunk when it matches requested size."""
        from src.web.api import _concat_tail

        chunks = [np.ones(100, dtype=np.float32)]
        result = _concat_tail(chunks, 100)
        assert len(result) == 100
        assert np.all(result == 1.0)

    def test_tail_from_multiple_chunks(self):
        """Extract exactly n_samples from the tail of multiple chunks."""
        from src.web.api import _concat_tail

        c1 = np.full(50, 1.0, dtype=np.float32)
        c2 = np.full(50, 2.0, dtype=np.float32)
        c3 = np.full(50, 3.0, dtype=np.float32)
        chunks = [c1, c2, c3]

        result = _concat_tail(chunks, 80)
        assert len(result) == 80
        # Last 80 samples = last 50 (3.0) + last 30 of c2 (2.0)
        assert np.all(result[:30] == 2.0)
        assert np.all(result[30:] == 3.0)

    def test_tail_larger_than_buffer(self):
        """When n_samples > total buffer, return everything."""
        from src.web.api import _concat_tail

        chunks = [np.ones(30, dtype=np.float32), np.ones(20, dtype=np.float32)]
        result = _concat_tail(chunks, 100)
        assert len(result) == 50  # only 50 samples exist

    def test_empty_chunks(self):
        """Empty chunk list returns empty array."""
        from src.web.api import _concat_tail

        result = _concat_tail([], 100)
        assert len(result) == 0


class TestClassifyWaveformUnit:
    """Unit-level tests for classify_waveform (no model load)."""

    def test_normalize_large_waveform(self):
        """classify_waveform should normalize waveforms outside [-1,1]."""
        from unittest.mock import patch, MagicMock
        from src.classifier import YAMNetClassifier

        classifier = YAMNetClassifier()

        # Mock the model so we don't need TF Hub
        mock_model = MagicMock()
        mock_model.return_value = (
            MagicMock(numpy=MagicMock(return_value=np.random.rand(2, 521).astype(np.float32))),
            MagicMock(numpy=MagicMock(return_value=np.random.rand(2, 1024).astype(np.float32))),
            MagicMock(numpy=MagicMock(return_value=np.random.rand(2, 64).astype(np.float32))),
        )
        mock_model.class_map_path.return_value.numpy.return_value = b""

        with patch.object(type(classifier), 'model', new_callable=lambda: property(lambda self: mock_model)):
            with patch.object(type(classifier), 'class_names', new_callable=lambda: property(lambda self: [f"class_{i}" for i in range(521)])):
                # Waveform with values > 1.0
                waveform = np.full(16000, 5.0, dtype=np.float32)
                result = classifier.classify_waveform(waveform, sr=16000, top_k=3)

                assert result is not None
                assert len(result.top_classes) <= 3


class TestWebSocketProtocol:
    """Test the WebSocket endpoint protocol (requires httpx / TestClient)."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        try:
            from starlette.testclient import TestClient
            from src.web.api import app
            return TestClient(app)
        except ImportError:
            pytest.skip("starlette TestClient not available")

    def test_websocket_start_stop(self, client):
        """Test basic start/stop protocol without audio data."""
        with client.websocket_connect("/ws/stream") as ws:
            # Start session
            ws.send_text(json.dumps({"action": "start", "sample_rate": 16000}))
            resp = ws.receive_json()
            assert resp["type"] == "started"
            assert resp["sample_rate"] == 16000

            # Stop without save
            ws.send_text(json.dumps({"action": "stop", "save": False}))
            resp = ws.receive_json()
            assert resp["type"] == "stopped"

    def test_websocket_stop_and_save(self, client):
        """Test sending audio then stopping with save=true."""
        with client.websocket_connect("/ws/stream") as ws:
            # Start
            ws.send_text(json.dumps({"action": "start", "sample_rate": 16000}))
            resp = ws.receive_json()
            assert resp["type"] == "started"

            # Send 2 seconds of sine wave as PCM
            t = np.linspace(0, 2.0, 32000, dtype=np.float32)
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            ws.send_bytes(audio.tobytes())

            # Stop and save
            ws.send_text(json.dumps({"action": "stop", "save": True}))

            # Collect responses — may get classification(s) then saved
            saved = None
            for _ in range(10):
                msg = ws.receive_json()
                if msg["type"] == "saved":
                    saved = msg
                    break
                elif msg["type"] == "classification":
                    assert "top_classes" in msg
                    continue

            assert saved is not None
            assert "file" in saved
            assert saved["file"]["id"]
            assert saved["file"]["path"].endswith(".wav")

"""Tests for utility functions."""

import numpy as np
import pytest

from src.utils import load_audio, save_audio, get_audio_info, normalize_audio


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_load_mono(self, test_audio, sample_rate):
        """Test loading audio as mono."""
        audio, sr = load_audio(test_audio, sr=sample_rate, mono=True)

        assert sr == sample_rate
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_load_with_resample(self, test_audio):
        """Test loading with resampling."""
        target_sr = 22050
        audio, sr = load_audio(test_audio, sr=target_sr, mono=True)

        assert sr == target_sr

    def test_load_duration_limit(self, test_audio, sample_rate):
        """Test loading with duration limit."""
        duration = 1.0
        audio, sr = load_audio(test_audio, sr=sample_rate, duration=duration)

        expected_samples = int(sample_rate * duration)
        assert len(audio) == expected_samples


class TestSaveAudio:
    """Tests for save_audio function."""

    def test_save_and_reload(self, tmp_path, sample_rate):
        """Test saving and reloading audio."""
        # Create test audio
        audio = np.sin(np.linspace(0, 10, sample_rate)).astype(np.float32)

        # Save
        path = tmp_path / "output.wav"
        save_audio(path, audio, sample_rate)

        # Reload
        loaded, sr = load_audio(path, sr=None, mono=True)

        assert sr == sample_rate
        assert len(loaded) == len(audio)
        np.testing.assert_allclose(audio, loaded, atol=1e-4)


class TestGetAudioInfo:
    """Tests for get_audio_info function."""

    def test_basic_info(self, test_audio, sample_rate, audio_duration):
        """Test getting basic audio info."""
        info = get_audio_info(test_audio)

        assert "duration" in info
        assert "sample_rate" in info
        assert "channels" in info

        assert info["sample_rate"] == sample_rate
        assert abs(info["duration"] - audio_duration) < 0.1


class TestNormalizeAudio:
    """Tests for normalize_audio function."""

    def test_normalize(self):
        """Test audio normalization."""
        audio = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        normalized = normalize_audio(audio)

        assert np.max(np.abs(normalized)) == pytest.approx(1.0)

    def test_normalize_already_normalized(self):
        """Test normalizing already normalized audio."""
        audio = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        normalized = normalize_audio(audio)

        np.testing.assert_allclose(audio, normalized)

    def test_normalize_silent(self):
        """Test normalizing silent audio."""
        audio = np.zeros(100, dtype=np.float32)
        normalized = normalize_audio(audio)

        np.testing.assert_allclose(normalized, audio)

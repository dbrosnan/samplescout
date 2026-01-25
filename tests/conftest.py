"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 16000


@pytest.fixture
def audio_duration():
    """Duration of test audio in seconds."""
    return 3.0


@pytest.fixture
def test_audio(sample_rate, audio_duration, tmp_path):
    """Create a test audio file with a simple sine wave."""
    from scipy.io import wavfile

    # Generate sine wave
    t = np.linspace(0, audio_duration, int(sample_rate * audio_duration))
    # Mix of frequencies to simulate music
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t)  # A4
        + 0.2 * np.sin(2 * np.pi * 554 * t)  # C#5
        + 0.2 * np.sin(2 * np.pi * 659 * t)  # E5
        + 0.1 * np.sin(2 * np.pi * 110 * t)  # Bass
    )

    # Add some "noise" for percussion-like content
    audio += 0.1 * np.random.randn(len(audio))

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    audio = audio.astype(np.float32)

    # Save
    path = tmp_path / "test_audio.wav"
    wavfile.write(str(path), sample_rate, audio)

    return path


@pytest.fixture
def test_stereo_audio(sample_rate, audio_duration, tmp_path):
    """Create a stereo test audio file."""
    from scipy.io import wavfile

    t = np.linspace(0, audio_duration, int(sample_rate * audio_duration))

    # Left channel - lower frequencies
    left = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 110 * t)

    # Right channel - higher frequencies
    right = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)

    audio = np.stack([left, right], axis=1).astype(np.float32)

    path = tmp_path / "test_stereo.wav"
    wavfile.write(str(path), sample_rate, audio)

    return path


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    path = tmp_path / "output"
    path.mkdir()
    return path

"""Tests for audio separators."""

import numpy as np
import pytest

from src.separators import get_separator
from src.separators.base import BaseSeparator, SeparationResult


class TestSeparationResult:
    """Tests for SeparationResult dataclass."""

    def test_creation(self):
        """Test creating a SeparationResult."""
        stems = {
            "vocals": np.random.randn(16000).astype(np.float32),
            "drums": np.random.randn(16000).astype(np.float32),
        }

        result = SeparationResult(
            stems=stems,
            sample_rate=16000,
            duration=1.5,
            separator="test",
            model="test-model",
        )

        assert result.stem_names == ["vocals", "drums"]
        assert result.sample_rate == 16000

    def test_get_stem(self):
        """Test getting a specific stem."""
        vocals = np.random.randn(16000).astype(np.float32)
        stems = {"vocals": vocals}

        result = SeparationResult(
            stems=stems,
            sample_rate=16000,
            duration=1.0,
            separator="test",
            model="test",
        )

        retrieved = result.get_stem("vocals")
        np.testing.assert_array_equal(retrieved, vocals)

        assert result.get_stem("nonexistent") is None

    def test_save_stems(self, tmp_path):
        """Test saving stems to disk."""
        stems = {
            "vocals": np.random.randn(16000).astype(np.float32),
            "drums": np.random.randn(16000).astype(np.float32),
        }

        result = SeparationResult(
            stems=stems,
            sample_rate=16000,
            duration=1.0,
            separator="test",
            model="test",
        )

        paths = result.save_stems(tmp_path, prefix="song")

        assert "vocals" in paths
        assert "drums" in paths
        assert paths["vocals"].exists()
        assert paths["drums"].exists()


class TestGetSeparator:
    """Tests for get_separator factory function."""

    def test_get_demucs(self):
        """Test getting demucs separator."""
        sep = get_separator("demucs")
        assert sep.name == "demucs"

    def test_get_spleeter(self):
        """Test getting spleeter separator."""
        sep = get_separator("spleeter")
        assert sep.name == "spleeter"

    def test_unknown_separator(self):
        """Test error on unknown separator."""
        with pytest.raises(ValueError):
            get_separator("unknown_separator")

    def test_case_insensitive(self):
        """Test separator name is case insensitive."""
        sep = get_separator("DEMUCS")
        assert sep.name == "demucs"


class TestDemucsSeparator:
    """Tests for DemucsSeparator."""

    @pytest.fixture
    def separator(self):
        """Create Demucs separator."""
        from src.separators.demucs import DemucsSeparator

        return DemucsSeparator(device="cpu")

    def test_init(self, separator):
        """Test separator initialization."""
        assert separator.name == "demucs"
        assert separator.device == "cpu"

    def test_available_stems(self, separator):
        """Test available stems."""
        stems = separator.AVAILABLE_STEMS
        assert "vocals" in stems
        assert "drums" in stems
        assert "bass" in stems


class TestSpleeterSeparator:
    """Tests for SpleeterSeparator."""

    @pytest.fixture
    def separator(self):
        """Create Spleeter separator."""
        from src.separators.spleeter import SpleeterSeparator

        return SpleeterSeparator(stems=4)

    def test_init(self, separator):
        """Test separator initialization."""
        assert separator.name == "spleeter"
        assert "4stems" in separator.model_name

    def test_stem_configs(self, separator):
        """Test stem configurations."""
        assert 2 in separator.STEM_CONFIGS
        assert 4 in separator.STEM_CONFIGS
        assert 5 in separator.STEM_CONFIGS

    def test_invalid_stems(self):
        """Test error on invalid stem count."""
        from src.separators.spleeter import SpleeterSeparator

        with pytest.raises(ValueError):
            SpleeterSeparator(stems=3)

    def test_available_stems_4stem(self, separator):
        """Test available stems for 4-stem model."""
        stems = separator.get_available_stems()
        assert stems == ["vocals", "drums", "bass", "other"]


class TestBaseSeparator:
    """Tests for BaseSeparator abstract class."""

    def test_detect_device(self):
        """Test device detection."""
        from src.separators.demucs import DemucsSeparator

        sep = DemucsSeparator(device=None)
        assert sep.device in ["cpu", "cuda", "mps"]

    def test_repr(self):
        """Test string representation."""
        from src.separators.demucs import DemucsSeparator

        sep = DemucsSeparator(device="cpu")
        repr_str = repr(sep)

        assert "DemucsSeparator" in repr_str
        assert "cpu" in repr_str

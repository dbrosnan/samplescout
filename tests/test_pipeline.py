"""Tests for SampleScout pipeline."""

import numpy as np
import pytest


class TestSampleScout:
    """Tests for main SampleScout pipeline."""

    @pytest.fixture
    def scout(self):
        """Create SampleScout instance."""
        from src.pipeline import SampleScout

        return SampleScout(separator_name="demucs", device="cpu")

    def test_init(self, scout):
        """Test pipeline initialization."""
        assert scout.separator_name == "demucs"

    def test_lazy_loading(self, scout):
        """Test that classifier and separator are lazy loaded."""
        # Should not be loaded yet
        assert scout._classifier is None
        assert scout._separator is None

    def test_classifier_property(self, scout):
        """Test classifier property creates classifier."""
        classifier = scout.classifier
        assert classifier is not None
        assert scout._classifier is not None

    def test_separator_property(self, scout):
        """Test separator property creates separator."""
        separator = scout.separator
        assert separator is not None
        assert scout._separator is not None

    def test_category_sets(self, scout):
        """Test category classifications are defined."""
        assert len(scout.MUSIC_CATEGORIES) > 0
        assert len(scout.SPEECH_CATEGORIES) > 0
        assert len(scout.PERCUSSION_CATEGORIES) > 0

        assert "Music" in scout.MUSIC_CATEGORIES
        assert "Speech" in scout.SPEECH_CATEGORIES
        assert "Drum" in scout.PERCUSSION_CATEGORIES


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    @pytest.fixture
    def mock_result(self, tmp_path):
        """Create a mock PipelineResult."""
        from src.pipeline import PipelineResult
        from src.classifier import ClassificationResult

        classification = ClassificationResult(
            scores=np.array([0.8, 0.1, 0.05, 0.05]),
            embeddings=np.random.randn(10, 1024),
            top_classes=[("Music", 0.8), ("Drum", 0.1)],
            embedding=np.random.randn(1024),
            processing_time=0.5,
        )

        return PipelineResult(
            audio_path=tmp_path / "test.wav",
            audio_info={"duration": 10.0, "sample_rate": 16000},
            classification=classification,
            processing_time=1.5,
        )

    def test_top_category(self, mock_result):
        """Test top_category property."""
        assert mock_result.top_category == "Music"

    def test_top_score(self, mock_result):
        """Test top_score property."""
        assert mock_result.top_score == 0.8

    def test_summary(self, mock_result):
        """Test summary method."""
        summary = mock_result.summary()

        assert "test.wav" in summary
        assert "Music" in summary
        assert "Processing Time" in summary

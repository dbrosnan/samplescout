"""Tests for YAMNet classifier."""

import numpy as np
import pytest


class TestYAMNetClassifier:
    """Tests for YAMNetClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        from src.classifier import YAMNetClassifier

        return YAMNetClassifier()

    def test_init(self, classifier):
        """Test classifier initialization."""
        assert classifier.SAMPLE_RATE == 16000
        assert classifier.MODEL_URL is not None

    def test_classify_returns_result(self, classifier, test_audio):
        """Test that classify returns a ClassificationResult."""
        from src.classifier import ClassificationResult

        result = classifier.classify(test_audio)

        assert isinstance(result, ClassificationResult)
        assert result.scores is not None
        assert result.embedding is not None
        assert len(result.top_classes) > 0

    def test_classify_top_k(self, classifier, test_audio):
        """Test top_k parameter."""
        result = classifier.classify(test_audio, top_k=5)

        assert len(result.top_classes) == 5

    def test_embedding_shape(self, classifier, test_audio):
        """Test embedding has correct shape."""
        result = classifier.classify(test_audio)

        # YAMNet produces 1024-dimensional embeddings
        assert result.embedding.shape == (1024,)

    def test_scores_normalized(self, classifier, test_audio):
        """Test that scores are probabilities (0-1)."""
        result = classifier.classify(test_audio)

        for name, score in result.top_classes:
            assert 0 <= score <= 1

    def test_classify_path_string(self, classifier, test_audio):
        """Test classify accepts string path."""
        result = classifier.classify(str(test_audio))

        assert result is not None


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_creation(self):
        """Test creating a ClassificationResult."""
        from src.classifier import ClassificationResult

        result = ClassificationResult(
            scores=np.array([0.5, 0.3, 0.2]),
            embeddings=np.random.randn(10, 1024),
            top_classes=[("Music", 0.5), ("Speech", 0.3)],
            embedding=np.random.randn(1024),
            processing_time=1.0,
        )

        assert result.top_classes[0][0] == "Music"
        assert result.processing_time == 1.0

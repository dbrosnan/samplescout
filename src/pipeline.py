"""
SampleScout Pipeline - Unified audio analysis and separation.

The pipeline combines classification (YAMNet) with source separation
(Demucs/Spleeter/AudioSep) for intelligent sample extraction.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.classifier import ClassificationResult, YAMNetClassifier
from src.separators import BaseSeparator, SeparationResult, get_separator
from src.utils import get_audio_info, load_audio, save_audio


@dataclass
class PipelineResult:
    """Result from the SampleScout pipeline."""

    audio_path: Path
    audio_info: Dict[str, Any]
    classification: ClassificationResult
    separation: Optional[SeparationResult] = None
    extracted_samples: Dict[str, np.ndarray] = field(default_factory=dict)
    processing_time: float = 0.0
    output_paths: Dict[str, Path] = field(default_factory=dict)

    @property
    def top_category(self) -> str:
        """Get the top classification category."""
        if self.classification.top_classes:
            return self.classification.top_classes[0][0]
        return "unknown"

    @property
    def top_score(self) -> float:
        """Get the top classification score."""
        if self.classification.top_classes:
            return self.classification.top_classes[0][1]
        return 0.0

    def summary(self) -> str:
        """Get a summary of the pipeline result."""
        lines = [
            f"Audio: {self.audio_path.name}",
            f"Duration: {self.audio_info.get('duration', 0):.2f}s",
            f"Top Class: {self.top_category} ({self.top_score:.2%})",
            f"Processing Time: {self.processing_time:.2f}s",
        ]

        if self.separation:
            lines.append(f"Stems: {', '.join(self.separation.stem_names)}")

        if self.extracted_samples:
            lines.append(f"Extracted: {', '.join(self.extracted_samples.keys())}")

        return "\n".join(lines)


class SampleScout:
    """
    Main SampleScout pipeline for audio analysis and extraction.

    Combines YAMNet classification with source separation to
    intelligently analyze and extract audio samples.

    Example:
        scout = SampleScout()
        result = scout.analyze("song.mp3")
        print(result.top_category)  # "Music"

        # With separation
        result = scout.analyze("song.mp3", separate=True)
        result.separation.save_stems("output/")

        # Auto-extract based on classification
        result = scout.extract("song.mp3", auto=True)

    Attributes:
        classifier: YAMNet classifier instance
        separator: Audio separator instance (lazy loaded)
    """

    # Category mappings for intelligent extraction
    MUSIC_CATEGORIES = {
        "Music",
        "Musical instrument",
        "Singing",
        "Song",
        "Drum",
        "Guitar",
        "Piano",
        "Bass guitar",
        "Synthesizer",
    }

    SPEECH_CATEGORIES = {
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
    }

    PERCUSSION_CATEGORIES = {
        "Drum",
        "Drum kit",
        "Snare drum",
        "Bass drum",
        "Hi-hat",
        "Cymbal",
        "Percussion",
    }

    def __init__(
        self,
        separator_name: str = "demucs",
        separator_kwargs: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize SampleScout pipeline.

        Args:
            separator_name: Name of separator ('demucs', 'spleeter', 'audiosep')
            separator_kwargs: Additional kwargs for separator
            device: Device for inference ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.separator_name = separator_name
        self.separator_kwargs = separator_kwargs or {}
        self.device = device

        # Lazy-loaded components
        self._classifier: Optional[YAMNetClassifier] = None
        self._separator: Optional[BaseSeparator] = None

    @property
    def classifier(self) -> YAMNetClassifier:
        """Get or create the classifier."""
        if self._classifier is None:
            self._classifier = YAMNetClassifier()
        return self._classifier

    @property
    def separator(self) -> BaseSeparator:
        """Get or create the separator."""
        if self._separator is None:
            kwargs = {"device": self.device, **self.separator_kwargs}
            self._separator = get_separator(self.separator_name, **kwargs)
        return self._separator

    def analyze(
        self,
        audio_path: Union[str, Path],
        separate: bool = False,
        stems: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> PipelineResult:
        """
        Analyze audio with classification and optional separation.

        Args:
            audio_path: Path to input audio file
            separate: Whether to perform source separation
            stems: Specific stems to extract (if separating)
            top_k: Number of top classifications to return

        Returns:
            PipelineResult with analysis results
        """
        audio_path = Path(audio_path)
        start_time = time.time()

        # Get audio info
        audio_info = get_audio_info(audio_path)

        # Classify
        classification = self.classifier.classify(audio_path, top_k=top_k)

        # Optionally separate
        separation = None
        if separate:
            separation = self.separator.separate(audio_path, stems=stems)

        elapsed = time.time() - start_time

        return PipelineResult(
            audio_path=audio_path,
            audio_info=audio_info,
            classification=classification,
            separation=separation,
            processing_time=elapsed,
        )

    def extract(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        auto: bool = True,
        stems: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
    ) -> PipelineResult:
        """
        Extract samples from audio based on classification.

        In auto mode, the pipeline intelligently selects which stems
        to extract based on the detected content type.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save extracted samples
            auto: Automatically determine what to extract
            stems: Manual list of stems to extract (overrides auto)
            confidence_threshold: Minimum confidence for auto decisions

        Returns:
            PipelineResult with extracted samples and paths
        """
        audio_path = Path(audio_path)
        start_time = time.time()

        # Get audio info
        audio_info = get_audio_info(audio_path)

        # Classify to determine content
        classification = self.classifier.classify(audio_path, top_k=20)

        # Determine stems to extract
        if stems is not None:
            extract_stems = stems
        elif auto:
            extract_stems = self._auto_select_stems(
                classification, confidence_threshold
            )
        else:
            extract_stems = self.separator.get_available_stems()

        # Separate
        separation = self.separator.separate(audio_path, stems=extract_stems)

        # Save if output directory specified
        output_paths = {}
        if output_dir:
            output_dir = Path(output_dir)
            output_paths = separation.save_stems(output_dir, prefix=audio_path.stem)

        elapsed = time.time() - start_time

        return PipelineResult(
            audio_path=audio_path,
            audio_info=audio_info,
            classification=classification,
            separation=separation,
            extracted_samples=separation.stems,
            processing_time=elapsed,
            output_paths=output_paths,
        )

    def _auto_select_stems(
        self,
        classification: ClassificationResult,
        threshold: float,
    ) -> List[str]:
        """
        Automatically select stems based on classification.

        Args:
            classification: Classification result
            threshold: Confidence threshold

        Returns:
            List of stems to extract
        """
        # Get category scores
        category_scores = {}
        for name, score in classification.top_classes:
            category_scores[name] = score

        # Check for music content
        is_music = any(
            category_scores.get(cat, 0) > threshold for cat in self.MUSIC_CATEGORIES
        )

        # Check for speech content
        is_speech = any(
            category_scores.get(cat, 0) > threshold for cat in self.SPEECH_CATEGORIES
        )

        # Check for percussion
        is_percussion = any(
            category_scores.get(cat, 0) > threshold
            for cat in self.PERCUSSION_CATEGORIES
        )

        # Determine extraction strategy
        stems = []
        available = self.separator.get_available_stems()

        if is_music:
            # For music, extract all available stems
            stems = available
        elif is_speech:
            # For speech, prioritize vocals
            if "vocals" in available:
                stems = ["vocals"]
            else:
                stems = available
        elif is_percussion:
            # For percussion, prioritize drums
            if "drums" in available:
                stems = ["drums"]
            else:
                stems = available
        else:
            # Default: extract all
            stems = available

        return stems

    def batch_analyze(
        self,
        audio_paths: List[Union[str, Path]],
        separate: bool = False,
        **kwargs,
    ) -> List[PipelineResult]:
        """
        Analyze multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            separate: Whether to perform separation
            **kwargs: Additional arguments for analyze()

        Returns:
            List of PipelineResult objects
        """
        results = []
        for path in audio_paths:
            try:
                result = self.analyze(path, separate=separate, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        return results

    def batch_extract(
        self,
        audio_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        **kwargs,
    ) -> List[PipelineResult]:
        """
        Extract samples from multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            output_dir: Directory to save extracted samples
            **kwargs: Additional arguments for extract()

        Returns:
            List of PipelineResult objects
        """
        output_dir = Path(output_dir)
        results = []

        for path in audio_paths:
            try:
                # Create subdirectory for each file
                file_output_dir = output_dir / Path(path).stem
                result = self.extract(path, output_dir=file_output_dir, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        return results

    def get_embedding(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Get audio embedding from YAMNet.

        Useful for similarity search and clustering.

        Args:
            audio_path: Path to audio file

        Returns:
            1024-dimensional embedding vector
        """
        result = self.classifier.classify(audio_path)
        return result.embedding

    def find_similar(
        self,
        query_path: Union[str, Path],
        candidate_paths: List[Union[str, Path]],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find audio files most similar to a query.

        Uses cosine similarity on YAMNet embeddings.

        Args:
            query_path: Path to query audio
            candidate_paths: List of paths to compare against
            top_k: Number of results to return

        Returns:
            List of (path, similarity_score) tuples, sorted by similarity
        """
        # Get query embedding
        query_embedding = self.get_embedding(query_path)

        # Compute similarities
        similarities = []
        for path in candidate_paths:
            try:
                embedding = self.get_embedding(path)
                # Cosine similarity
                sim = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((path, float(sim)))
            except Exception as e:
                print(f"Error processing {path}: {e}")

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def __repr__(self) -> str:
        return f"SampleScout(separator={self.separator_name}, device={self.device})"

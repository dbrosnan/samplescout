"""
YAMNet Audio Classifier

Classifies audio into 521 sound categories using Google's YAMNet model.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from src.utils import load_audio


@dataclass
class ClassificationResult:
    """Result of audio classification."""

    scores: np.ndarray  # Shape: (num_frames, 521)
    embeddings: np.ndarray  # Shape: (num_frames, 1024)
    spectrogram: np.ndarray  # Log mel spectrogram
    top_classes: List[Tuple[str, float]]  # Top class names and scores
    all_classes: List[str]  # All 521 class names

    @property
    def top_sounds(self) -> List[str]:
        """Get top sound category names."""
        return [name for name, _ in self.top_classes]

    @property
    def top_class(self) -> str:
        """Get the most likely sound category."""
        return self.top_classes[0][0] if self.top_classes else "unknown"

    @property
    def confidence(self) -> float:
        """Get confidence score for top class."""
        return self.top_classes[0][1] if self.top_classes else 0.0


class YAMNetClassifier:
    """
    Audio classifier using YAMNet (Yet Another Mobile Network).

    YAMNet is trained on AudioSet to predict 521 audio event classes.
    It uses the MobileNetV1 architecture and operates on 0.96s audio frames.

    Example:
        classifier = YAMNetClassifier()
        result = classifier.classify("audio.wav")
        print(f"Detected: {result.top_class} ({result.confidence:.2%})")
    """

    MODEL_URL = "https://tfhub.dev/google/yamnet/1"
    SAMPLE_RATE = 16000  # YAMNet expects 16kHz audio

    def __init__(self, model_url: Optional[str] = None):
        """
        Initialize the YAMNet classifier.

        Args:
            model_url: Optional custom model URL. Defaults to official YAMNet.
        """
        self.model_url = model_url or self.MODEL_URL
        self._model = None
        self._class_names = None

    @property
    def model(self):
        """Lazy load the YAMNet model."""
        if self._model is None:
            print("Loading YAMNet model...")
            self._model = hub.load(self.model_url)
            print("YAMNet model loaded.")
        return self._model

    @property
    def class_names(self) -> List[str]:
        """Get the 521 class names from YAMNet."""
        if self._class_names is None:
            class_map_path = self.model.class_map_path().numpy().decode("utf-8")
            self._class_names = self._load_class_names(class_map_path)
        return self._class_names

    def _load_class_names(self, class_map_path: str) -> List[str]:
        """Load class names from YAMNet's class map CSV."""
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            return [row["display_name"] for row in reader]

    def classify(
        self,
        audio_path: str,
        top_k: int = 10,
        min_score: float = 0.1,
    ) -> ClassificationResult:
        """
        Classify audio file into sound categories.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            top_k: Number of top classes to return
            min_score: Minimum score threshold for results

        Returns:
            ClassificationResult with scores, embeddings, and top classes
        """
        # Load and preprocess audio
        waveform = self._load_audio(audio_path)

        # Run inference
        scores, embeddings, spectrogram = self.model(waveform)

        # Convert to numpy
        scores_np = scores.numpy()
        embeddings_np = embeddings.numpy()
        spectrogram_np = spectrogram.numpy()

        # Get mean scores across all frames
        mean_scores = np.mean(scores_np, axis=0)

        # Get top classes
        top_indices = np.argsort(mean_scores)[::-1][:top_k]
        top_classes = [
            (self.class_names[idx], float(mean_scores[idx]))
            for idx in top_indices
            if mean_scores[idx] >= min_score
        ]

        return ClassificationResult(
            scores=scores_np,
            embeddings=embeddings_np,
            spectrogram=spectrogram_np,
            top_classes=top_classes,
            all_classes=self.class_names,
        )

    def classify_frames(
        self,
        audio_path: str,
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """
        Classify audio and return per-frame results.

        Useful for detecting when specific sounds occur in longer audio.

        Args:
            audio_path: Path to audio file
            top_k: Number of top classes per frame

        Returns:
            List of top classes for each ~0.96s frame
        """
        waveform = self._load_audio(audio_path)
        scores, _, _ = self.model(waveform)
        scores_np = scores.numpy()

        frame_results = []
        for frame_scores in scores_np:
            top_indices = np.argsort(frame_scores)[::-1][:top_k]
            frame_classes = [
                (self.class_names[idx], float(frame_scores[idx]))
                for idx in top_indices
            ]
            frame_results.append(frame_classes)

        return frame_results

    def _load_audio(self, audio_path: str) -> tf.Tensor:
        """Load and preprocess audio for YAMNet."""
        # Load audio at YAMNet's expected sample rate
        waveform, sr = load_audio(audio_path, sr=self.SAMPLE_RATE, mono=True)

        # Ensure float32
        waveform = waveform.astype(np.float32)

        # Normalize to [-1, 1] if needed
        if np.abs(waveform).max() > 1.0:
            waveform = waveform / np.abs(waveform).max()

        return tf.constant(waveform)

    def get_category_groups(self) -> dict:
        """
        Get grouped sound categories for easier navigation.

        Returns:
            Dictionary mapping group names to lists of class names
        """
        # Common groupings based on AudioSet ontology
        groups = {
            "music": [],
            "speech": [],
            "animals": [],
            "nature": [],
            "vehicles": [],
            "tools": [],
            "domestic": [],
            "other": [],
        }

        music_keywords = [
            "music",
            "drum",
            "guitar",
            "piano",
            "bass",
            "synth",
            "beat",
            "instrument",
        ]
        speech_keywords = ["speech", "voice", "talk", "sing", "vocal"]
        animal_keywords = ["animal", "dog", "cat", "bird", "insect"]
        nature_keywords = ["wind", "rain", "thunder", "water", "fire"]
        vehicle_keywords = ["car", "engine", "motor", "vehicle", "train", "plane"]
        tool_keywords = ["tool", "hammer", "saw", "drill", "machine"]
        domestic_keywords = ["door", "knock", "bell", "alarm", "phone"]

        for name in self.class_names:
            name_lower = name.lower()
            if any(kw in name_lower for kw in music_keywords):
                groups["music"].append(name)
            elif any(kw in name_lower for kw in speech_keywords):
                groups["speech"].append(name)
            elif any(kw in name_lower for kw in animal_keywords):
                groups["animals"].append(name)
            elif any(kw in name_lower for kw in nature_keywords):
                groups["nature"].append(name)
            elif any(kw in name_lower for kw in vehicle_keywords):
                groups["vehicles"].append(name)
            elif any(kw in name_lower for kw in tool_keywords):
                groups["tools"].append(name)
            elif any(kw in name_lower for kw in domestic_keywords):
                groups["domestic"].append(name)
            else:
                groups["other"].append(name)

        return groups

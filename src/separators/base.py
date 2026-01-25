"""
Base class for audio source separators.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class SeparationResult:
    """Result of audio source separation."""

    stems: Dict[str, np.ndarray]  # Stem name -> audio waveform
    sample_rate: int
    duration: float  # Processing duration in seconds
    separator: str  # Name of separator used
    model: str  # Model name/variant
    output_paths: Dict[str, Path] = field(default_factory=dict)

    @property
    def stem_names(self) -> List[str]:
        """Get list of stem names."""
        return list(self.stems.keys())

    def get_stem(self, name: str) -> Optional[np.ndarray]:
        """Get a specific stem by name."""
        return self.stems.get(name)

    def save_stems(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, Path]:
        """
        Save all stems to output directory.

        Args:
            output_dir: Directory to save stems
            prefix: Optional prefix for filenames

        Returns:
            Dictionary mapping stem names to output paths
        """
        from src.utils import save_audio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for name, audio in self.stems.items():
            filename = f"{prefix}_{name}.wav" if prefix else f"{name}.wav"
            path = output_dir / filename
            save_audio(path, audio, self.sample_rate)
            paths[name] = path

        self.output_paths = paths
        return paths


class BaseSeparator(ABC):
    """
    Abstract base class for audio source separators.

    All separator implementations should inherit from this class
    and implement the `separate` method.
    """

    # Default stems that can be extracted
    AVAILABLE_STEMS: List[str] = []

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the separator.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.device = device or self._detect_device()
        self._model = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the separator."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model being used."""
        pass

    @abstractmethod
    def separate(
        self,
        audio_path: Union[str, Path],
        stems: Optional[List[str]] = None,
    ) -> SeparationResult:
        """
        Separate audio into stems.

        Args:
            audio_path: Path to input audio file
            stems: Optional list of stems to extract (None = all)

        Returns:
            SeparationResult with separated stems
        """
        pass

    def separate_and_save(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        stems: Optional[List[str]] = None,
    ) -> SeparationResult:
        """
        Separate audio and save stems to disk.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save stems
            stems: Optional list of stems to extract

        Returns:
            SeparationResult with output paths populated
        """
        result = self.separate(audio_path, stems)

        # Use input filename as prefix
        prefix = Path(audio_path).stem
        result.save_stems(output_dir, prefix)

        return result

    def benchmark(
        self,
        audio_path: Union[str, Path],
        runs: int = 3,
    ) -> Dict:
        """
        Benchmark separation performance.

        Args:
            audio_path: Path to input audio file
            runs: Number of benchmark runs

        Returns:
            Dictionary with benchmark results
        """
        from src.utils import get_audio_info

        info = get_audio_info(audio_path)
        times = []

        for _ in range(runs):
            start = time.time()
            self.separate(audio_path)
            elapsed = time.time() - start
            times.append(elapsed)

        return {
            "separator": self.name,
            "model": self.model_name,
            "device": self.device,
            "audio_duration": info["duration"],
            "runs": runs,
            "times": times,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "realtime_factor": np.mean(times) / info["duration"],
        }

    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"

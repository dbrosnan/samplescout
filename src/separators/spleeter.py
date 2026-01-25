"""
Spleeter audio source separator.

Spleeter is Deezer's fast source separation library.
It's faster than Demucs but with slightly lower quality.
Great for real-time applications and edge deployment.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from src.separators.base import BaseSeparator, SeparationResult
from src.utils import load_audio


class SpleeterSeparator(BaseSeparator):
    """
    Audio source separator using Spleeter.

    Spleeter offers fast separation with pre-trained models for
    2, 4, or 5 stems. Lower quality than Demucs but much faster.

    Models:
        - 2stems: vocals, accompaniment
        - 4stems: vocals, drums, bass, other
        - 5stems: vocals, drums, bass, piano, other

    Example:
        separator = SpleeterSeparator(stems=4)
        result = separator.separate("song.mp3")
        vocals = result.get_stem("vocals")
    """

    STEM_CONFIGS = {
        2: ["vocals", "accompaniment"],
        4: ["vocals", "drums", "bass", "other"],
        5: ["vocals", "drums", "bass", "piano", "other"],
    }

    def __init__(
        self,
        stems: int = 4,
        device: Optional[str] = None,
        bitrate: str = "320k",
    ):
        """
        Initialize Spleeter separator.

        Args:
            stems: Number of stems (2, 4, or 5)
            device: Device ('cpu' or 'gpu')
            bitrate: Output bitrate for audio
        """
        super().__init__(device)

        if stems not in self.STEM_CONFIGS:
            raise ValueError(f"stems must be 2, 4, or 5. Got: {stems}")

        self._stems = stems
        self.bitrate = bitrate
        self._separator = None

    @property
    def name(self) -> str:
        return "spleeter"

    @property
    def model_name(self) -> str:
        return f"spleeter:{self._stems}stems"

    @property
    def AVAILABLE_STEMS(self) -> List[str]:
        return self.STEM_CONFIGS[self._stems]

    @property
    def separator(self):
        """Lazy load the Spleeter separator."""
        if self._separator is None:
            self._separator = self._load_separator()
        return self._separator

    def _load_separator(self):
        """Load the Spleeter separator."""
        try:
            from spleeter.separator import Separator

            # Suppress TensorFlow warnings
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

            print(f"Loading Spleeter model: {self._stems}stems...")
            separator = Separator(f"spleeter:{self._stems}stems")
            print("Spleeter model loaded.")

            return separator

        except ImportError as e:
            raise ImportError(
                "Spleeter not installed. Install with: pip install spleeter"
            ) from e

    def separate(
        self,
        audio_path: Union[str, Path],
        stems: Optional[List[str]] = None,
    ) -> SeparationResult:
        """
        Separate audio into stems using Spleeter.

        Args:
            audio_path: Path to input audio file
            stems: Optional list of stems to extract (None = all)

        Returns:
            SeparationResult with separated stems
        """
        audio_path = Path(audio_path)
        start_time = time.time()

        # Get sample rate from input
        _, sample_rate = load_audio(audio_path, sr=None, mono=True, duration=0.1)

        # Spleeter outputs to a directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run separation
            self.separator.separate_to_file(
                str(audio_path),
                temp_dir,
                codec="wav",
            )

            # Find output directory (named after input file)
            stem_dir = Path(temp_dir) / audio_path.stem

            # Load separated stems
            available_stems = self.STEM_CONFIGS[self._stems]

            # Filter requested stems
            if stems is not None:
                stems = [s.lower() for s in stems]
            else:
                stems = available_stems

            result_stems: Dict[str, np.ndarray] = {}
            for stem_name in stems:
                if stem_name in available_stems:
                    stem_path = stem_dir / f"{stem_name}.wav"
                    if stem_path.exists():
                        audio, sr = load_audio(stem_path, sr=None, mono=True)
                        result_stems[stem_name] = audio
                        sample_rate = sr

        elapsed = time.time() - start_time

        return SeparationResult(
            stems=result_stems,
            sample_rate=sample_rate,
            duration=elapsed,
            separator=self.name,
            model=self.model_name,
        )

    def get_available_stems(self) -> List[str]:
        """Get list of available stems for the current model."""
        return self.STEM_CONFIGS[self._stems]

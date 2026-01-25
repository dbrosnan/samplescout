"""
Demucs audio source separator.

Demucs is a state-of-the-art music source separation model from Facebook Research.
It provides high-quality separation of drums, bass, vocals, and other instruments.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio

from src.separators.base import BaseSeparator, SeparationResult


class DemucsSeparator(BaseSeparator):
    """
    Audio source separator using Demucs.

    Demucs v4 (Hybrid Transformer) provides the best quality separation
    but requires more memory. Use 'htdemucs' for best quality or
    'htdemucs_ft' for fine-tuned model.

    Available stems: drums, bass, vocals, other

    Example:
        separator = DemucsSeparator(model="htdemucs")
        result = separator.separate("song.mp3")
        drums = result.get_stem("drums")
    """

    AVAILABLE_STEMS = ["drums", "bass", "vocals", "other"]

    # Available Demucs models
    MODELS = {
        "htdemucs": "Hybrid Transformer Demucs (best quality)",
        "htdemucs_ft": "Fine-tuned Hybrid Transformer",
        "htdemucs_6s": "6-stem model (adds guitar, piano)",
        "hdemucs_mmi": "Hybrid Demucs trained on MMI",
        "mdx": "MDX-Net architecture",
        "mdx_extra": "MDX-Net with extra training",
    }

    def __init__(
        self,
        model: str = "htdemucs",
        device: Optional[str] = None,
        shifts: int = 1,
        overlap: float = 0.25,
    ):
        """
        Initialize Demucs separator.

        Args:
            model: Model name (see MODELS for options)
            device: Device ('cpu', 'cuda', 'mps', or None for auto)
            shifts: Number of random shifts for augmentation (1 = no shift)
            overlap: Overlap between chunks (0-1)
        """
        super().__init__(device)
        self._model_name = model
        self.shifts = shifts
        self.overlap = overlap
        self._demucs_model = None

    @property
    def name(self) -> str:
        return "demucs"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model(self):
        """Lazy load the Demucs model."""
        if self._demucs_model is None:
            self._demucs_model = self._load_model()
        return self._demucs_model

    def _load_model(self):
        """Load the Demucs model."""
        try:
            from demucs import pretrained
            from demucs.apply import apply_model

            print(f"Loading Demucs model: {self._model_name}...")
            model = pretrained.get_model(self._model_name)
            model.to(self.device)
            model.eval()
            print(f"Demucs model loaded on {self.device}")

            # Store apply_model function for later use
            self._apply_model = apply_model

            return model

        except ImportError as e:
            raise ImportError(
                "Demucs not installed. Install with: pip install demucs"
            ) from e

    def separate(
        self,
        audio_path: Union[str, Path],
        stems: Optional[List[str]] = None,
    ) -> SeparationResult:
        """
        Separate audio into stems using Demucs.

        Args:
            audio_path: Path to input audio file
            stems: Optional list of stems to extract (None = all)

        Returns:
            SeparationResult with separated stems
        """
        audio_path = Path(audio_path)
        start_time = time.time()

        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Demucs expects stereo audio
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(self.device)

        # Get model and apply
        model = self.model

        # Apply separation
        with torch.no_grad():
            sources = self._apply_model(
                model,
                waveform,
                shifts=self.shifts,
                overlap=self.overlap,
            )

        # sources shape: (batch, num_stems, channels, samples)
        sources = sources.squeeze(0).cpu().numpy()

        # Get stem names from model
        stem_names = model.sources

        # Filter requested stems
        if stems is not None:
            stems = [s.lower() for s in stems]
        else:
            stems = stem_names

        # Build result dictionary
        result_stems: Dict[str, np.ndarray] = {}
        for i, name in enumerate(stem_names):
            if name in stems:
                # Average stereo to mono for consistency
                audio = sources[i]
                if audio.shape[0] == 2:
                    audio = np.mean(audio, axis=0)
                result_stems[name] = audio

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
        model = self.model
        return list(model.sources)

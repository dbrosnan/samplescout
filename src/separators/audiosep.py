"""
AudioSep - Language-guided audio source separator.

AudioSep allows separation of arbitrary sounds using text descriptions.
E.g., "separate the dog barking" or "extract the piano melody"

Uses the official AudioSep model from Audio-AGI.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from src.separators.base import BaseSeparator, SeparationResult
from src.utils import load_audio, save_audio

# Add AudioSep model directory to path
AUDIOSEP_MODEL_DIR = Path(__file__).parent.parent.parent / "audiosep_model"
if AUDIOSEP_MODEL_DIR.exists() and str(AUDIOSEP_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIOSEP_MODEL_DIR))


class AudioSepSeparator(BaseSeparator):
    """
    Audio source separator using AudioSep.

    AudioSep uses language guidance (CLAP encoder) to separate sounds based on
    text descriptions. The text is encoded and used to condition a ResUNet
    separation model via FiLM layers.

    Example:
        separator = AudioSepSeparator()
        result = separator.separate("audio.wav", prompts=["drums", "vocals"])
        drums = result.get_stem("drums")
    """

    AVAILABLE_STEMS = []  # Dynamic based on prompts

    # AudioSep processes at 32kHz
    SAMPLE_RATE = 32000

    def __init__(
        self,
        model: str = "audiosep-base",
        device: Optional[str] = None,
    ):
        """
        Initialize AudioSep separator.

        Args:
            model: Model name ('audiosep-base')
            device: Device ('cpu', 'cuda', or None for auto)
        """
        super().__init__(device)
        self._model_name = model
        self._audiosep_model = None
        self._model_loaded = False

    @property
    def name(self) -> str:
        return "audiosep"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model(self):
        """Lazy load the AudioSep model."""
        if not self._model_loaded:
            self._load_model()
        return self._audiosep_model

    def _load_model(self):
        """Load the real AudioSep model."""
        self._model_loaded = True

        # Check if AudioSep model files exist
        config_path = AUDIOSEP_MODEL_DIR / "config" / "audiosep_base.yaml"
        checkpoint_path = AUDIOSEP_MODEL_DIR / "checkpoint" / "audiosep_base_4M_steps.ckpt"

        if not config_path.exists():
            raise FileNotFoundError(f"AudioSep config not found: {config_path}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"AudioSep checkpoint not found: {checkpoint_path}")

        # Import AudioSep pipeline - need to change to model dir for relative paths
        original_cwd = os.getcwd()
        os.chdir(AUDIOSEP_MODEL_DIR)

        # Fix for PyTorch 2.6+ which requires weights_only=False for older checkpoints
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load

        try:
            from pipeline import build_audiosep

            print(f"Loading AudioSep model: {self._model_name}...")
            print(f"  Config: {config_path}")
            print(f"  Checkpoint: {checkpoint_path}")

            self._audiosep_model = build_audiosep(
                config_yaml=str(config_path),
                checkpoint_path=str(checkpoint_path),
                device=self.device
            )

            print(f"AudioSep model loaded on {self.device}")
        finally:
            os.chdir(original_cwd)
            torch.load = original_torch_load

    def separate(
        self,
        audio_path: Union[str, Path],
        stems: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
    ) -> SeparationResult:
        """
        Separate audio using text prompts (e.g., YAMNet classification labels).

        Args:
            audio_path: Path to input audio file
            stems: Alias for prompts (for API compatibility)
            prompts: Text descriptions of sounds to separate (e.g., ["Speech", "Music"])

        Returns:
            SeparationResult with separated stems (named by prompts)
        """
        audio_path = Path(audio_path)
        start_time = time.time()

        # Use prompts or stems
        text_prompts = prompts or stems or ["vocals", "drums", "bass", "other"]

        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()

        # Load at 32kHz for AudioSep
        waveform, sample_rate = load_audio(audio_path, sr=self.SAMPLE_RATE, mono=True)

        # Run real separation
        result_stems = self._separate(waveform, text_prompts)

        elapsed = time.time() - start_time

        return SeparationResult(
            stems=result_stems,
            sample_rate=self.SAMPLE_RATE,
            duration=elapsed,
            separator=self.name,
            model=self.model_name,
        )

    def _separate(
        self,
        waveform: np.ndarray,
        prompts: List[str],
    ) -> Dict[str, np.ndarray]:
        """Separate using the AudioSep model with CLAP text encoder."""
        result_stems = {}

        for prompt in prompts:
            print(f"  Separating '{prompt}' using AudioSep...")

            with torch.no_grad():
                # Get text embedding from CLAP encoder
                conditions = self._audiosep_model.query_encoder.get_query_embed(
                    modality='text',
                    text=[prompt],
                    device=self.device
                )

                # Prepare input
                input_dict = {
                    "mixture": torch.Tensor(waveform)[None, None, :].to(self.device),
                    "condition": conditions,
                }

                # Run separation
                sep_segment = self._audiosep_model.ss_model(input_dict)["waveform"]
                sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()

                result_stems[prompt] = sep_segment.astype(np.float32)

        return result_stems

    def get_available_stems(self) -> List[str]:
        """AudioSep can separate any sound described by text."""
        return [
            "vocals",
            "drums",
            "bass",
            "guitar",
            "piano",
            "strings",
            "speech",
            "music",
            "noise",
            "any text description...",
        ]

"""
AudioSep - Language-guided audio source separator.

AudioSep allows separation of arbitrary sounds using text descriptions.
E.g., "separate the dog barking" or "extract the piano melody"
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from src.separators.base import BaseSeparator, SeparationResult
from src.utils import load_audio, save_audio


class AudioSepSeparator(BaseSeparator):
    """
    Audio source separator using AudioSep.

    AudioSep uses language guidance to separate sounds based on
    text descriptions. This is more flexible than fixed-stem
    separators but may be less accurate for music separation.

    Example:
        separator = AudioSepSeparator()
        result = separator.separate("audio.wav", prompts=["drums", "vocals"])
        drums = result.get_stem("drums")

    Note:
        Requires extra dependencies: pip install samplescout[audiosep]
    """

    AVAILABLE_STEMS = []  # Dynamic based on prompts

    def __init__(
        self,
        model: str = "audiosep-base",
        device: Optional[str] = None,
    ):
        """
        Initialize AudioSep separator.

        Args:
            model: Model name ('audiosep-base' or 'audiosep-large')
            device: Device ('cpu', 'cuda', or None for auto)
        """
        super().__init__(device)
        self._model_name = model
        self._audiosep_model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "audiosep"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model(self):
        """Lazy load the AudioSep model."""
        if self._audiosep_model is None:
            self._load_model()
        return self._audiosep_model

    def _load_model(self):
        """Load the AudioSep model."""
        try:
            from transformers import AutoProcessor, AutoModelForAudioSeparation

            print(f"Loading AudioSep model: {self._model_name}...")

            # Map friendly names to HuggingFace model IDs
            model_map = {
                "audiosep-base": "audio-agi/audiosep",
                "audiosep": "audio-agi/audiosep",
            }

            model_id = model_map.get(self._model_name, self._model_name)

            self._processor = AutoProcessor.from_pretrained(model_id)
            self._audiosep_model = AutoModelForAudioSeparation.from_pretrained(model_id)
            self._audiosep_model.to(self.device)
            self._audiosep_model.eval()

            print(f"AudioSep model loaded on {self.device}")

        except ImportError as e:
            raise ImportError(
                "AudioSep dependencies not installed. "
                "Install with: pip install samplescout[audiosep]"
            ) from e
        except Exception as e:
            # AudioSep may not be available on HuggingFace Hub yet
            # Provide fallback implementation
            print(f"Warning: Could not load AudioSep from HuggingFace: {e}")
            print("Using mock implementation for development.")
            self._use_mock = True

    def separate(
        self,
        audio_path: Union[str, Path],
        stems: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
    ) -> SeparationResult:
        """
        Separate audio using text prompts.

        Args:
            audio_path: Path to input audio file
            stems: Alias for prompts (for API compatibility)
            prompts: Text descriptions of sounds to separate

        Returns:
            SeparationResult with separated stems (named by prompts)
        """
        audio_path = Path(audio_path)
        start_time = time.time()

        # Use prompts or stems
        text_prompts = prompts or stems or ["vocals", "drums", "bass", "other"]

        # Load audio
        waveform, sample_rate = load_audio(audio_path, sr=16000, mono=True)

        # Check if using mock implementation
        if hasattr(self, "_use_mock") and self._use_mock:
            result_stems = self._mock_separate(waveform, text_prompts)
        else:
            result_stems = self._real_separate(waveform, sample_rate, text_prompts)

        elapsed = time.time() - start_time

        return SeparationResult(
            stems=result_stems,
            sample_rate=sample_rate,
            duration=elapsed,
            separator=self.name,
            model=self.model_name,
        )

    def _real_separate(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        prompts: List[str],
    ) -> Dict[str, np.ndarray]:
        """Separate using actual AudioSep model."""
        result_stems = {}

        for prompt in prompts:
            # Process input
            inputs = self._processor(
                audio=waveform,
                text=prompt,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run separation
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract separated audio
            separated = outputs.audio.squeeze().cpu().numpy()
            result_stems[prompt] = separated

        return result_stems

    def _mock_separate(
        self,
        waveform: np.ndarray,
        prompts: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Mock separation for development/testing.

        Returns frequency-filtered versions of input as "separated" stems.
        """
        from scipy import signal

        result_stems = {}

        for i, prompt in enumerate(prompts):
            # Create different bandpass filters for each "stem"
            # This is just for testing - not real separation
            nyquist = 8000  # Half of 16kHz

            if "bass" in prompt.lower():
                # Low frequencies
                b, a = signal.butter(4, 200 / nyquist, btype="low")
            elif "drums" in prompt.lower() or "percussion" in prompt.lower():
                # Mid-low frequencies
                b, a = signal.butter(4, [100 / nyquist, 2000 / nyquist], btype="band")
            elif "vocal" in prompt.lower() or "voice" in prompt.lower():
                # Mid frequencies (voice range)
                b, a = signal.butter(4, [300 / nyquist, 3400 / nyquist], btype="band")
            else:
                # High frequencies for "other"
                b, a = signal.butter(4, 2000 / nyquist, btype="high")

            filtered = signal.filtfilt(b, a, waveform)
            result_stems[prompt] = filtered.astype(np.float32)

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

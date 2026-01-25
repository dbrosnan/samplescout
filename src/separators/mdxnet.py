"""
MDX-Net - High-quality music source separation.

Uses the python-audio-separator package which wraps MDX-Net models
from Ultimate Vocal Remover (UVR).
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from src.separators.base import BaseSeparator, SeparationResult
from src.utils import load_audio, save_audio


class MDXNetSeparator(BaseSeparator):
    """
    Audio source separator using MDX-Net models.

    MDX-Net provides high-quality stem separation using models trained
    for the Music Demixing Challenge. Models are automatically downloaded
    on first use.

    Available models:
        - UVR-MDX-NET-Inst_HQ_3: High quality instrumental
        - UVR_MDXNET_KARA_2: Karaoke (vocal removal)
        - Kim_Vocal_2: Excellent vocal isolation
        - UVR-MDX-NET-Voc_FT: Fine-tuned vocals

    Example:
        separator = MDXNetSeparator()
        result = separator.separate("audio.wav", stems=["vocals", "instrumental"])
    """

    AVAILABLE_STEMS = ["vocals", "instrumental"]

    # MDX-Net models and their output types
    MODELS = {
        "UVR-MDX-NET-Inst_HQ_3": {
            "description": "High quality instrumental separation",
            "outputs": ["Vocals", "Instrumental"],
        },
        "UVR_MDXNET_KARA_2": {
            "description": "Karaoke (vocal removal)",
            "outputs": ["Vocals", "Instrumental"],
        },
        "Kim_Vocal_2": {
            "description": "Excellent vocal isolation",
            "outputs": ["Vocals", "Instrumental"],
        },
        "UVR-MDX-NET-Voc_FT": {
            "description": "Fine-tuned vocal separation",
            "outputs": ["Vocals", "Instrumental"],
        },
    }

    DEFAULT_MODEL = "UVR-MDX-NET-Inst_HQ_3"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        """
        Initialize MDX-Net separator.

        Args:
            model: Model name (see MODELS dict for options)
            device: Device ('cpu', 'cuda', or None for auto)
        """
        super().__init__(device)
        self._model_name = model
        self._separator = None
        self._model_loaded = False

    @property
    def name(self) -> str:
        return "mdxnet"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self):
        """Load the MDX-Net model."""
        if self._model_loaded:
            return

        from audio_separator.separator import Separator

        print(f"Loading MDX-Net model: {self._model_name}...")

        # Initialize separator with model
        self._separator = Separator(
            model_file_dir="/tmp/audio-separator-models",
            output_format="WAV",
        )

        # Load the specific model
        self._separator.load_model(model_filename=f"{self._model_name}.onnx")

        print(f"MDX-Net model loaded: {self._model_name}")
        self._model_loaded = True

    def separate(
        self,
        audio_path: Union[str, Path],
        stems: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
    ) -> SeparationResult:
        """
        Separate audio into vocals and instrumental stems.

        Args:
            audio_path: Path to input audio file
            stems: Not used for MDX-Net (always outputs vocals + instrumental)
            prompts: Not used for MDX-Net

        Returns:
            SeparationResult with vocals and instrumental stems
        """
        audio_path = Path(audio_path)
        start_time = time.time()

        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()

        # Create temp output directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set output directory
            self._separator.output_dir = temp_dir

            # Run separation
            print(f"  Separating with MDX-Net ({self._model_name})...")
            output_files = self._separator.separate(str(audio_path))

            # Load the separated stems
            result_stems = {}
            for output_file in output_files:
                output_path = Path(output_file)
                stem_name = self._get_stem_name(output_path.stem)

                # Load the audio
                audio, sr = load_audio(output_path, sr=None, mono=False)
                result_stems[stem_name] = audio

        elapsed = time.time() - start_time

        # Get sample rate from first stem
        sample_rate = 44100  # MDX-Net default

        return SeparationResult(
            stems=result_stems,
            sample_rate=sample_rate,
            duration=elapsed,
            separator=self.name,
            model=self.model_name,
        )

    def _get_stem_name(self, filename: str) -> str:
        """Extract stem name from output filename."""
        filename_lower = filename.lower()
        if "vocal" in filename_lower:
            return "vocals"
        elif "instrument" in filename_lower or "no_vocal" in filename_lower:
            return "instrumental"
        elif "drum" in filename_lower:
            return "drums"
        elif "bass" in filename_lower:
            return "bass"
        else:
            return "other"

    def get_available_stems(self) -> List[str]:
        """Return list of stems this separator can produce."""
        return self.AVAILABLE_STEMS

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List available MDX-Net models."""
        return {name: info["description"] for name, info in cls.MODELS.items()}

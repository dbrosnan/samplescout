"""
SampleScout - AI-powered audio sampling system.

Classify, separate, and process audio for use in synthesizers.
"""

from src.classifier import YAMNetClassifier
from src.pipeline import SampleScout
from src.utils import load_audio, save_audio

__version__ = "0.1.0"
__all__ = ["SampleScout", "YAMNetClassifier", "load_audio", "save_audio"]

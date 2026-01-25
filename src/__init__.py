"""
SampleScout - AI-powered audio sampling system.

Classify, separate, and process audio for use in synthesizers.
"""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies upfront."""
    if name == "YAMNetClassifier":
        from src.classifier import YAMNetClassifier
        return YAMNetClassifier
    elif name == "SampleScout":
        from src.pipeline import SampleScout
        return SampleScout
    elif name == "load_audio":
        from src.utils import load_audio
        return load_audio
    elif name == "save_audio":
        from src.utils import save_audio
        return save_audio
    raise AttributeError(f"module 'src' has no attribute '{name}'")

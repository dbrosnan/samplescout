"""
Audio source separators for SampleScout.

Provides wrappers for various source separation tools:
- Demucs: High-quality music separation
- Spleeter: Fast separation (good for edge deployment)
- AudioSep: Language-guided separation
"""

from src.separators.base import BaseSeparator, SeparationResult
from src.separators.demucs import DemucsSeparator
from src.separators.spleeter import SpleeterSeparator

# AudioSep is optional (requires extra dependencies)
try:
    from src.separators.audiosep import AudioSepSeparator

    AUDIOSEP_AVAILABLE = True
except ImportError:
    AUDIOSEP_AVAILABLE = False
    AudioSepSeparator = None  # type: ignore


def get_separator(name: str, **kwargs) -> BaseSeparator:
    """
    Factory function to get a separator by name.

    Args:
        name: Separator name ('demucs', 'spleeter', 'audiosep')
        **kwargs: Additional arguments for the separator

    Returns:
        Separator instance
    """
    separators = {
        "demucs": DemucsSeparator,
        "spleeter": SpleeterSeparator,
    }

    if AUDIOSEP_AVAILABLE:
        separators["audiosep"] = AudioSepSeparator

    name = name.lower()
    if name not in separators:
        available = list(separators.keys())
        raise ValueError(f"Unknown separator: {name}. Available: {available}")

    return separators[name](**kwargs)


__all__ = [
    "BaseSeparator",
    "SeparationResult",
    "DemucsSeparator",
    "SpleeterSeparator",
    "AudioSepSeparator",
    "get_separator",
    "AUDIOSEP_AVAILABLE",
]

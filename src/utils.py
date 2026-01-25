"""
Audio utility functions for SampleScout.

Provides common audio loading, saving, and manipulation functions.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    path: Union[str, Path],
    sr: Optional[int] = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.

    Args:
        path: Path to audio file
        sr: Target sample rate (None to keep original)
        mono: Convert to mono if True
        offset: Start reading at this time (seconds)
        duration: Only load this duration (seconds)

    Returns:
        Tuple of (audio waveform, sample rate)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    y, orig_sr = librosa.load(
        str(path),
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration,
    )

    return y, sr or orig_sr


def save_audio(
    path: Union[str, Path],
    audio: np.ndarray,
    sr: int,
    normalize: bool = True,
) -> Path:
    """
    Save audio to file.

    Args:
        path: Output path (format determined by extension)
        audio: Audio waveform
        sr: Sample rate
        normalize: Normalize audio to prevent clipping

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        audio = normalize_audio(audio)

    sf.write(str(path), audio, sr)
    return path


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to target peak level.

    Args:
        audio: Audio waveform
        target_peak: Target peak amplitude (0-1)

    Returns:
        Normalized audio
    """
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 30,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio: Audio waveform
        sr: Sample rate
        top_db: Threshold below reference to consider silence
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation

    Returns:
        Trimmed audio
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return trimmed


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    semitones: float,
) -> np.ndarray:
    """
    Shift pitch by semitones.

    Args:
        audio: Audio waveform
        sr: Sample rate
        semitones: Number of semitones to shift (positive = up, negative = down)

    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def time_stretch(
    audio: np.ndarray,
    rate: float,
) -> np.ndarray:
    """
    Time-stretch audio without changing pitch.

    Args:
        audio: Audio waveform
        rate: Stretch factor (>1 = slower, <1 = faster)

    Returns:
        Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def split_into_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_duration: float = 5.0,
    overlap: float = 0.0,
) -> list:
    """
    Split audio into fixed-duration chunks.

    Args:
        audio: Audio waveform
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds

    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int((chunk_duration - overlap) * sr)

    chunks = []
    for start in range(0, len(audio), hop_samples):
        end = start + chunk_samples
        chunk = audio[start:end]

        # Pad last chunk if necessary
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        chunks.append(chunk)

    return chunks


def get_audio_info(path: Union[str, Path]) -> dict:
    """
    Get audio file information.

    Args:
        path: Path to audio file

    Returns:
        Dictionary with audio info (duration, sample_rate, channels, etc.)
    """
    path = Path(path)
    info = sf.info(str(path))

    return {
        "path": str(path),
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio waveform
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono.

    Args:
        audio: Audio waveform (can be mono or stereo)

    Returns:
        Mono audio
    """
    if audio.ndim == 1:
        return audio
    return librosa.to_mono(audio)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Convert mono audio to stereo.

    Args:
        audio: Audio waveform

    Returns:
        Stereo audio (2, samples)
    """
    if audio.ndim == 2:
        return audio
    return np.stack([audio, audio], axis=0)


def apply_fade(
    audio: np.ndarray,
    sr: int,
    fade_in: float = 0.01,
    fade_out: float = 0.01,
) -> np.ndarray:
    """
    Apply fade in/out to audio.

    Args:
        audio: Audio waveform
        sr: Sample rate
        fade_in: Fade in duration in seconds
        fade_out: Fade out duration in seconds

    Returns:
        Audio with fades applied
    """
    audio = audio.copy()

    # Fade in
    if fade_in > 0:
        fade_samples = int(fade_in * sr)
        fade_curve = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_curve

    # Fade out
    if fade_out > 0:
        fade_samples = int(fade_out * sr)
        fade_curve = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_curve

    return audio


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute RMS (root mean square) energy of audio.

    Args:
        audio: Audio waveform

    Returns:
        RMS value
    """
    return float(np.sqrt(np.mean(audio**2)))


def detect_bpm(audio: np.ndarray, sr: int) -> float:
    """
    Detect tempo (BPM) of audio.

    Args:
        audio: Audio waveform
        sr: Sample rate

    Returns:
        Estimated BPM
    """
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    return float(tempo)

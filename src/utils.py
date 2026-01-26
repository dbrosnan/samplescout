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

    # soundfile expects (samples, channels) for stereo, transpose if needed
    if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
        # Shape is (channels, samples), transpose to (samples, channels)
        audio = audio.T

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


def compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """
    Compute the spectral centroid (brightness) of audio.

    Args:
        audio: Audio waveform
        sr: Sample rate

    Returns:
        Mean spectral centroid in Hz
    """
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return float(np.mean(centroid))


def compute_audio_similarity(
    source: np.ndarray,
    separated: np.ndarray,
    sr: int,
) -> dict:
    """
    Compute similarity metrics between source and separated audio.

    Args:
        source: Source audio waveform
        separated: Separated audio waveform
        sr: Sample rate

    Returns:
        Dictionary with similarity metrics
    """
    # Ensure same length
    min_len = min(len(source), len(separated))
    source = source[:min_len]
    separated = separated[:min_len]

    # Energy metrics
    source_energy = np.sum(source ** 2)
    separated_energy = np.sum(separated ** 2)
    energy_ratio = separated_energy / source_energy if source_energy > 0 else 0

    # RMS comparison
    source_rms = compute_rms(source)
    separated_rms = compute_rms(separated)
    rms_ratio = separated_rms / source_rms if source_rms > 0 else 0

    # Correlation coefficient
    if np.std(source) > 0 and np.std(separated) > 0:
        correlation = float(np.corrcoef(source, separated)[0, 1])
    else:
        correlation = 0.0

    # Spectral similarity using mel spectrograms
    n_mels = 128
    source_mel = librosa.feature.melspectrogram(y=source, sr=sr, n_mels=n_mels)
    separated_mel = librosa.feature.melspectrogram(y=separated, sr=sr, n_mels=n_mels)

    # Flatten and compute cosine similarity
    source_flat = source_mel.flatten()
    separated_flat = separated_mel.flatten()

    # Cosine similarity
    dot_product = np.dot(source_flat, separated_flat)
    norm_source = np.linalg.norm(source_flat)
    norm_separated = np.linalg.norm(separated_flat)
    spectral_similarity = dot_product / (norm_source * norm_separated) if (norm_source > 0 and norm_separated > 0) else 0

    # Spectral centroid comparison (brightness)
    source_centroid = compute_spectral_centroid(source, sr)
    separated_centroid = compute_spectral_centroid(separated, sr)
    centroid_shift = separated_centroid - source_centroid

    # Isolation score: how different is the separated audio from source
    # Higher = more different = better isolation
    isolation_score = 1.0 - abs(correlation)

    return {
        "energy_ratio": float(energy_ratio),
        "energy_percent": float(energy_ratio * 100),
        "rms_ratio": float(rms_ratio),
        "correlation": float(correlation),
        "spectral_similarity": float(spectral_similarity),
        "source_centroid_hz": float(source_centroid),
        "separated_centroid_hz": float(separated_centroid),
        "centroid_shift_hz": float(centroid_shift),
        "isolation_score": float(isolation_score),
    }


def compute_separation_insights(
    source_path: Union[str, Path],
    separated_path: Union[str, Path],
) -> dict:
    """
    Compute comprehensive insights comparing source and separated audio.

    Args:
        source_path: Path to source audio file
        separated_path: Path to separated audio file

    Returns:
        Dictionary with separation insights and quality metrics
    """
    # Load both audio files at same sample rate
    source, sr = load_audio(source_path, sr=22050, mono=True)
    separated, _ = load_audio(separated_path, sr=22050, mono=True)

    # Get basic info
    source_info = get_audio_info(source_path)
    separated_info = get_audio_info(separated_path)

    # Compute similarity metrics
    similarity = compute_audio_similarity(source, separated, sr)

    # Compute additional metrics
    source_peak = float(np.max(np.abs(source)))
    separated_peak = float(np.max(np.abs(separated)))

    # Zero crossing rate (texture/noisiness)
    source_zcr = float(np.mean(librosa.feature.zero_crossing_rate(source)))
    separated_zcr = float(np.mean(librosa.feature.zero_crossing_rate(separated)))

    return {
        "source": {
            "duration": source_info["duration"],
            "rms": float(compute_rms(source)),
            "peak": source_peak,
            "zcr": source_zcr,
        },
        "separated": {
            "duration": separated_info["duration"],
            "rms": float(compute_rms(separated)),
            "peak": separated_peak,
            "zcr": separated_zcr,
        },
        "comparison": {
            **similarity,
            "peak_ratio": separated_peak / source_peak if source_peak > 0 else 0,
            "zcr_ratio": separated_zcr / source_zcr if source_zcr > 0 else 0,
        },
        "quality": {
            # Interpretation of metrics
            "extraction_strength": _interpret_extraction_strength(similarity["energy_ratio"]),
            "isolation_quality": _interpret_isolation(similarity["isolation_score"]),
            "spectral_match": _interpret_spectral_match(similarity["spectral_similarity"]),
        },
    }


def _interpret_extraction_strength(energy_ratio: float) -> str:
    """Interpret the energy ratio as extraction strength."""
    if energy_ratio < 0.05:
        return "minimal"
    elif energy_ratio < 0.15:
        return "low"
    elif energy_ratio < 0.35:
        return "moderate"
    elif energy_ratio < 0.60:
        return "strong"
    else:
        return "dominant"


def _interpret_isolation(isolation_score: float) -> str:
    """Interpret the isolation score."""
    if isolation_score < 0.3:
        return "poor"
    elif isolation_score < 0.5:
        return "fair"
    elif isolation_score < 0.7:
        return "good"
    elif isolation_score < 0.85:
        return "very good"
    else:
        return "excellent"


def _interpret_spectral_match(spectral_similarity: float) -> str:
    """Interpret spectral similarity."""
    if spectral_similarity < 0.3:
        return "very different"
    elif spectral_similarity < 0.5:
        return "different"
    elif spectral_similarity < 0.7:
        return "similar"
    elif spectral_similarity < 0.85:
        return "very similar"
    else:
        return "nearly identical"

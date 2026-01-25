#!/usr/bin/env python3
"""
SampleScout Demo Script.

Demonstrates the key features of SampleScout:
1. Audio classification with YAMNet
2. Source separation with Demucs/Spleeter
3. Unified pipeline for intelligent extraction
4. Similarity search with embeddings

Usage:
    python run_demo.py                    # Run with generated test audio
    python run_demo.py path/to/audio.wav  # Run with your own audio
"""

import sys
import tempfile
from pathlib import Path

import numpy as np


def generate_test_audio(path: Path, duration: float = 5.0, sample_rate: int = 16000):
    """Generate a test audio file with mixed content."""
    from scipy.io import wavfile

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a mix of musical elements

    # Bass (low frequency sine)
    bass = 0.3 * np.sin(2 * np.pi * 110 * t)

    # Chord (A minor: A, C, E)
    chord = (
        0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.15 * np.sin(2 * np.pi * 261.63 * t)
        + 0.15 * np.sin(2 * np.pi * 329.63 * t)
    )

    # Melody
    melody_freqs = [440, 494, 523, 494, 440, 392, 440, 0]  # Simple pattern
    melody = np.zeros_like(t)
    note_duration = duration / len(melody_freqs)
    for i, freq in enumerate(melody_freqs):
        start = int(i * note_duration * sample_rate)
        end = int((i + 1) * note_duration * sample_rate)
        if freq > 0:
            note_t = np.linspace(0, note_duration, end - start)
            envelope = np.exp(-note_t * 3)  # Decay
            melody[start:end] = 0.3 * np.sin(2 * np.pi * freq * note_t) * envelope

    # Percussion (noise bursts)
    percussion = np.zeros_like(t)
    beat_interval = int(0.5 * sample_rate)  # 120 BPM
    for i in range(0, len(t), beat_interval):
        end = min(i + int(0.05 * sample_rate), len(t))
        percussion[i:end] = 0.2 * np.random.randn(end - i) * np.exp(
            -np.linspace(0, 1, end - i) * 10
        )

    # Mix everything
    audio = bass + chord + melody + percussion

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    audio = audio.astype(np.float32)

    # Save
    wavfile.write(str(path), sample_rate, audio)
    print(f"Generated test audio: {path}")
    print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz")

    return path


def demo_classification(audio_path: Path):
    """Demonstrate audio classification."""
    print("\n" + "=" * 60)
    print("DEMO 1: Audio Classification with YAMNet")
    print("=" * 60)

    from src.classifier import YAMNetClassifier

    print("\nLoading YAMNet classifier...")
    classifier = YAMNetClassifier()

    print(f"Classifying: {audio_path}")
    result = classifier.classify(audio_path, top_k=10)

    print("\nTop 10 Classifications:")
    print("-" * 40)
    for i, (name, score) in enumerate(result.top_classes, 1):
        bar = "█" * int(score * 30)
        print(f"{i:2d}. {name:25s} {score:5.1%} {bar}")

    print(f"\nProcessing time: {result.processing_time:.2f}s")
    print(f"Embedding shape: {result.embedding.shape}")

    return result


def demo_separation(audio_path: Path, output_dir: Path):
    """Demonstrate source separation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Source Separation with Demucs")
    print("=" * 60)

    from src.separators import get_separator

    print("\nLoading Demucs separator...")
    separator = get_separator("demucs", device="cpu")

    print(f"Separating: {audio_path}")
    print(f"Output: {output_dir}")

    result = separator.separate(audio_path)

    print("\nSeparated Stems:")
    print("-" * 40)
    for stem_name, audio in result.stems.items():
        duration = len(audio) / result.sample_rate
        rms = np.sqrt(np.mean(audio**2))
        print(f"  {stem_name:10s}: {duration:.2f}s, RMS: {rms:.4f}")

    # Save stems
    paths = result.save_stems(output_dir, prefix="demo")
    print("\nSaved stems:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print(f"\nSeparation time: {result.duration:.2f}s")

    return result


def demo_pipeline(audio_path: Path, output_dir: Path):
    """Demonstrate the full pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 3: Full SampleScout Pipeline")
    print("=" * 60)

    from src.pipeline import SampleScout

    print("\nInitializing SampleScout pipeline...")
    scout = SampleScout(separator_name="demucs", device="cpu")

    # Analyze without separation
    print("\n--- Analysis Only ---")
    result = scout.analyze(audio_path, separate=False, top_k=5)

    print(f"Top category: {result.top_category} ({result.top_score:.1%})")
    print(f"Processing time: {result.processing_time:.2f}s")

    # Extract with auto mode
    print("\n--- Auto Extraction ---")
    result = scout.extract(
        audio_path, output_dir=output_dir / "auto_extract", auto=True
    )

    print(f"Auto-selected stems: {list(result.extracted_samples.keys())}")
    print(f"Processing time: {result.processing_time:.2f}s")

    print("\nOutput files:")
    for name, path in result.output_paths.items():
        print(f"  {name}: {path}")

    return result


def demo_similarity(audio_paths: list):
    """Demonstrate similarity search."""
    print("\n" + "=" * 60)
    print("DEMO 4: Audio Similarity Search")
    print("=" * 60)

    from src.pipeline import SampleScout

    if len(audio_paths) < 2:
        print("Need at least 2 audio files for similarity demo")
        return

    scout = SampleScout()
    query = audio_paths[0]
    candidates = audio_paths[1:]

    print(f"\nQuery: {query}")
    print(f"Candidates: {len(candidates)} files")

    results = scout.find_similar(query, candidates, top_k=min(5, len(candidates)))

    print("\nMost Similar:")
    print("-" * 40)
    for path, score in results:
        print(f"  {score:.1%}  {Path(path).name}")


def demo_benchmark(audio_path: Path):
    """Demonstrate benchmarking."""
    print("\n" + "=" * 60)
    print("DEMO 5: Performance Benchmarking")
    print("=" * 60)

    from src.benchmark import Benchmark, get_system_info

    # Show system info
    info = get_system_info()
    print("\nSystem Info:")
    print(f"  Platform: {info.get('platform', 'unknown')}")
    print(f"  CPU Count: {info.get('cpu_count', 'unknown')}")
    print(f"  GPU: {info.get('gpu', 'unknown')}")

    # Quick benchmark
    print("\nRunning quick benchmark (1 run each)...")
    bench = Benchmark(warmup_runs=0, benchmark_runs=1)

    # Just benchmark classifier for demo speed
    from src.classifier import YAMNetClassifier
    from src.utils import get_audio_info
    import time

    info = get_audio_info(audio_path)
    classifier = YAMNetClassifier()

    start = time.time()
    classifier.classify(audio_path)
    elapsed = time.time() - start

    print("\nResults:")
    print("-" * 40)
    print(f"  YAMNet Classifier:")
    print(f"    Time: {elapsed:.2f}s")
    print(f"    Realtime factor: {elapsed / info['duration']:.2f}x")


def main():
    """Run all demos."""
    print("=" * 60)
    print("        SAMPLESCOUT DEMO")
    print("    AI-Powered Audio Sampling")
    print("=" * 60)

    # Determine audio source
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if not audio_path.exists():
            print(f"Error: File not found: {audio_path}")
            sys.exit(1)
    else:
        # Generate test audio
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "demo_audio.wav"
            generate_test_audio(audio_path, duration=5.0)

            # Run demos with generated audio
            run_demos(audio_path, Path(temp_dir))
            return

    # Run demos with provided audio
    with tempfile.TemporaryDirectory() as temp_dir:
        run_demos(audio_path, Path(temp_dir))


def run_demos(audio_path: Path, output_dir: Path):
    """Run all demo functions."""
    try:
        # Demo 1: Classification
        demo_classification(audio_path)

    except ImportError as e:
        print(f"\nSkipping classification demo (missing dependency): {e}")

    try:
        # Demo 5: Benchmark (run before separation for speed)
        demo_benchmark(audio_path)

    except ImportError as e:
        print(f"\nSkipping benchmark demo (missing dependency): {e}")

    # Run separation demo
    try:
        demo_separation(audio_path, output_dir / "stems")
    except ImportError as e:
        print(f"\nSkipping separation demo (missing dependency): {e}")
    except Exception as e:
        print(f"\nSeparation demo error: {e}")

    try:
        demo_pipeline(audio_path, output_dir / "pipeline")
    except ImportError as e:
        print(f"\nSkipping pipeline demo (missing dependency): {e}")
    except Exception as e:
        print(f"\nPipeline demo error: {e}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

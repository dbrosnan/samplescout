"""
SampleScout CLI - Command-line interface for audio analysis and extraction.

Usage:
    samplescout analyze song.mp3
    samplescout extract song.mp3 --output ./stems
    samplescout classify song.mp3 --top-k 5
    samplescout benchmark audio.wav --separator demucs
"""

import sys
from pathlib import Path
from typing import List, Optional

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="samplescout")
def cli():
    """SampleScout - AI-powered audio sampling and analysis."""
    pass


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--top-k", "-k", default=10, help="Number of top classes to show")
@click.option("--show-embedding", is_flag=True, help="Show embedding statistics")
def classify(audio_path: str, top_k: int, show_embedding: bool):
    """Classify audio content using YAMNet."""
    from src.classifier import YAMNetClassifier

    click.echo(f"Classifying: {audio_path}")
    click.echo()

    classifier = YAMNetClassifier()
    result = classifier.classify(audio_path, top_k=top_k)

    click.echo("Top Classifications:")
    click.echo("-" * 40)

    for i, (name, score) in enumerate(result.top_classes, 1):
        bar = "█" * int(score * 20)
        click.echo(f"{i:2d}. {name:30s} {score:5.1%} {bar}")

    click.echo()
    click.echo(f"Processing time: {result.processing_time:.2f}s")

    if show_embedding:
        click.echo()
        click.echo("Embedding Statistics:")
        click.echo(f"  Shape: {result.embedding.shape}")
        click.echo(f"  Mean: {result.embedding.mean():.4f}")
        click.echo(f"  Std: {result.embedding.std():.4f}")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--separate", "-s", is_flag=True, help="Also perform source separation")
@click.option(
    "--separator",
    type=click.Choice(["demucs", "spleeter", "audiosep"]),
    default="demucs",
    help="Separator to use",
)
@click.option("--top-k", "-k", default=10, help="Number of classifications")
def analyze(audio_path: str, separate: bool, separator: str, top_k: int):
    """Analyze audio with classification and optional separation."""
    from src.pipeline import SampleScout
    from src.utils import get_audio_info

    click.echo(f"Analyzing: {audio_path}")
    click.echo()

    # Show audio info
    info = get_audio_info(audio_path)
    click.echo("Audio Info:")
    click.echo(f"  Duration: {info['duration']:.2f}s")
    click.echo(f"  Sample Rate: {info['sample_rate']} Hz")
    click.echo(f"  Channels: {info['channels']}")
    click.echo()

    # Run pipeline
    scout = SampleScout(separator_name=separator)
    result = scout.analyze(audio_path, separate=separate, top_k=top_k)

    # Show classification
    click.echo("Classification:")
    click.echo("-" * 40)
    for name, score in result.classification.top_classes[:5]:
        click.echo(f"  {name:30s} {score:5.1%}")

    # Show separation if performed
    if result.separation:
        click.echo()
        click.echo("Separated Stems:")
        for stem_name in result.separation.stem_names:
            audio = result.separation.stems[stem_name]
            click.echo(f"  {stem_name}: {len(audio)} samples")

    click.echo()
    click.echo(f"Total processing time: {result.processing_time:.2f}s")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output directory"
)
@click.option(
    "--separator",
    type=click.Choice(["demucs", "spleeter", "audiosep"]),
    default="demucs",
    help="Separator to use",
)
@click.option("--stems", "-t", multiple=True, help="Specific stems to extract")
@click.option("--auto", is_flag=True, help="Auto-select stems based on content")
def extract(
    audio_path: str,
    output: str,
    separator: str,
    stems: tuple,
    auto: bool,
):
    """Extract audio stems using source separation."""
    from src.pipeline import SampleScout

    click.echo(f"Extracting stems from: {audio_path}")
    click.echo(f"Using separator: {separator}")
    click.echo()

    scout = SampleScout(separator_name=separator)

    stems_list = list(stems) if stems else None

    result = scout.extract(
        audio_path,
        output_dir=output,
        auto=auto,
        stems=stems_list,
    )

    click.echo("Extracted Stems:")
    click.echo("-" * 40)

    for stem_name, path in result.output_paths.items():
        click.echo(f"  {stem_name}: {path}")

    click.echo()
    click.echo(f"Processing time: {result.processing_time:.2f}s")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option(
    "--separator",
    type=click.Choice(["demucs", "spleeter", "audiosep"]),
    default="demucs",
    help="Separator to benchmark",
)
@click.option("--runs", "-n", default=3, help="Number of benchmark runs")
@click.option("--classify", is_flag=True, help="Also benchmark classifier")
def benchmark(audio_path: str, separator: str, runs: int, classify: bool):
    """Benchmark separation and classification performance."""
    from src.separators import get_separator
    from src.utils import get_audio_info

    click.echo(f"Benchmarking: {audio_path}")
    click.echo(f"Runs: {runs}")
    click.echo()

    info = get_audio_info(audio_path)
    click.echo(f"Audio duration: {info['duration']:.2f}s")
    click.echo()

    # Benchmark separator
    click.echo(f"Separator: {separator}")
    click.echo("-" * 40)

    sep = get_separator(separator)
    results = sep.benchmark(audio_path, runs=runs)

    click.echo(f"  Device: {results['device']}")
    click.echo(f"  Mean time: {results['mean_time']:.2f}s")
    click.echo(f"  Std time: {results['std_time']:.3f}s")
    click.echo(f"  Realtime factor: {results['realtime_factor']:.2f}x")

    # Benchmark classifier if requested
    if classify:
        from src.classifier import YAMNetClassifier
        import time
        import numpy as np

        click.echo()
        click.echo("Classifier: YAMNet")
        click.echo("-" * 40)

        classifier = YAMNetClassifier()
        times = []

        for _ in range(runs):
            start = time.time()
            classifier.classify(audio_path)
            times.append(time.time() - start)

        mean_time = np.mean(times)
        std_time = np.std(times)
        rtf = mean_time / info["duration"]

        click.echo(f"  Mean time: {mean_time:.2f}s")
        click.echo(f"  Std time: {std_time:.3f}s")
        click.echo(f"  Realtime factor: {rtf:.2f}x")


@cli.command()
@click.argument("query_path", type=click.Path(exists=True))
@click.argument("candidate_paths", type=click.Path(exists=True), nargs=-1)
@click.option("--top-k", "-k", default=5, help="Number of results")
def similar(query_path: str, candidate_paths: tuple, top_k: int):
    """Find audio files similar to a query."""
    from src.pipeline import SampleScout

    if not candidate_paths:
        click.echo("Error: Need candidate files to compare against.", err=True)
        sys.exit(1)

    click.echo(f"Finding files similar to: {query_path}")
    click.echo()

    scout = SampleScout()
    results = scout.find_similar(query_path, list(candidate_paths), top_k=top_k)

    click.echo("Similar Files:")
    click.echo("-" * 40)

    for path, score in results:
        click.echo(f"  {score:.1%}  {Path(path).name}")


@cli.command()
def info():
    """Show system and package information."""
    import platform

    click.echo("SampleScout System Info")
    click.echo("=" * 40)
    click.echo()

    # Python info
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"Platform: {platform.system()} {platform.release()}")
    click.echo()

    # Package versions
    click.echo("Package Versions:")

    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("librosa", "librosa"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
        ("demucs", "demucs"),
        ("spleeter", "spleeter"),
    ]

    for display_name, import_name in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            click.echo(f"  {display_name}: {version}")
        except ImportError:
            click.echo(f"  {display_name}: not installed")

    # GPU info
    click.echo()
    click.echo("GPU Availability:")

    try:
        import torch

        if torch.cuda.is_available():
            click.echo(f"  CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            click.echo("  MPS: Available (Apple Silicon)")
        else:
            click.echo("  GPU: Not available (using CPU)")
    except ImportError:
        click.echo("  PyTorch not installed")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()

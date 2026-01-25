"""
SampleScout Benchmark Suite.

Comprehensive benchmarking for audio classification and separation tools.
Measures performance, quality, and resource usage.
"""

import gc
import json
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    model: str
    device: str
    audio_duration: float
    processing_time: float
    realtime_factor: float
    memory_peak_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model": self.model,
            "device": self.device,
            "audio_duration": self.audio_duration,
            "processing_time": self.processing_time,
            "realtime_factor": self.realtime_factor,
            "memory_peak_mb": self.memory_peak_mb,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Union[str, Path]):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 60,
            "BENCHMARK RESULTS",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Platform: {self.system_info.get('platform', 'unknown')}",
            "",
        ]

        # Group by name
        by_name: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_name.setdefault(r.name, []).append(r)

        for name, results in by_name.items():
            lines.append(f"\n{name}")
            lines.append("-" * 40)

            for r in results:
                lines.append(
                    f"  {r.model:20s} | "
                    f"{r.processing_time:6.2f}s | "
                    f"{r.realtime_factor:5.2f}x | "
                    f"{r.memory_peak_mb:6.0f}MB"
                )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmarking context."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    # GPU info
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu"] = "Apple Silicon MPS"
        else:
            info["gpu"] = "None (CPU)"
    except ImportError:
        info["gpu"] = "PyTorch not installed"

    # Memory info
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_memory_gb"] = mem.total / (1024**3)
        info["available_memory_gb"] = mem.available / (1024**3)
    except ImportError:
        pass

    return info


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**2)
    except ImportError:
        return 0.0


class Benchmark:
    """
    Benchmark runner for SampleScout components.

    Example:
        bench = Benchmark()
        suite = bench.run_all("test_audio.wav")
        print(suite.summary())
        suite.save("benchmark_results.json")
    """

    def __init__(self, warmup_runs: int = 1, benchmark_runs: int = 3):
        """
        Initialize benchmark runner.

        Args:
            warmup_runs: Number of warmup runs before timing
            benchmark_runs: Number of timed runs for averaging
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def run_all(
        self,
        audio_path: Union[str, Path],
        separators: Optional[List[str]] = None,
        include_classifier: bool = True,
    ) -> BenchmarkSuite:
        """
        Run all benchmarks on an audio file.

        Args:
            audio_path: Path to test audio file
            separators: List of separators to benchmark (default: all available)
            include_classifier: Whether to benchmark the classifier

        Returns:
            BenchmarkSuite with all results
        """
        from src.utils import get_audio_info

        audio_path = Path(audio_path)
        audio_info = get_audio_info(audio_path)

        suite = BenchmarkSuite(system_info=get_system_info())

        # Benchmark classifier
        if include_classifier:
            print("Benchmarking YAMNet classifier...")
            result = self.benchmark_classifier(audio_path, audio_info["duration"])
            suite.add_result(result)
            gc.collect()

        # Benchmark separators
        if separators is None:
            separators = ["demucs", "spleeter"]

        for sep_name in separators:
            print(f"Benchmarking {sep_name}...")
            try:
                result = self.benchmark_separator(
                    audio_path, sep_name, audio_info["duration"]
                )
                suite.add_result(result)
            except Exception as e:
                print(f"  Error: {e}")
            gc.collect()

        return suite

    def benchmark_classifier(
        self,
        audio_path: Union[str, Path],
        audio_duration: float,
    ) -> BenchmarkResult:
        """Benchmark YAMNet classifier."""
        from src.classifier import YAMNetClassifier

        classifier = YAMNetClassifier()

        # Warmup
        for _ in range(self.warmup_runs):
            classifier.classify(audio_path)

        # Clear memory baseline
        gc.collect()
        mem_before = get_memory_usage_mb()

        # Timed runs
        times = []
        for _ in range(self.benchmark_runs):
            start = time.time()
            classifier.classify(audio_path)
            times.append(time.time() - start)

        mem_after = get_memory_usage_mb()

        mean_time = np.mean(times)
        std_time = np.std(times)

        return BenchmarkResult(
            name="classifier",
            model="yamnet",
            device="cpu",  # TensorFlow handles this internally
            audio_duration=audio_duration,
            processing_time=mean_time,
            realtime_factor=mean_time / audio_duration,
            memory_peak_mb=max(0, mem_after - mem_before),
            metadata={
                "std_time": std_time,
                "all_times": times,
            },
        )

    def benchmark_separator(
        self,
        audio_path: Union[str, Path],
        separator_name: str,
        audio_duration: float,
    ) -> BenchmarkResult:
        """Benchmark an audio separator."""
        from src.separators import get_separator

        separator = get_separator(separator_name)

        # Warmup
        for _ in range(self.warmup_runs):
            separator.separate(audio_path)

        # Clear memory baseline
        gc.collect()
        mem_before = get_memory_usage_mb()

        # Timed runs
        times = []
        for _ in range(self.benchmark_runs):
            start = time.time()
            separator.separate(audio_path)
            times.append(time.time() - start)

        mem_after = get_memory_usage_mb()

        mean_time = np.mean(times)
        std_time = np.std(times)

        return BenchmarkResult(
            name="separator",
            model=separator.model_name,
            device=separator.device,
            audio_duration=audio_duration,
            processing_time=mean_time,
            realtime_factor=mean_time / audio_duration,
            memory_peak_mb=max(0, mem_after - mem_before),
            metadata={
                "std_time": std_time,
                "all_times": times,
                "stems": separator.get_available_stems(),
            },
        )

    def compare_separators(
        self,
        audio_path: Union[str, Path],
        separators: Optional[List[str]] = None,
    ) -> str:
        """
        Compare separators and return formatted comparison.

        Args:
            audio_path: Path to test audio
            separators: Separators to compare

        Returns:
            Formatted comparison string
        """
        suite = self.run_all(
            audio_path,
            separators=separators,
            include_classifier=False,
        )

        lines = [
            "",
            "SEPARATOR COMPARISON",
            "=" * 60,
            "",
            f"{'Separator':<15} {'Model':<20} {'Time (s)':<10} {'RTF':<8} {'Memory'}",
            "-" * 60,
        ]

        for r in suite.results:
            lines.append(
                f"{r.name:<15} "
                f"{r.model:<20} "
                f"{r.processing_time:<10.2f} "
                f"{r.realtime_factor:<8.2f}x "
                f"{r.memory_peak_mb:.0f}MB"
            )

        lines.append("")
        lines.append("RTF = Realtime Factor (lower is faster)")
        lines.append("")

        return "\n".join(lines)


def run_quick_benchmark(audio_path: str):
    """Run a quick benchmark from command line."""
    bench = Benchmark(warmup_runs=0, benchmark_runs=1)
    suite = bench.run_all(audio_path)
    print(suite.summary())
    return suite


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <audio_file>")
        sys.exit(1)

    run_quick_benchmark(sys.argv[1])

# SampleScout

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/drewbrosnan/samplescout/actions/workflows/tests.yml/badge.svg)](https://github.com/drewbrosnan/samplescout/actions/workflows/tests.yml)

AI-powered audio sampling system that classifies, separates, and processes audio for use in synthesizers.

## Features

- **Audio Classification** - YAMNet-based classification with 521 sound categories
- **Source Separation** - Multiple backends: Demucs, Spleeter, AudioSep
- **Unified Pipeline** - Classify → Separate → Process workflow
- **Benchmark Suite** - Compare separation tools on your audio
- **Edge-Ready** - Optimized for NVIDIA Jetson Orin Nano deployment

## Quick Start

```bash
# Clone the repository
git clone https://github.com/drewbrosnan/samplescout.git
cd samplescout

# Run setup
make setup

# Activate environment
source venv/bin/activate

# Run demo
make demo

# Run tests
make test
```

## Installation

### Prerequisites

- Python 3.9+
- FFmpeg (`brew install ffmpeg` on macOS)
- 8GB+ RAM recommended

### Setup

```bash
# Create virtual environment and install dependencies
make setup

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Command Line

```bash
# Classify audio
python -m samplescout classify samples/input/audio.wav

# Separate audio (default: demucs)
python -m samplescout separate samples/input/audio.wav --output samples/output/

# Full pipeline: classify then separate based on detected sounds
python -m samplescout pipeline samples/input/audio.wav

# Benchmark all separators
python -m samplescout benchmark samples/input/audio.wav
```

### Python API

```python
from samplescout import SampleScout

# Initialize pipeline
scout = SampleScout(separator="demucs")

# Classify audio
results = scout.classify("audio.wav")
print(f"Detected: {results.top_sounds}")

# Separate audio
stems = scout.separate("audio.wav", output_dir="output/")
print(f"Separated: {list(stems.keys())}")

# Full pipeline
output = scout.process("audio.wav")
```

## Architecture

```
SampleScout Pipeline:

  [Audio Input] → [YAMNet Classify] → [Sound Categories]
                                              ↓
                                    [User Selection / Auto]
                                              ↓
                        [Demucs/Spleeter/AudioSep Separate]
                                              ↓
                                    [Processed Stems]
                                              ↓
                    [Pitch Shift → SF2/SFZ Soundfont Generation]
```

## Project Structure

```
samplescout/
├── src/
│   ├── __init__.py
│   ├── classifier.py      # YAMNet audio classification
│   ├── pipeline.py        # Unified processing pipeline
│   ├── utils.py           # Audio utilities
│   └── separators/
│       ├── __init__.py
│       ├── base.py        # Abstract separator interface
│       ├── demucs.py      # Demucs wrapper
│       ├── spleeter.py    # Spleeter wrapper
│       └── audiosep.py    # AudioSep wrapper
├── tests/
│   ├── test_classifier.py
│   ├── test_separators.py
│   ├── test_pipeline.py
│   └── benchmark.py       # Performance benchmarks
├── samples/
│   ├── input/             # Input audio files
│   └── output/            # Processed output
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
├── Makefile               # Common commands
└── README.md
```

## Benchmarks

Run benchmarks to compare separation tools:

```bash
make benchmark AUDIO=samples/input/song.wav
```

Results are saved to `samples/output/benchmark/benchmark_results.json`.

| Tool | Quality | Speed | Memory | Best For |
|------|---------|-------|--------|----------|
| Demucs | ★★★★★ | Slow | High | Highest quality separation |
| Spleeter | ★★★☆☆ | Fast | Low | Real-time / edge deployment |
| AudioSep | ★★★★☆ | Medium | Medium | Flexible sound targeting |

## Phase 2: Jetson Deployment

See [docs/jetson-deployment.md](docs/jetson-deployment.md) for edge deployment guide.

### Expected Performance on Jetson Orin Nano

- YAMNet: ~500ms per second of audio
- Spleeter (TensorRT): ~1.5s for 5s audio chunk
- Total pipeline: ~2-3s for 5s sample

## Development

```bash
# Install dev dependencies
make setup-dev

# Run tests
make test

# Run linter
make lint

# Format code
make format

# Run all checks
make check
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) - Audio classification
- [Demucs](https://github.com/facebookresearch/demucs) - Music source separation
- [Spleeter](https://github.com/deezer/spleeter) - Fast audio separation
- [AudioSep](https://github.com/Audio-AGI/AudioSep) - Language-guided separation

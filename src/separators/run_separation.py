#!/usr/bin/env python3
"""
Subprocess runner for audio separation.
Runs separation in an isolated process to avoid import conflicts.

Usage:
    python -m src.separators.run_separation <separator> <input_path> <output_dir> [--prompt <prompt>]
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def run_separation(separator_name: str, input_path: str, output_dir: str, prompt: str = None):
    """Run separation and output results as JSON."""
    import time
    start_time = time.time()

    try:
        from src.separators import get_separator
        from src.utils import save_audio
        import re

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get separator
        if separator_name == "spleeter":
            separator = get_separator(separator_name, stems=4)
        else:
            separator = get_separator(separator_name, device="cpu")

        # Run separation
        if separator_name == "audiosep" and prompt:
            result = separator.separate(str(input_path), prompts=[prompt])
        else:
            result = separator.separate(str(input_path))

        # Save stems
        stem_paths = {}
        for stem_name, audio in result.stems.items():
            safe_stem_name = re.sub(r'[^a-zA-Z0-9]', '_', stem_name)
            stem_filename = f"{safe_stem_name}.wav"
            stem_path = output_dir / stem_filename
            save_audio(stem_path, audio, result.sample_rate)
            stem_paths[stem_name] = str(stem_path)

        elapsed = time.time() - start_time

        output = {
            "success": True,
            "stems": stem_paths,
            "sample_rate": result.sample_rate,
            "processing_time": elapsed,
            "separator": separator_name,
            "model": separator.model_name,
        }

    except Exception as e:
        import traceback
        output = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    print(json.dumps(output))
    return output


def main():
    parser = argparse.ArgumentParser(description="Run audio separation")
    parser.add_argument("separator", choices=["demucs", "spleeter", "audiosep", "mdxnet"])
    parser.add_argument("input_path", help="Path to input audio file")
    parser.add_argument("output_dir", help="Directory to save output stems")
    parser.add_argument("--prompt", help="Text prompt for AudioSep")

    args = parser.parse_args()

    result = run_separation(args.separator, args.input_path, args.output_dir, args.prompt)
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()

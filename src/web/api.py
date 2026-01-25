"""
SampleScout Web API - FastAPI backend for audio processing.
"""

import asyncio
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Note: Separation runs in subprocess to avoid torch import conflicts

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="SampleScout",
    description="AI-powered audio sampling and analysis",
    version="0.1.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "src" / "web" / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ============================================================================
# Models
# ============================================================================

class AudioFile(BaseModel):
    id: str
    filename: str
    path: str
    size: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    uploaded_at: str


class ClassificationResultResponse(BaseModel):
    file_id: str
    top_classes: List[Dict]  # [{"name": str, "score": float}]
    processing_time: float


class SeparationResult(BaseModel):
    file_id: str
    separator: str
    model: str
    stems: Dict[str, str]  # stem_name -> output path
    processing_time: float


class ProcessingStatus(BaseModel):
    file_id: str
    status: str  # pending, processing, completed, error
    progress: float
    message: str
    result: Optional[Dict] = None


# ============================================================================
# In-memory storage (for demo - use database in production)
# ============================================================================

audio_files: Dict[str, AudioFile] = {}
processing_jobs: Dict[str, ProcessingStatus] = {}
classification_results: Dict[str, ClassificationResultResponse] = {}
separation_results: Dict[str, List[SeparationResult]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def get_audio_info(path: Path) -> dict:
    """Get audio file information."""
    try:
        from src.utils import get_audio_info as _get_info
        return _get_info(path)
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(PROJECT_ROOT / "src" / "web" / "static" / "index.html")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/files")
async def list_files():
    """List all uploaded audio files."""
    return {"files": list(audio_files.values())}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an audio file."""
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"File type {ext} not allowed. Use: {allowed_extensions}")

    # Generate unique ID
    file_id = str(uuid.uuid4())[:8]

    # Save file
    filename = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Get audio info
    info = get_audio_info(file_path)

    # Store metadata
    audio_file = AudioFile(
        id=file_id,
        filename=file.filename,
        path=f"/uploads/{filename}",
        size=len(content),
        duration=info.get("duration"),
        sample_rate=info.get("sample_rate"),
        uploaded_at=datetime.now().isoformat(),
    )
    audio_files[file_id] = audio_file

    return {"success": True, "file": audio_file}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    if file_id not in audio_files:
        raise HTTPException(404, "File not found")

    audio_file = audio_files[file_id]
    file_path = PROJECT_ROOT / audio_file.path.lstrip("/")

    if file_path.exists():
        file_path.unlink()

    del audio_files[file_id]

    # Clean up results
    if file_id in classification_results:
        del classification_results[file_id]
    if file_id in separation_results:
        del separation_results[file_id]

    return {"success": True}


@app.post("/api/classify/{file_id}")
async def classify_audio(file_id: str, background_tasks: BackgroundTasks):
    """Classify audio using YAMNet."""
    if file_id not in audio_files:
        raise HTTPException(404, "File not found")

    # Create processing job
    job_id = f"classify_{file_id}"
    processing_jobs[job_id] = ProcessingStatus(
        file_id=file_id,
        status="processing",
        progress=0.0,
        message="Starting classification...",
    )

    # Run classification in background
    background_tasks.add_task(run_classification, file_id, job_id)

    return {"job_id": job_id, "status": "started"}


def run_classification(file_id: str, job_id: str):
    """Run YAMNet classification (background task)."""
    try:
        from src.classifier import YAMNetClassifier

        audio_file = audio_files[file_id]
        file_path = PROJECT_ROOT / audio_file.path.lstrip("/")

        processing_jobs[job_id].message = "Loading YAMNet model..."
        processing_jobs[job_id].progress = 0.2

        classifier = YAMNetClassifier()

        processing_jobs[job_id].message = "Classifying audio..."
        processing_jobs[job_id].progress = 0.5

        result = classifier.classify(str(file_path), top_k=20, min_score=0.01)

        # Store result
        classification_results[file_id] = ClassificationResultResponse(
            file_id=file_id,
            top_classes=[{"name": name, "score": score} for name, score in result.top_classes],
            processing_time=result.processing_time,
        )

        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 1.0
        processing_jobs[job_id].message = "Classification complete"
        processing_jobs[job_id].result = classification_results[file_id].model_dump()

    except Exception as e:
        processing_jobs[job_id].status = "error"
        processing_jobs[job_id].message = str(e)


@app.post("/api/separate/{file_id}")
async def separate_audio(
    file_id: str,
    separator: str = "demucs",
    prompt: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
):
    """Separate audio into stems.

    For AudioSep, pass a text prompt describing the sound to isolate.
    Example: ?separator=audiosep&prompt=siren
    """
    if file_id not in audio_files:
        raise HTTPException(404, "File not found")

    valid_separators = ["demucs", "spleeter", "audiosep", "mdxnet"]
    if separator not in valid_separators:
        raise HTTPException(400, f"Invalid separator. Use: {valid_separators}")

    # AudioSep requires a prompt
    if separator == "audiosep" and not prompt:
        raise HTTPException(400, "AudioSep requires a 'prompt' parameter")

    # Create processing job - include prompt in job_id for AudioSep
    if separator == "audiosep" and prompt:
        # Sanitize prompt for job_id - remove all special chars
        import re
        safe_prompt = re.sub(r'[^a-zA-Z0-9]', '_', prompt)[:20]
        job_id = f"separate_{file_id}_{separator}_{safe_prompt}"
    else:
        job_id = f"separate_{file_id}_{separator}"

    processing_jobs[job_id] = ProcessingStatus(
        file_id=file_id,
        status="processing",
        progress=0.0,
        message=f"Starting {separator} separation..." + (f" (prompt: {prompt})" if prompt else ""),
    )

    # Run separation in background
    background_tasks.add_task(run_separation, file_id, separator, job_id, prompt)

    return {"job_id": job_id, "status": "started"}


def run_separation(file_id: str, separator_name: str, job_id: str, prompt: Optional[str] = None):
    """Run audio separation via subprocess to avoid import conflicts.

    Args:
        file_id: ID of the uploaded audio file
        separator_name: Name of separator ('demucs', 'spleeter', 'audiosep', 'mdxnet')
        job_id: Job ID for tracking progress
        prompt: Text prompt for AudioSep (e.g., "siren", "music")
    """
    import subprocess
    import re

    try:
        audio_file = audio_files[file_id]
        file_path = PROJECT_ROOT / audio_file.path.lstrip("/")

        # Create output directory
        if separator_name == "audiosep" and prompt:
            safe_prompt = re.sub(r'[^a-zA-Z0-9]', '_', prompt)[:30]
            output_dir = OUTPUT_DIR / file_id / separator_name / safe_prompt
        else:
            output_dir = OUTPUT_DIR / file_id / separator_name
        output_dir.mkdir(parents=True, exist_ok=True)

        processing_jobs[job_id].message = f"Starting {separator_name} subprocess..."
        processing_jobs[job_id].progress = 0.2

        # Build subprocess command using venv python
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        cmd = [
            str(venv_python), "-m", "src.separators.run_separation",
            separator_name,
            str(file_path),
            str(output_dir),
        ]
        if prompt:
            cmd.extend(["--prompt", prompt])

        if prompt:
            processing_jobs[job_id].message = f"Separating '{prompt}' from audio..."
        else:
            processing_jobs[job_id].message = "Separating audio..."
        processing_jobs[job_id].progress = 0.4

        # Run subprocess with ffmpeg in PATH
        env = os.environ.copy()
        venv_bin = PROJECT_ROOT / ".venv" / "bin"
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=600,  # 10 minute timeout
        )

        processing_jobs[job_id].progress = 0.8

        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed (rc={result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

        # Parse JSON output from subprocess - look for JSON in stdout
        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError(f"Subprocess returned empty stdout.\nSTDERR: {result.stderr}")

        # Find JSON object in output (might have other prints before it)
        # Look for {"success": which is how our output starts
        json_start = stdout.find('{"success":')
        if json_start == -1:
            raise RuntimeError(f"No JSON found in stdout: {stdout}\nSTDERR: {result.stderr}")

        # Find the end of the JSON object
        json_str = stdout[json_start:]
        # The JSON ends at the first newline or end of string
        json_end = json_str.find('\n')
        if json_end != -1:
            json_str = json_str[:json_end]

        output = json.loads(json_str)

        if not output.get("success"):
            raise RuntimeError(output.get("error", "Unknown error"))

        # Convert absolute paths to web paths
        stem_paths = {}
        for stem_name, abs_path in output["stems"].items():
            rel_path = Path(abs_path).relative_to(OUTPUT_DIR)
            stem_paths[stem_name] = f"/outputs/{rel_path}"

        # Store result
        result_separator = f"{separator_name}:{prompt}" if separator_name == "audiosep" and prompt else separator_name

        sep_result = SeparationResult(
            file_id=file_id,
            separator=result_separator,
            model=output.get("model", separator_name),
            stems=stem_paths,
            processing_time=output.get("processing_time", 0),
        )

        if file_id not in separation_results:
            separation_results[file_id] = []

        separation_results[file_id] = [
            r for r in separation_results[file_id] if r.separator != result_separator
        ]
        separation_results[file_id].append(sep_result)

        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 1.0
        if separator_name == "audiosep" and prompt:
            processing_jobs[job_id].message = f"AudioSep '{prompt}' separation complete"
        else:
            processing_jobs[job_id].message = f"{separator_name} separation complete"
        processing_jobs[job_id].result = sep_result.model_dump()

    except subprocess.TimeoutExpired:
        processing_jobs[job_id].status = "error"
        processing_jobs[job_id].message = "Separation timed out after 10 minutes"
    except Exception as e:
        import traceback
        processing_jobs[job_id].status = "error"
        processing_jobs[job_id].message = f"{str(e)}\n{traceback.format_exc()}"


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing job status."""
    if job_id not in processing_jobs:
        raise HTTPException(404, "Job not found")
    return processing_jobs[job_id]


@app.get("/api/results/{file_id}")
async def get_results(file_id: str):
    """Get all results for a file."""
    if file_id not in audio_files:
        raise HTTPException(404, "File not found")

    return {
        "file": audio_files[file_id],
        "classification": classification_results.get(file_id),
        "separations": separation_results.get(file_id, []),
    }


@app.get("/api/separators")
async def list_separators():
    """List available separators."""
    return {
        "separators": [
            {
                "id": "demucs",
                "name": "Demucs",
                "description": "High-quality hybrid transformer model (Meta)",
                "stems": ["drums", "bass", "vocals", "other"],
            },
            {
                "id": "spleeter",
                "name": "Spleeter",
                "description": "Fast separation model (Deezer)",
                "stems": ["drums", "bass", "vocals", "other"],
            },
            {
                "id": "audiosep",
                "name": "AudioSep",
                "description": "Language-guided separation (experimental)",
                "stems": ["custom prompts"],
            },
        ]
    }


# ============================================================================
# Run Server
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

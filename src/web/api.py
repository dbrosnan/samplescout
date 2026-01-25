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
    background_tasks: BackgroundTasks = None,
):
    """Separate audio into stems."""
    if file_id not in audio_files:
        raise HTTPException(404, "File not found")

    valid_separators = ["demucs", "spleeter", "audiosep"]
    if separator not in valid_separators:
        raise HTTPException(400, f"Invalid separator. Use: {valid_separators}")

    # Create processing job
    job_id = f"separate_{file_id}_{separator}"
    processing_jobs[job_id] = ProcessingStatus(
        file_id=file_id,
        status="processing",
        progress=0.0,
        message=f"Starting {separator} separation...",
    )

    # Run separation in background
    background_tasks.add_task(run_separation, file_id, separator, job_id)

    return {"job_id": job_id, "status": "started"}


def run_separation(file_id: str, separator_name: str, job_id: str):
    """Run audio separation (background task)."""
    try:
        from src.separators import get_separator

        audio_file = audio_files[file_id]
        file_path = PROJECT_ROOT / audio_file.path.lstrip("/")

        processing_jobs[job_id].message = f"Loading {separator_name} model..."
        processing_jobs[job_id].progress = 0.2

        # Get separator
        if separator_name == "spleeter":
            separator = get_separator(separator_name, stems=4)
        else:
            separator = get_separator(separator_name, device="cpu")

        processing_jobs[job_id].message = "Separating audio..."
        processing_jobs[job_id].progress = 0.4

        # Run separation
        result = separator.separate(str(file_path))

        processing_jobs[job_id].message = "Saving stems..."
        processing_jobs[job_id].progress = 0.8

        # Create output directory for this file
        output_dir = OUTPUT_DIR / file_id / separator_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save stems
        stem_paths = {}
        for stem_name, audio in result.stems.items():
            stem_filename = f"{stem_name}.wav"
            stem_path = output_dir / stem_filename

            from src.utils import save_audio
            save_audio(stem_path, audio, result.sample_rate)

            stem_paths[stem_name] = f"/outputs/{file_id}/{separator_name}/{stem_filename}"

        # Store result
        sep_result = SeparationResult(
            file_id=file_id,
            separator=separator_name,
            model=separator.model_name,
            stems=stem_paths,
            processing_time=result.duration,
        )

        if file_id not in separation_results:
            separation_results[file_id] = []

        # Replace existing result for same separator
        separation_results[file_id] = [
            r for r in separation_results[file_id] if r.separator != separator_name
        ]
        separation_results[file_id].append(sep_result)

        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 1.0
        processing_jobs[job_id].message = f"{separator_name} separation complete"
        processing_jobs[job_id].result = sep_result.model_dump()

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

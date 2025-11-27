from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import tempfile, uuid, shutil, subprocess, sys, os

app = FastAPI()

@app.post("/scanpath/")
async def scanpath(file: UploadFile = File(None)):
    # 1) Validate input early
    if file is None:
        raise HTTPException(status_code=400, detail="Missing form field 'file'. Send a video via multipart/form-data with key = 'file'.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    # Optional: basic content-type guard (relax if needed)
    if file.content_type and not file.content_type.startswith(("video/", "application/octet-stream")):
        raise HTTPException(status_code=415, detail=f"Unsupported content-type: {file.content_type}. Expected a video/* file.")

    # 2) Work in an isolated temp directory with absolute paths
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Keep original suffix if possible, else default to .mp4
        suffix = Path(file.filename).suffix or ".mp4"
        temp_video = tmpdir / f"temp_{uuid.uuid4().hex}{suffix}"
        output_path = tmpdir / f"scanpath_{uuid.uuid4().hex}.png"

        # 3) Save uploaded file to disk and validate size
        try:
            with temp_video.open("wb") as buf:
                shutil.copyfileobj(file.file, buf)
        finally:
            await file.close()

        if not temp_video.exists() or temp_video.stat().st_size < 1024:  # 1 KB sanity check
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file is empty or too small (size={temp_video.stat().st_size if temp_video.exists() else 0} bytes)."
            )

        # 4) Resolve script path and interpreter
        script_path = Path(__file__).parent / "generate_scanpaths.py"
        if not script_path.exists():
            raise HTTPException(status_code=500, detail=f"Script not found: {script_path}")

        python_exec = sys.executable  # ensures same conda/env python

        # 5) Run the script, capture logs (don’t raise automatically)
        proc = subprocess.run(
            [python_exec, str(script_path), str(temp_video), str(output_path)],
            capture_output=True,
            text=True,
            cwd=str(tmpdir),
            check=False
        )

        # 6) Decide response based on file existence + return code
        if output_path.exists() and output_path.stat().st_size > 0:
            # Success: stream the PNG
            return FileResponse(
                path=str(output_path),
                media_type="image/png",
                filename=output_path.name
            )

        # If no file, return detailed diagnostics
        error_payload = {
            "status": "failed",
            "error": "No scanpath generated",
            "script_returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
            "video_size": temp_video.stat().st_size if temp_video.exists() else 0,
            "script_used": str(script_path),
            "python_used": python_exec,
        }
        # If script failed (non-zero), reflect server error; else bad request
        status = 500 if proc.returncode != 0 else 400
        return JSONResponse(status_code=status, content=error_payload)
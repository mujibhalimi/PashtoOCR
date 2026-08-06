"""FastAPI server for the Pashto OCR model (mhalimi3008/pashtoOCR).

Run from the repo root:

  uvicorn fastOCR.main:app --port 8000

or from inside the fastOCR/ directory:

  uvicorn main:app --port 8000

Then open http://localhost:8000 — upload a PNG/JPG or a multi-page PDF and get
the recognized Pashto text back, or call the API directly:

  curl -F "file=@test.png" http://localhost:8000/ocr
"""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

try:  # uvicorn fastOCR.main:app (repo root)
    from .ocr_engine import device, load_model, ocr_file
except ImportError:  # uvicorn main:app (inside fastOCR/)
    from ocr_engine import device, load_model, ocr_file

WEIGHTS = "crnn_pashtoOCR.pt"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {WEIGHTS} (device={device})...")
    STATE["model"], STATE["id2char"] = load_model(WEIGHTS)
    yield
    STATE.clear()


app = FastAPI(title="Pashto OCR", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/ocr")
async def ocr(file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type '{suffix}' — "
                                 f"use one of: {', '.join(sorted(ALLOWED_SUFFIXES))}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")

    # ocr_file works on paths (pypdfium2 needs one for PDFs), so spool to a temp file
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            text = ocr_file(STATE["model"], STATE["id2char"], tmp.name)
        except Exception as e:
            raise HTTPException(422, f"could not read the file: {e}")

    return JSONResponse({"filename": file.filename, "text": text})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

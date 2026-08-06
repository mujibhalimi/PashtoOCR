# Pashto OCR — CRNN + CTC (printed)

An OCR model for **Pashto** (right-to-left, Arabic script) that reads **printed
text**. Handwriting support is planned but not available yet — the current model
only supports and runs on printed text. The model is trained on Kaggle and published
to the Hugging Face Hub: **[`mhalimi3008/pashtoOCR`](https://huggingface.co/mhalimi3008/pashtoOCR)**.

Inference accepts **PNG / JPG / any common image format / multi-page PDF**.

## Framework & tech stack

| Layer | Technology |
|-------|------------|
| Deep learning framework | **PyTorch** (`torch`, trained with CTC loss) |
| Image processing | OpenCV (`opencv-python`), Pillow, NumPy |
| Model hosting / download | Hugging Face Hub (`huggingface_hub`) |
| PDF rendering | `pypdfium2` (300 dpi) |
| RTL terminal display | `arabic-reshaper` + `python-bidi` |
| Training environment | Kaggle notebook, GPU (T4 x2 / P100) |

## Architecture

| Component | Details |
|-----------|---------|
| CNN backbone | VGG-style, grayscale 48 px-high line images → feature sequence (width ÷ 4) |
| Sequence model | 2-layer bidirectional LSTM (256 hidden per direction) |
| Loss / decoding | CTC, character vocabulary from the training data; greedy decode |
| Size | ~12 M parameters |

Pashto is right-to-left while CTC alignment is monotonic left-to-right, so every
line image is horizontally flipped during preprocessing (training *and* inference)
while labels stay in logical order — the standard trick for Arabic-script CTC.

## Training pipeline (Kaggle)

The model is trained on Kaggle (GPU T4 x2 / P100) on
[`zirak-ai/PashtoOCR`](https://huggingface.co/datasets/zirak-ai/PashtoOCR) —
10,000 synthetic printed paragraph images → ~60k line crops — and the resulting
weights (`crnn_pashtoOCR.pt`, `crnn.pt`) and `charset.json` are pushed to the Hub.
The training notebook is kept private and is not part of this repo.

Expected accuracy: low single-digit CER on the printed validation set.

A second training stage — fine-tuning on the **KPTI** handwritten dataset
([github.com/rahmad77/KPTI](https://github.com/rahmad77/KPTI)) — is planned but
not yet published.

## Testing the model locally

`test_model.py` downloads the trained weights and charset from the Hub automatically
on first run (cached in `~/.cache/huggingface` afterwards). A sample image
(`test.png`) is included in this repo.

```bash
# install dependencies (Python 3.10+)
pip install torch opencv-python pillow numpy huggingface_hub \
            arabic-reshaper python-bidi pypdfium2

# OCR the included sample image
python test_model.py test.png

# OCR any image or a multi-page PDF
python test_model.py page.jpg
python test_model.py scan.pdf

# save the recognized text (logical order, ready to paste) to a file
python test_model.py test.png -o out.txt

# use the earlier checkpoint instead of the default crnn_pashtoOCR.pt
python test_model.py test.png --weights crnn.pt
```

Terminal output is reshaped for RTL display (joined glyphs, visual order) because most
terminals can't render Arabic script; use `--raw` (or `-o file.txt`) for the
logical-order text you'd paste into a document. The script auto-selects the device:
Apple Silicon **MPS** → CUDA → CPU.

## Web API + browser UI (FastAPI)

`fastOCR/` contains a small FastAPI server with a web page for uploading a
PNG/JPG or PDF and getting the recognized text back (rendered right-to-left):

```bash
pip install -r fastOCR/requirements.txt
uvicorn fastOCR.main:app --port 8000     # run from the repo root
```

Open **http://localhost:8000** in a browser, or call the API directly:

```bash
curl -F "file=@test.png" http://localhost:8000/ocr
# → {"filename": "test.png", "text": "..."}
```

## Using the trained model in your own code

```python
import json, torch
from huggingface_hub import hf_hub_download
from test_model import CRNN, ocr_file  # architecture + full page/PDF pipeline

REPO = "mhalimi3008/pashtoOCR"
weights = hf_hub_download(REPO, "crnn_pashtoOCR.pt")
cfg = json.loads(open(hf_hub_download(REPO, "charset.json")).read())

model = CRNN(len(cfg["charset"]) + 1)
model.load_state_dict(torch.load(weights, map_location="cpu"))
model.eval()
```

For full pages and PDFs use `segment_lines` / `ocr_image` / `ocr_file` from
`test_model.py` (automatic line segmentation → per-line recognition).

## Repo structure

| File | Purpose |
|------|---------|
| `test_model.py` | Local inference script (images, PDFs) |
| `fastOCR/main.py` | FastAPI server (`/ocr` endpoint + serves the web UI) |
| `fastOCR/index.html` | Browser upload page with RTL result display |
| `fastOCR/requirements.txt` | Dependencies for the API server |
| `test.png` | Sample Pashto image for testing |
| `README.md` | This file |

Datasets, model checkpoints and other heavy files are intentionally
**not** stored in this repo — weights live on the
[Hugging Face Hub](https://huggingface.co/mhalimi3008/pashtoOCR) and are
downloaded on demand.

## Limitations

- **Printed text only (for now):** the published weights are trained purely on
  printed/synthetic Pashto text. Handwriting (manuscripts, notes, forms) is **not
  supported yet** — a handwriting fine-tune is planned.
- **Layout:** the line splitter assumes roughly horizontal lines; for skewed/complex
  layouts put a text detector (CRAFT / PaddleOCR DBNet) in front.

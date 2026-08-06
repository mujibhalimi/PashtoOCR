# Pashto OCR — CRNN + CTC (printed + handwritten)

An OCR model for **Pashto** (right-to-left, Arabic script) that reads both printed
text and handwritten manuscript lines. The model is trained on Kaggle and published
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
| Evaluation metrics | `jiwer` (CER / WER) |
| Training environment | Kaggle notebook, GPU (T4 x2 / P100) |

## Architecture

| Component | Details |
|-----------|---------|
| CNN backbone | VGG-style, grayscale 48 px-high line images → feature sequence (width ÷ 4) |
| Sequence model | 2-layer bidirectional LSTM (256 hidden per direction) |
| Loss / decoding | CTC, character vocabulary from both datasets; greedy decode |
| Size | ~12 M parameters |

Pashto is right-to-left while CTC alignment is monotonic left-to-right, so every
line image is horizontally flipped during preprocessing (training *and* inference)
while labels stay in logical order — the standard trick for Arabic-script CTC.

## Training pipeline (Kaggle)

Two-stage training in `pashto_ocr_kaggle.ipynb`:

1. **Stage 1 — printed** (~25 min): [`zirak-ai/PashtoOCR`](https://huggingface.co/datasets/zirak-ai/PashtoOCR)
   — 10,000 synthetic paragraph images → ~60k line crops.
2. **Stage 2 — handwriting** (~15 min): fine-tune on **KPTI**
   ([github.com/rahmad77/KPTI](https://github.com/rahmad77/KPTI)) — 17,015 real
   hand-scribed (katib) Pashto text lines scanned at 300 dpi from books, with UTF-8
   ground truth.

### How to run the training

1. kaggle.com → **Create → Notebook** → **File → Import Notebook** → upload
   `pashto_ocr_kaggle.ipynb`.
2. Settings panel: **Accelerator:** GPU **T4 x2** (P100 / single T4 also fine) ·
   **Internet: ON**.
3. Add your Hugging Face **write** token ([hf.co/settings/tokens](https://huggingface.co/settings/tokens))
   via **Add-ons → Secrets** as `HF_TOKEN` and attach it to the notebook.
4. **Run All.** First pass with `CFG.quick_test = True` (~10 min) to verify everything
   end-to-end including the Hub push, then set it back to `False` for the real run
   (~45–55 min total).
5. Outputs: `pashto_ocr_model.zip` in the **Output** tab, and on the Hub —
   `crnn.pt` (printed), `crnn_handwriting.pt` (printed + handwritten), `charset.json`,
   and an auto-generated model card with the final CER/WER.

Expected accuracy: low single-digit CER on the printed validation set; ~10% CER on the
KPTI handwriting test set (the published MDLSTM benchmark on KPTI is ~9–10% CER —
handwriting is intrinsically harder than print).

## Testing the model locally

`test_model.py` downloads the trained weights and charset from the Hub automatically
on first run (cached in `~/.cache/huggingface` afterwards). A sample image
(`test.png`) is included in this repo.

```bash
# install dependencies (Python 3.10+)
pip install torch opencv-python pillow numpy huggingface_hub \
            arabic-reshaper python-bidi pypdfium2 jiwer

# OCR the included sample image
python test_model.py test.png

# OCR any image or a multi-page PDF
python test_model.py page.jpg
python test_model.py scan.pdf

# save the recognized text (logical order, ready to paste) to a file
python test_model.py test.png -o out.txt

# use the printed-only weights instead of the handwriting fine-tune
python test_model.py test.png --weights crnn.pt

# evaluate CER/WER on the KPTI handwritten test set (clones KPTI, ~207 MB, one-time)
python test_model.py --benchmark
```

Terminal output is reshaped for RTL display (joined glyphs, visual order) because most
terminals can't render Arabic script; use `--raw` (or `-o file.txt`) for the
logical-order text you'd paste into a document. The script auto-selects the device:
Apple Silicon **MPS** → CUDA → CPU.

## Using the trained model in your own code

```python
import json, torch
from huggingface_hub import hf_hub_download
from test_model import CRNN, ocr_file  # architecture + full page/PDF pipeline

REPO = "mhalimi3008/pashtoOCR"
weights = hf_hub_download(REPO, "crnn_handwriting.pt")   # or "crnn.pt" for printed-only
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
| `pashto_ocr_kaggle.ipynb` | Complete Kaggle training notebook (data → two-stage training → evaluation → Hub push → inference incl. PDF) |
| `test_model.py` | Local inference + benchmark script (images, PDFs, KPTI CER/WER) |
| `test.png` | Sample Pashto image for testing |
| `README.md` | This file |

Datasets (KPTI, ...), model checkpoints and other heavy files are intentionally
**not** stored in this repo — weights live on the
[Hugging Face Hub](https://huggingface.co/mhalimi3008/pashtoOCR) and datasets are
downloaded on demand.

## Limitations

- **Katib-style manuscript/book handwriting:** covered by the KPTI fine-tune — the best
  openly available handwritten-Pashto data today.
- **Arbitrary personal handwriting** (notes, letters, forms): partially covered at best.
  For a specific writer/document style, transcribing even 1–2k of your own lines and
  running one more fine-tune round gives the largest possible accuracy jump.
- **Layout:** the line splitter assumes roughly horizontal lines; for skewed/complex
  layouts put a text detector (CRAFT / PaddleOCR DBNet) in front.

**Citation requirement:** research use of the KPTI-fine-tuned weights must cite
*Ahmad et al., "KPTI: Katib's Pashto Text Imagebase and Deep Learning Benchmark", ICFHR 2016.*

"""Test the published Pashto OCR model (mhalimi3008/pashtoOCR) locally.

Usage:
  python test_model.py page.png                 # OCR an image (PNG/JPG/BMP/TIFF...)
  python test_model.py scan.pdf                 # OCR a multi-page PDF
  python test_model.py page.png -o out.txt      # also save raw text to a file
  python test_model.py page.png --weights crnn.pt   # use the earlier checkpoint

Terminal output is reshaped for RTL display (joined glyphs, visual order) because
most terminals can't render Arabic script; use --raw (or -o file.txt) for the
logical-order text you'd paste into a document.

All model code lives in fastOCR/ocr_engine.py (also used by the FastAPI server);
weights and charset are downloaded from the Hugging Face Hub on first run.
"""

import argparse
from pathlib import Path

from fastOCR.ocr_engine import (  # noqa: F401  (re-exported for library use)
    CRNN, HF_REPO, device, load_model, load_pages,
    ocr_file, ocr_image, ocr_line, segment_lines,
)

RAW_OUTPUT = False   # set by --raw: skip the RTL display reshaping


def term(text):
    # Terminals lack bidi + Arabic joining: reshape logical-order text into
    # visually-ordered presentation forms so Pashto reads correctly on screen.
    if RAW_OUTPUT:
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    return "\n".join(get_display(arabic_reshaper.reshape(line))
                     for line in text.split("\n"))


def main():
    ap = argparse.ArgumentParser(description="Test the Pashto OCR model")
    ap.add_argument("file", help="image or PDF to OCR")
    ap.add_argument("--weights", default="crnn_pashtoOCR.pt",
                    help="crnn_pashtoOCR.pt (default) or crnn.pt (earlier checkpoint)")
    ap.add_argument("-o", "--out", metavar="FILE",
                    help="write the recognized text (logical order) to this file")
    ap.add_argument("--raw", action="store_true",
                    help="print logical-order text without RTL display reshaping")
    args = ap.parse_args()
    global RAW_OUTPUT
    RAW_OUTPUT = args.raw

    print(f"Loading {args.weights} from {HF_REPO} (device={device})...")
    model, id2char = load_model(args.weights)

    text = ocr_file(model, id2char, args.file)
    print(term(text))
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n[saved raw text to {args.out}]")


if __name__ == "__main__":
    main()

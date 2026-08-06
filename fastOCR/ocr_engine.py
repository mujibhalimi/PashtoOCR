"""Pashto OCR engine — CRNN+CTC model, preprocessing, line segmentation, inference.

Weights and charset are downloaded from the Hugging Face Hub on first use
(cached in ~/.cache/huggingface afterwards).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

HF_REPO = "mhalimi3008/pashtoOCR"
DEFAULT_WEIGHTS = "crnn_pashtoOCR.pt"

IMG_H = 48
MAX_W = 1600
MIN_CROP_W = 8
DOWNSAMPLE = 4

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- model

def conv_bn(cin, cout, pool=None):
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return layers


class CRNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            *conv_bn(1, 64, pool=(2, 2)),
            *conv_bn(64, 128, pool=(2, 2)),
            *conv_bn(128, 256),
            *conv_bn(256, 256, pool=(2, 1)),
            *conv_bn(256, 512),
            *conv_bn(512, 512, pool=(2, 1)),
            nn.Conv2d(512, 512, (3, 3), padding=(0, 1)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):                 # x: (B, 1, 48, W)
        f = self.cnn(x)                   # (B, 512, 1, W/4)
        f = f.squeeze(2).permute(0, 2, 1) # (B, T, 512)
        out, _ = self.rnn(f)
        return self.fc(out)               # (B, T, n_classes)


def load_model(weights_name=DEFAULT_WEIGHTS):
    weights = hf_hub_download(HF_REPO, weights_name)
    cfg = json.loads(Path(hf_hub_download(HF_REPO, "charset.json")).read_text())
    charset = cfg["charset"]
    id2char = {i + 1: c for i, c in enumerate(charset)}
    model = CRNN(len(charset) + 1).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location=device))
    return model, id2char


# ---------------------------------------------------------- preprocessing

def maybe_invert(img):
    if np.asarray(img.convert("L")).mean() < 127:
        return ImageOps.invert(img)
    return img


def preprocess_crop(img):
    g = maybe_invert(img).convert("L")
    w, h = g.size
    new_w = max(MIN_CROP_W, min(MAX_W, round(w * IMG_H / h)))
    return g.resize((new_w, IMG_H), Image.BILINEAR)


def to_tensor(img):
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
    t = (t - 0.5) / 0.5
    return torch.flip(t, dims=[2])        # hflip: RTL text -> left-to-right frames


def ctc_greedy_decode(best_ids, id2char):
    out, prev = [], 0
    for k in best_ids.tolist():
        if k != prev and k != 0:
            out.append(id2char[k])
        prev = k
    return "".join(out)


# ------------------------------------------------------ line segmentation

def binarize(gray):
    # Flatten photo illumination (shadows/gradients), then Otsu -> ink mask (255 = ink).
    k = max(15, (min(gray.shape) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, cv2.max(bg, 1), scale=255)
    thr = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return norm, thr


def strip_rules_and_borders(thr):
    # Remove ruled notebook lines, page borders and decorative bars.
    H, W = thr.shape
    horiz = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 3), 1)))
    # only subtract THIN horizontal structures (true ruled lines, not bold text strokes)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(horiz)
    thin = np.zeros_like(horiz)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_HEIGHT] <= 6:
            thin[lab == i] = 255
    vert = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H // 3))))
    thr = cv2.subtract(cv2.subtract(thr, thin), vert)
    return cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def segment_lines(pil_img, pad=4):
    # Split a page into text-line crops: binarize -> drop rules/borders -> smear
    # words into line blobs -> group connected components into lines (top-to-bottom).
    rgb = pil_img.convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    if gray.mean() < 127:
        gray = 255 - gray
    norm, thr = binarize(gray)
    thr = strip_rules_and_borders(thr)
    H, W = thr.shape

    # median stroke-blob height ~ text height scale
    n, _, stats, _ = cv2.connectedComponentsWithStats(thr)
    hs = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n)
          if stats[i, cv2.CC_STAT_AREA] >= 4 and stats[i, cv2.CC_STAT_HEIGHT] < H // 2]
    if not hs:
        return [Image.fromarray(norm)]
    text_h = max(6, int(np.median(hs)))

    # smear horizontally so words merge into line blobs (small vertical tolerance)
    smeared = cv2.dilate(thr, cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, text_h * 2), max(1, text_h // 4))))
    n, _, stats, _ = cv2.connectedComponentsWithStats(smeared)
    boxes = [stats[i, :4] for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= text_h * text_h
             and stats[i, cv2.CC_STAT_HEIGHT] >= max(6, text_h // 2)]

    # group blobs whose vertical centers fall in the same band into one line
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    lines = []
    for x, y, w, h in boxes:
        cy = y + h / 2
        if lines and cy < lines[-1]["y1"] and (
                min(lines[-1]["y1"], y + h) - max(lines[-1]["y0"], y)
                > 0.5 * min(h, lines[-1]["y1"] - lines[-1]["y0"])):
            ln = lines[-1]
            ln["x0"], ln["y0"] = min(ln["x0"], x), min(ln["y0"], y)
            ln["x1"], ln["y1"] = max(ln["x1"], x + w), max(ln["y1"], y + h)
        else:
            lines.append({"x0": x, "y0": y, "x1": x + w, "y1": y + h})

    # Reference line height = width-weighted median, so wide real text lines set the
    # scale and small noise blobs don't. Then drop far-too-short boxes (specks,
    # leftover rule fragments) and far-too-tall ones (pen photos, graphics).
    hs = np.array([ln["y1"] - ln["y0"] for ln in lines], dtype=float)
    ws = np.array([ln["x1"] - ln["x0"] for ln in lines], dtype=float)
    order = np.argsort(hs)
    cum = np.cumsum(ws[order])
    med_line_h = hs[order][np.searchsorted(cum, cum[-1] / 2)]
    lines = [ln for ln, h in zip(lines, hs)
             if 0.35 * med_line_h <= h <= 3.5 * med_line_h]
    if not lines:
        return [Image.fromarray(norm)]

    crops = []
    for ln in sorted(lines, key=lambda l: l["y0"]):
        x0, y0 = max(0, ln["x0"] - pad), max(0, ln["y0"] - pad)
        x1, y1 = min(W, ln["x1"] + pad), min(H, ln["y1"] + pad)
        crops.append(Image.fromarray(norm[y0:y1, x0:x1]))
    return crops


# -------------------------------------------------------------- inference

@torch.no_grad()
def ocr_line(model, id2char, crop):
    t = to_tensor(preprocess_crop(crop)).unsqueeze(0).to(device)
    best = model(t).argmax(2)[0].cpu()
    return ctc_greedy_decode(best, id2char)


def ocr_image(model, id2char, path_or_img):
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return "\n".join(ocr_line(model, id2char, c).strip() for c in segment_lines(img))


def load_pages(path, dpi=300):
    if str(path).lower().endswith(".pdf"):
        import pypdfium2 as pdfium
        return [page.render(scale=dpi / 72).to_pil() for page in pdfium.PdfDocument(str(path))]
    return [Image.open(str(path))]


def ocr_file(model, id2char, path):
    return "\n\n".join(ocr_image(model, id2char, pg) for pg in load_pages(path))

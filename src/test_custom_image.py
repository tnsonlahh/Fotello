import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def apply_correction(img, k1, k2=None):
    h, w = img.shape[:2]
    if k2 is None:
        k2 = k1 * k1 * 0.40
    f = max(w, h) * 0.9
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    d = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    new_k, roi = cv2.getOptimalNewCameraMatrix(k, d, (w, h), 1, (w, h))
    out = cv2.undistort(img, k, d, None, new_k)

    x, y, rw, rh = roi
    if 0 < rw < w and 0 < rh < h and rw > w * 0.4:
        crop = out[y : y + rh, x : x + rw]
        if crop.size > 0:
            out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return out


def estimate_k1_from_lines(gray):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        50,
        minLineLength=min(w, h) // 8,
        maxLineGap=20,
    )
    if lines is None or len(lines) < 5:
        return -0.15

    cx, cy = w / 2, h / 2
    dists = []
    for ln in lines[:60]:
        x1, y1, x2, y2 = ln[0]
        if np.hypot(x2 - x1, y2 - y1) < 30:
            continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        d = np.hypot(mx - cx, my - cy) / np.hypot(cx, cy)
        if d > 0.3:
            dists.append(d)
    if not dists:
        return -0.15
    return float(np.clip(-np.mean(dists) * 0.30, -0.35, 0.05))


def edge_score(gray):
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        40,
        minLineLength=gray.shape[1] // 10,
        maxLineGap=15,
    )
    if lines is None or len(lines) < 3:
        return 0.0

    angles = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angles.append(abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) % 180)
    a = np.array(angles)
    h_lines = np.sum((a < 12) | (a > 168))
    v_lines = np.sum((a > 78) & (a < 102))
    return float((h_lines + v_lines) / max(len(a), 1))


def find_best_k1(img, gray):
    est = estimate_k1_from_lines(gray)
    candidates = np.linspace(max(est - 0.12, -0.35), min(est + 0.05, 0.05), 9)
    best_k1, best_sc = -0.15, -1
    for k1 in candidates:
        if abs(k1) < 0.005:
            continue
        try:
            corr = apply_correction(img, float(k1))
            sc = edge_score(cv2.cvtColor(corr, cv2.COLOR_BGR2GRAY))
            if sc > best_sc:
                best_sc, best_k1 = sc, float(k1)
        except Exception:
            pass
    return best_k1, best_sc


def load_k1_from_weights(weights_path):
    if not weights_path:
        return None
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    with open(weights_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["global_k1"])


def main():
    parser = argparse.ArgumentParser(description="Test lens correction on one custom image")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument(
        "--weights",
        default="",
        help="Path to lens_correction_weights.json (optional). If omitted, adaptive mode is used.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    img = cv2.imread(str(in_path))
    if img is None:
        raise ValueError(f"Cannot read image: {in_path}")

    if args.weights:
        k1 = load_k1_from_weights(args.weights)
        print(f"Using weights mode: k1={k1:.4f}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k1, score = find_best_k1(img, gray)
        print(f"Using adaptive mode: k1={k1:.4f}, score={score:.4f}")

    corrected = apply_correction(img, k1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"Failed to save output image: {out_path}")

    print(f"Saved corrected image to: {out_path}")


if __name__ == "__main__":
    main()

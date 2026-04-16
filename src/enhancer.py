from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class EnhanceConfig:
    # Geometry correction
    enable_straighten: bool = True
    enable_perspective: bool = True

    # Brightness adjustment
    auto_brightness: bool = True
    target_luma: float = 0.52
    highlight_threshold: int = 245
    max_brightness_drop: float = 0.28

    # Color correction
    auto_white_balance: bool = True
    use_clahe: bool = True
    clahe_clip_limit: float = 3.5
    clahe_grid_size: int = 8
    saturation_boost: float = 1.35

    # Manual overrides (applied at the end)
    manual_exposure: float = 0.0  # EV-like: -1.0..1.0
    manual_saturation: float = 1.0
    manual_warmth: float = 0.0  # -1.0 cool -> +1.0 warm


@dataclass
class PipelineStats:
    rotation_angle: float
    used_perspective: bool
    brightness_scale: float
    highlight_ratio: float


def _auto_rotate_by_lines(img: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=max(50, img.shape[1] // 12),
        maxLineGap=20,
    )

    if lines is None:
        return img, 0.0

    angles = []
    for l in lines[:, 0]:
        x1, y1, x2, y2 = l
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle < -45:
            angle += 90
        if angle > 45:
            angle -= 90
        if -20 <= angle <= 20:
            angles.append(angle)

    if not angles:
        return img, 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.25:
        return img, 0.0

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _auto_perspective_correct(img: np.ndarray) -> tuple[np.ndarray, bool]:
    h, w = img.shape[:2]
    area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 180)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: Optional[np.ndarray] = None
    best_score = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        quad = approx.reshape(4, 2)
        quad_area = cv2.contourArea(quad)
        if quad_area < 0.2 * area:
            continue

        x, y, cw, ch = cv2.boundingRect(quad)
        if cw == 0 or ch == 0:
            continue

        rectangularity = float(quad_area / (cw * ch))
        coverage = float(quad_area / area)
        score = 0.7 * coverage + 0.3 * rectangularity
        if score > best_score:
            best_score = score
            best_quad = quad.astype(np.float32)

    if best_quad is None or best_score < 0.3:
        return img, False

    rect = _order_quad_points(best_quad)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 50 or max_height < 50:
        return img, False

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    mat = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, mat, (max_width, max_height), flags=cv2.INTER_CUBIC)
    warped = cv2.resize(warped, (w, h), interpolation=cv2.INTER_CUBIC)
    return warped, True



def _reduce_overexposure(img: np.ndarray, cfg: EnhanceConfig) -> tuple[np.ndarray, float, float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]

    mean_luma = float(np.mean(v) / 255.0)
    highlight_ratio = float(np.mean(v >= cfg.highlight_threshold))

    scale = 1.0
    if mean_luma > cfg.target_luma:
        drop = min(cfg.max_brightness_drop, mean_luma - cfg.target_luma)
        scale = 1.0 - drop

    if highlight_ratio > 0.12:
        scale *= 0.90

    v = np.clip(v * scale, 0, 255)
    hsv[:, :, 2] = v
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, float(scale), highlight_ratio


def _gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    bgr = img.astype(np.float32)
    b, g, r = cv2.split(bgr)

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6

    b *= mean_gray / (mean_b + 1e-6)
    g *= mean_gray / (mean_g + 1e-6)
    r *= mean_gray / (mean_r + 1e-6)

    merged = cv2.merge((b, g, r))
    return np.clip(merged, 0, 255).astype(np.uint8)


def _clahe_l_channel(img: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l = clahe.apply(l)
    out = cv2.merge((l, a, b))
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _apply_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_manual_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    if abs(ev) < 1e-6:
        return img
    gain = 2.0 ** ev
    out = np.clip(img.astype(np.float32) * gain, 0, 255)
    return out.astype(np.uint8)


def _apply_warmth(img: np.ndarray, warmth: float) -> np.ndarray:
    if abs(warmth) < 1e-6:
        return img

    warmth = float(np.clip(warmth, -1.0, 1.0))
    b, g, r = cv2.split(img.astype(np.float32))

    # Warmth shifts red-blue balance while preserving green channel stability.
    r *= 1.0 + 0.15 * warmth
    b *= 1.0 - 0.15 * warmth

    merged = cv2.merge((b, g, r))
    return np.clip(merged, 0, 255).astype(np.uint8)


def enhance_image(img: np.ndarray, cfg: EnhanceConfig) -> tuple[np.ndarray, PipelineStats]:
    out = img.copy()
    rotation_angle = 0.0
    used_perspective = False

    if cfg.enable_straighten:
        out, rotation_angle = _auto_rotate_by_lines(out)

    if cfg.enable_perspective:
        out, used_perspective = _auto_perspective_correct(out)

    brightness_scale = 1.0
    highlight_ratio = 0.0
    if cfg.auto_brightness:
        out, brightness_scale, highlight_ratio = _reduce_overexposure(out, cfg)

    # --- Color correction (disabled) ---
    # if cfg.auto_white_balance:
    #     out = _gray_world_white_balance(out)

    # if cfg.use_clahe:
    #     out = _clahe_l_channel(out, cfg.clahe_clip_limit, cfg.clahe_grid_size)

    # if abs(cfg.saturation_boost - 1.0) > 1e-6:
    #     out = _apply_saturation(out, cfg.saturation_boost)

    # out = _apply_manual_exposure(out, cfg.manual_exposure)

    # if abs(cfg.manual_saturation - 1.0) > 1e-6:
    #     out = _apply_saturation(out, cfg.manual_saturation)

    # out = _apply_warmth(out, cfg.manual_warmth)
    # --- End color correction ---

    stats = PipelineStats(
        rotation_angle=rotation_angle,
        used_perspective=used_perspective,
        brightness_scale=brightness_scale,
        highlight_ratio=highlight_ratio,
    )
    return out, stats


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".jpg"
    ext = ".png" if suffix == ".png" else ".jpg"

    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]

    ok, encoded = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))

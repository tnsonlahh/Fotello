from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from src.enhancer import EnhanceConfig, enhance_image, read_image, write_image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTS:
            yield input_path
        return

    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def build_output_path(src: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / f"{src.stem}_enhanced{src.suffix}"
    rel = src.relative_to(input_root)
    return output_root / rel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Image Enhancement for architecture/real-estate photos (no training required)."
    )
    parser.add_argument("--input", required=True, help="Input image path or folder path")
    parser.add_argument("--output", required=True, help="Output folder path")

    parser.add_argument("--no-straighten", action="store_true", help="Disable auto straightening")
    parser.add_argument("--no-perspective", action="store_true", help="Disable auto perspective correction")


    parser.add_argument("--no-auto-brightness", action="store_true", help="Disable overexposure reduction")
    parser.add_argument("--target-luma", type=float, default=0.52, help="Target brightness for auto adjustment")

    parser.add_argument("--no-auto-wb", action="store_true", help="Disable auto white balance")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE local contrast")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE grid size")
    parser.add_argument("--saturation-boost", type=float, default=1.08, help="Auto saturation boost")

    parser.add_argument("--manual-exposure", type=float, default=0.0, help="Manual EV offset, -1.0..1.0")
    parser.add_argument("--manual-saturation", type=float, default=1.0, help="Manual saturation, 0.5..1.5")
    parser.add_argument("--manual-warmth", type=float, default=0.0, help="Manual warmth, -1.0..1.0")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    cfg = EnhanceConfig(
        enable_straighten=not args.no_straighten,
        enable_perspective=not args.no_perspective,
        auto_brightness=not args.no_auto_brightness,
        target_luma=max(0.2, min(0.8, args.target_luma)),
        auto_white_balance=not args.no_auto_wb,
        use_clahe=not args.no_clahe,
        clahe_clip_limit=max(0.5, args.clahe_clip),
        clahe_grid_size=max(4, args.clahe_grid),
        saturation_boost=max(0.5, min(1.8, args.saturation_boost)),
        manual_exposure=max(-1.0, min(1.0, args.manual_exposure)),
        manual_saturation=max(0.5, min(1.5, args.manual_saturation)),
        manual_warmth=max(-1.0, min(1.0, args.manual_warmth)),
    )

    images = list(collect_images(input_path))
    if not images:
        print("No valid images found.")
        return

    print(f"Found {len(images)} image(s).")
    success = 0

    for idx, src in enumerate(images, start=1):
        try:
            img = read_image(src)
            enhanced, stats = enhance_image(img, cfg)
            out_file = build_output_path(src, input_path, output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            write_image(out_file, enhanced)

            print(
                f"[{idx}/{len(images)}] OK {src.name} -> {out_file.name} | "
                f"rot={stats.rotation_angle:.2f}deg, "
                f"persp={stats.used_perspective}, "
                f"brightScale={stats.brightness_scale:.3f}, "
                f"highlight={stats.highlight_ratio:.2%}"
            )
            success += 1
        except Exception as exc:
            print(f"[{idx}/{len(images)}] FAIL {src}: {exc}")

    print(f"Done. {success}/{len(images)} image(s) processed successfully.")


if __name__ == "__main__":
    main()

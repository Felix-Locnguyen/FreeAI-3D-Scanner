"""Chạy baseline trên toàn bộ dữ liệu thử trong data/input.

- Thư mục ảnh : quality check -> COLMAP sparse -> 1 dòng runs.jsonl
- Video (.mp4): trích frame -> quality check -> COLMAP sparse -> 1 dòng

Chạy:
    python src/scripts/run_baseline.py
    python src/scripts/run_baseline.py --dense    # ~10 phút/bộ
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # để import được src khi chạy trực tiếp

import pycolmap

from src.preprocessing.quality_checker import check_images
from src.preprocessing.video_processor import extract_frames
from src.reconstruction.colmap_pipeline import run_colmap
from src.utils.metrics import count_status, make_record, save_run, summary_table

INPUT_DIR = ROOT / "data/input"
FRAMES_DIR = ROOT / "data/frames"
OUTPUT_DIR = ROOT / "data/output"


def run_one(name, image_dir, work_dir, dense=False):
    """Quality check + COLMAP trên 1 bộ, ghi 1 dòng metrics."""
    results = check_images(image_dir)
    quality = count_status(results)
    print(f"[quality] {name}: " + ", ".join(f"{k}={v}" for k, v in quality.items() if v))

    stats = run_colmap(image_dir, work_dir, dense=dense, log=lambda m: print(m, flush=True))

    if stats.get("dense_ply"):
        ply = Path(stats["dense_ply"])
    else:  # sparse: xuất PLY để có file xem được + số liệu dung lượng như app
        ply = Path(work_dir) / "sparse.ply"
        pycolmap.Reconstruction(stats["sparse_dir"]).export_PLY(str(ply))

    record = make_record(stats, quality, source=name, ply=ply)
    save_run(record, OUTPUT_DIR / "runs.jsonl")
    return record


def main():
    dense = "--dense" in sys.argv
    datasets = []

    for path in sorted(INPUT_DIR.iterdir()):
        if path.is_dir():
            if path.name.startswith("upload_"):    # ảnh người dùng upload từ app, không phải bộ test
                continue
            datasets.append((path.name, path))                  # bộ ảnh
        elif path.suffix.lower() in {".mp4", ".mov", ".avi"}:
            frames = FRAMES_DIR / f"{path.stem}_baseline"
            extract_frames(path, frames)                        # video -> frame
            datasets.append((path.stem, frames))

    if not datasets:
        print("Khong tim thay du lieu trong data/input")
        return 1

    for name, image_dir in datasets:
        print(f"\n=== {name} ===")
        try:
            run_one(name, image_dir, OUTPUT_DIR / f"baseline_{name}", dense)
        except Exception as exc:
            print(f"LOI {name}: {exc}")

    print("\n=== BANG BASELINE ===")
    print(summary_table(OUTPUT_DIR / "runs.jsonl").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

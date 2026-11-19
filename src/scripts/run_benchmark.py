"""Chạy 4 cấu hình benchmark trên toàn bộ dữ liệu thử (Giai đoạn 2.3).

4 cấu hình (theo project_tasks.md):
    sift            SIFT — mặc định của COLMAP, nhanh
    sift+mask       SIFT + loại nền SAM 2
    lightglue       SuperPoint + LightGlue — mạng nơ-ron
    lightglue+mask  SuperPoint + LightGlue + loại nền SAM 2

Kết quả: mỗi (bộ dữ liệu, cấu hình) = 1 dòng JSON -> benchmarks/results.jsonl

Chạy:
    python src/scripts/run_benchmark.py                 # chạy nốt phần chưa chạy
    python src/scripts/run_benchmark.py --only test_video
    python src/scripts/run_benchmark.py --force         # chạy lại tất cả
    python src/scripts/run_benchmark.py --render        # in bảng, không chạy lại
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # để import được src khi chạy trực tiếp

from src.preprocessing.video_processor import extract_frames
from src.reconstruction.colmap_pipeline import run_colmap

INPUT_DIR = ROOT / "data/input"
FRAMES_DIR = ROOT / "data/frames"
OUTPUT_DIR = ROOT / "data/output"
RESULTS = ROOT / "benchmarks/results.jsonl"

# (tên hiển thị, matcher, có dùng mask SAM 2 hay không)
CONFIGS = [
    ("sift", "sift", False),
    ("sift+mask", "sift", True),
    ("lightglue", "superpoint", False),
    ("lightglue+mask", "superpoint", True),
]


# ---------------------------------------------------------------- dữ liệu thử

def find_datasets():
    """Liệt kê bộ dữ liệu thử: thư mục ảnh trong data/input + video đã trích frame.

    Bỏ qua thư mục `upload_*` (ảnh người dùng tải lên app, không phải bộ test).
    """
    datasets = []
    for path in sorted(INPUT_DIR.iterdir()):
        if path.is_dir() and not path.name.startswith("upload_"):
            datasets.append((path.name, path))
        elif path.suffix.lower() in {".mp4", ".mov", ".avi"}:
            frames = FRAMES_DIR / path.stem
            if not frames.is_dir():                # chưa trích frame -> trích 1 lần
                extract_frames(path, frames)
            datasets.append((path.stem, frames))
    return datasets


# ------------------------------------------------------------ đọc / ghi kết quả

def load_results():
    """Đọc toàn bộ kết quả benchmark đã có (dạng JSONL)."""
    if not RESULTS.exists():
        return []
    return [
        json.loads(line)
        for line in RESULTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_record(record):
    """Ghi thêm 1 dòng kết quả (append) vào file JSONL."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------- in bảng

def fmt(value, default="—"):
    """Định dạng số kiểu Việt Nam: 7.625 điểm, 0,306 px, 210,5 giây.

    Số nguyên thêm dấu chấm phân cách hàng nghìn; số thập phân đổi dấu phẩy.
    """
    if value is None:
        return default
    if isinstance(value, float):
        return str(value).replace(".", ",")
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    return str(value)


def render_table(records):
    """Gộp record thành bảng Markdown.

    Mỗi (bộ, cấu hình) chỉ lấy bản ghi mới nhất — tránh trùng dòng khi chạy
    lại bằng --force.
    """
    latest = {}
    for r in records:
        latest[(r.get("dataset"), r.get("config"))] = r

    rows = [
        "| Bộ dữ liệu | Cấu hình | Đăng ký | Điểm 3D | Reproj (px) | Inlier | "
        "Mask (s) | Tổng (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in latest.values():
        timings = r.get("timings", {}) or {}
        rows.append(
            f"| {r.get('dataset')} | {r.get('config')} "
            f"| {fmt(r.get('images_registered'))}/{fmt(r.get('images_input'))} "
            f"| {fmt(r.get('points3d'))} "
            f"| {fmt(r.get('mean_reprojection_error_px'))} "
            f"| {fmt(r.get('inlier_ratio'))} "
            f"| {fmt(timings.get('mask'))} "
            f"| {fmt(timings.get('total'))} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------- main

def run_single(dataset, label):
    """Chạy đúng 1 cấu hình rồi ghi 1 dòng kết quả. Gọi từ process con.

    Tách mỗi cấu hình ra 1 process riêng vì PyTorch giữ lại bộ nhớ CUDA sau
    khi chạy — gộp 8 cấu hình vào 1 process sẽ tích lũy dần tới OOM, và 1 lần
    OOM làm hỏng CUDA context nên các lần chạy sau hỏng theo (bài học 2.3).
    """
    image_dir = dict(find_datasets()).get(dataset)
    config = next((c for c in CONFIGS if c[0] == label), None)
    if image_dir is None or config is None:
        print(f"Khong tim thay bo du lieu '{dataset}' hoac cau hinh '{label}'")
        return 1

    _, matcher, mask = config
    # Mỗi cấu hình 1 thư mục riêng để không ghi đè kết quả của nhau
    work_dir = OUTPUT_DIR / "bench" / f"{dataset}_{label.replace('+', '_')}"

    print(f"\n=== {dataset} / {label} ===", flush=True)
    t = time.time()
    try:
        stats = run_colmap(image_dir, work_dir, matcher=matcher, mask=mask,
                           log=lambda m: print(m, flush=True))
    except Exception as exc:            # 1 cấu hình lỗi không chặn các cấu hình sau
        print(f"LOI {dataset}/{label}: {exc}", flush=True)
        return 1

    record = {"dataset": dataset, "config": label,
              "wall_s": round(time.time() - t, 1)}
    record.update(stats)
    save_record(record)
    return 0


def main():
    args = sys.argv[1:]

    # --render: chỉ in bảng từ dữ liệu đã có, không chạy lại pipeline
    if "--render" in args:
        print(render_table(load_results()))
        return 0

    # --run <bo> <cau hinh>: chế độ nội bộ, chạy trong 1 process con
    if "--run" in args:
        i = args.index("--run")
        return run_single(args[i + 1], args[i + 2])

    force = "--force" in args
    only = args[args.index("--only") + 1] if "--only" in args else None

    records = load_results()
    done = {(r.get("dataset"), r.get("config")) for r in records}

    datasets = find_datasets()
    if only:
        datasets = [(n, d) for n, d in datasets if n == only]
    if not datasets:
        print("Khong tim thay du lieu thu trong data/input")
        return 1

    for dataset, _ in datasets:
        for label, _, _ in CONFIGS:
            if not force and (dataset, label) in done:
                print(f"[bo qua] {dataset} / {label} (da chay truoc do)")
                continue
            # 1 cấu hình = 1 process con: mỗi lần chạy bắt đầu với CUDA context sạch
            code = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--run", dataset, label],
                cwd=ROOT,
            ).returncode
            if code != 0:
                print(f"LOI {dataset}/{label}: process con exit {code}", flush=True)

    print("\n=== BENCHMARK ===")
    print(render_table(load_results()))   # đọc lại file vì process con đã ghi thêm
    return 0


if __name__ == "__main__":
    sys.exit(main())

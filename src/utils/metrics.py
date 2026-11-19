"""
Nơi lưu số liệu mỗi lần chạy ra kết quả (JSON) và in bảng baseline.
- JSONL (JSON Lines) = Mỗi dòng là 1 JSON hoàn chỉnh: 
  + Hỗ trợ ghi nối tiếp (append) cực kỳ nhanh mà không cần định nghĩa schema phức tạp.
  + Có thể mở bằng trình soạn thảo văn bản thông thường để đọc trực tiếp.
- Lệnh chạy:
    python src/utils/metrics.py
    python src/utils/metrics.py --path data/output/runs.jsonl
"""
from datetime import datetime
import json
from pathlib import Path
import sys

# Đường dẫn mặc định lưu trữ lịch sử các lần chạy (dạng file JSONL)
DEFAULT_RUNS_PATH = Path("data/output/runs.jsonl")

# Các trạng thái chuẩn của ảnh sau khi qua bộ lọc chất lượng (quality_checker)
STATUSES = ["Good", "Blurry", "Underexposed", "Overexposed", "Duplicate"]


# Các cột dữ liệu sẽ được hiển thị trên bảng baseline tổng hợp (đúng theo thứ tự ưu tiên)
COLUMNS = [
    "run_id",
    "timestamp",
    "source",
    "images_input",
    "matcher",
    "mask",
    "images_registered",
    "points3d",
    "inlier_ratio",
    "mean_reprojection_error_px",
    "q_good",
    "q_blurry",
    "q_underexposed",
    "q_overexposed",
    "q_duplicate",
    "dense",
    "ply_mb",
    "total_s",
]


def count_status(results):
    """Đếm tổng số lượng ảnh phân theo từng trạng thái chất lượng.
    
    Ví dụ trả về: {'Good': 46, 'Duplicate': 1, 'Blurry': 2, ...}
    """
    # Khởi tạo bộ đếm với giá trị 0 cho tất cả trạng thái mặc định
    counts = {s: 0 for s in STATUSES}
    for r in results:
        # Cộng dồn số lượng dựa trên trạng thái (status) của từng ảnh trong kết quả
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def make_record(stats, quality=None, source="images", run_id=None, ply=None):
    """Gói gọn thông số thống kê kỹ thuật (từ quá trình chạy colmap) và kết quả kiểm tra
    chất lượng ảnh thành 1 dictionary phẳng (1 dòng dữ liệu) để dễ dàng đưa vào bảng.
    """
    ts = datetime.now()
    quality = quality or {}
    ply_path = Path(ply) if ply else None
    
    return {
        # Định danh phiên chạy và thời gian thực hiện
        "run_id": run_id or ts.strftime("%Y%m%d-%H%M%S"),
        "timestamp": ts.isoformat(timespec="seconds"),
        "source": source,
        "image_dir": stats.get("image_dir"),
        
        # Thống kê số lượng ảnh và điểm 3D từ SfM
        "images_input": stats.get("images_input"),
        # Matcher dùng để ghép cặp: "sift" (mặc định) hoặc "superpoint"/"aliked"
        "matcher": stats.get("matcher", "sift"),
        # Có tô nền đen bằng SAM 2 trước khi trích đặc trưng không (Giai đoạn 2.2)
        "mask": bool(stats.get("mask")),
        "images_registered": stats.get("images_registered"),
        "points3d": stats.get("points3d"),
        # Tỷ lệ match còn lại sau RANSAC (cao = match chất lượng tốt)
        "inlier_ratio": stats.get("inlier_ratio"),
        "mean_reprojection_error_px": stats.get("mean_reprojection_error_px"),
        
        # Thống kê phân loại chất lượng ảnh đầu vào
        "q_good": quality.get("Good", 0),
        "q_blurry": quality.get("Blurry", 0),
        "q_underexposed": quality.get("Underexposed", 0),
        "q_overexposed": quality.get("Overexposed", 0),
        "q_duplicate": quality.get("Duplicate", 0),
        
        # Thông tin mô hình Dense 3D & kích thước file PLY xuất ra
        "dense": bool(stats.get("dense_ply")),
        "ply": str(ply_path) if ply_path else stats.get("dense_ply"),
        "ply_mb": (
            round(ply_path.stat().st_size / 1e6, 2)
            if ply_path and ply_path.exists()
            else stats.get("dense_ply_mb")
        ),
        
        # Tổng thời gian thực thi toàn bộ pipeline (tính bằng giây)
        "total_s": stats.get("timings", {}).get("total"),
    }


def save_run(record, path=DEFAULT_RUNS_PATH):
    """Ghi nối tiếp (Append) 1 record mới vào file JSONL.
    
    Tự động tạo thư mục chứa file nếu chưa tồn tại. Trả lại record vừa ghi để tiện xử lý tiếp.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Mở file ở chế độ append ("a") để thêm dòng mới vào cuối file mà không ghi đè dữ liệu cũ
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_runs(path=DEFAULT_RUNS_PATH):
    """Đọc toàn bộ file JSONL và chuyển đổi thành danh sách dictionary (list[dict]).
    
    Lưu ý: Nếu phát hiện dòng nào bị hỏng định dạng JSON, hàm sẽ văng lỗi ngay lập tức 
    thay vì nuốt lỗi im lặng, giúp phát hiện sớm dữ liệu lỗi.
    """
    path = Path(path)
    if not path.exists():
        return []
        
    runs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():  # Bỏ qua các dòng trống
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} dòng {i} bị lỗi định dạng JSON: {exc}") from exc
    return runs


def summary_table(path=DEFAULT_RUNS_PATH):
    """Đọc lịch sử chạy từ file JSONL và chuyển hóa thành pandas DataFrame.
    
    Chỉ giữ lại các cột nằm trong danh sách `COLUMNS` (các cột nào thiếu sẽ tự động được bỏ qua).
    """
    import pandas as pd

    df = pd.DataFrame(load_runs(path))
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
        
    # Lọc và sắp xếp DataFrame chỉ lấy các cột có ý nghĩa theo định nghĩa COLUMNS
    return df[[c for c in COLUMNS if c in df.columns]]


def main():
    """Hàm chạy chính khi gọi trực tiếp từ dòng lệnh (CLI).
    
    Nhiệm vụ: Đọc file lưu lịch sử chạy, in ra màn hình bảng tổng hợp baseline hoàn chỉnh cho người dùng xem.
    """
    import pandas as pd

    args = sys.argv[1:]
    # Cho phép tùy chỉnh đường dẫn file log thông qua tham số dòng lệnh `--path`
    if "--path" in args and args.index("--path") + 1 < len(args):
        path = Path(args[args.index("--path") + 1])
    else:
        path = DEFAULT_RUNS_PATH
    
    df = summary_table(path)
    print(f"File: {path}  ({len(df)} lần chạy)")
    
    if df.empty:
        print("Chưa có dữ liệu lịch sử. Hãy chạy app.py hoặc src/scripts/run_baseline.py trước.")
        return 1
        
    # Cấu hình hiển thị bảng đẹp mắt trên terminal (mở rộng chiều rộng hiển thị cột)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(df.to_string(index=False))
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

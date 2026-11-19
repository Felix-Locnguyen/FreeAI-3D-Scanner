"""
Pipeline COLMAP: sparse (SfM) + tùy chọn dense (MVS).

Chạy:
    python src/reconstruction/colmap_pipeline.py data/input/sample_temple
    python src/reconstruction/colmap_pipeline.py data/input/sample_temple --dense

Kết quả: <work_dir>/stats.json, <work_dir>/sparse, <work_dir>/mvs/dense.ply
(work_dir mặc định = data/output/<tên_thư_mục_ảnh>)

Ghi chú nhanh:
- SIFT (Scale-Invariant Feature Transform): trích điểm đặc trưng bất biến với góc nhìn,
  tỷ lệ và ánh sáng.
- SfM (Structure from Motion): từ các điểm khớp giữa các ảnh 2D -> tái tạo cấu trúc 3D
  thưa (sparse point cloud) + vị trí/hướng camera.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import pycolmap #Thư viện tính toán tái tạo cấu trúc 3D (SfM) và xác định vị trí camera từ ảnh chụp 2D.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ai.feature_matching.matching_model import db_match_stats, run_deep_matching


# các loại đuôi ảnh chấp nhận 
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def count_images(image_dir):
    "Trả về danh sách ảnh hợp lệ ( đã được sorted)"
    return sorted(pic.name for pic in Path(image_dir).iterdir() if pic.suffix.lower() in IMAGE_EXTS)


def run_colmap(image_dir, work_dir, dense=False, matcher="sift", mask=False,
               mask_point=None, log=print):
    """Chạy COLMAP end-to-end, trả về dict số liệu.

    matcher: "sift" (COLMAP tự làm) | "superpoint" | "aliked" (mạng nơ-ron, xem matching_model.py)
    mask: True -> tô nền đen bằng SAM 2 trước khi trích đặc trưng (xem sam2_model.py)
    mask_point: (u, v) tỉ lệ 0–1 nếu người dùng click chọn vật thể; None = tự động
    log: hàm nhận 1 chuỗi (mặc định print; truyền hàm khác để đẩy lên Gradio).
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)

    # --- Kiểm tra đầu vào ---
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Khong thay thu muc anh: {image_dir}")
    images = count_images(image_dir)
    if len(images) < 3:
        raise ValueError(f"Can it nhat 3 anh, dang co {len(images)}")

    # clean workspace để chạy, tránh đọc nhầm kết quả
    sparse_dir = work_dir / "sparse"      # Thư mục lưu kết quả tái tạo cấu trúc thưa (Sparse Reconstruction / SfM model)
    mvs_dir = work_dir / "mvs"            # Thư mục chứa dữ liệu và kết quả cho công đoạn tái tạo mật độ cao (Multi-View Stereo)
    database_path = work_dir / "database.db" # Đường dẫn tới file cơ sở dữ liệu SQLite lưu thông tin đặc trưng (features) và cặp ảnh khớp nhau (matches)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Dọn workspace để chạy sạch, tránh đọc nhầm kết quả cũ
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    if dense and mvs_dir.exists():   # chỉ xóa MVS khi thực sự chạy lại dense
        shutil.rmtree(mvs_dir)
    if database_path.exists():
        database_path.unlink()

    stats = {
        "image_dir": str(image_dir),
        "images_input": len(images),
        "matcher": matcher,
        "mask": bool(mask),
        "timings": {},
    }

    # --- 0 (tùy chọn): Loại nền bằng SAM 2, TRƯỚC khi trích đặc trưng ---
    # import ở đây chứ không ở đầu file: người không dùng mask sẽ không cần cài SAM 2
    if mask:
        from src.ai.segmentation.sam2_model import apply_mask

        mask_stats = apply_mask(image_dir, work_dir / "images_masked",
                                point=mask_point, log=log)
        stats.update(mask_stats)
        image_dir = work_dir / "images_masked"    # các bước sau đều đọc bản đã mask

    # 1 + 2: Trích điểm đặc trưng và ghép cặp ảnh
    if matcher == "sift":
        # --- SIFT: để COLMAP tự làm (đường cũ, giữ nguyên) ---
        t = time.time()
        log("[1/3] Trich xuat dac trung SIFT...")
        pycolmap.extract_features(database_path, image_dir)
        stats["timings"]["extract"] = round(time.time() - t, 1)

        t = time.time()
        log("[2/3] Ghep cap anh (exhaustive matching)...")
        pycolmap.match_exhaustive(database_path)
        stats["timings"]["match"] = round(time.time() - t, 1)
        stats.update(db_match_stats(database_path))
    else:
        # --- Matcher học sâu: SuperPoint/ALIKED + LightGlue (matching_model.py) ---
        stats.update(run_deep_matching(image_dir, database_path,
                                       extractor=matcher, log=log))

    # Ghi thời gian mask SAU bước matching: nhánh deep trả về dict "timings" riêng,
    # ghi trước đó sẽ bị `stats.update(...)` ghi đè mất -> tổng thời gian sai.
    if mask:
        stats["timings"]["mask"] = stats["mask_s"]

    # --- 3. Sparse reconstruction (SfM) ---
    t = time.time()
    log("[3/3] Tao mo hinh 3D sparse...")
    maps = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir)
    if not maps:
        raise RuntimeError("Khong tao duoc mo hinh 3D (thieu overlap hoac anh qua mo?)")

    # Chọn model (reconstruction) tốt nhất: model có nhiều điểm 3D nhất.
    # LƯU Ý: num_points3D() là PHƯƠNG THỨC, không phải property.
    # (incremental_mapping có thể trả nhiều model nếu ảnh bị tách thành nhiều cụm)
    rec = max(maps.values(), key=lambda r: r.num_points3D())

    # Ghi model tốt nhất ra sparse/ (định dạng COLMAP, dùng cho bước dense sau)
    rec.write(sparse_dir)

    # Thống kê thời gian và kết quả của quá trình Sparse Reconstruction
    stats["timings"]["sparse"] = round(time.time() - t, 1)
    stats["images_registered"] = len(rec.images)  # số ảnh đã được định vị vào model
    stats["points3d"] = rec.num_points3D()        # số điểm 3D thưa tái tạo được
    stats["mean_reprojection_error_px"] = round(rec.compute_mean_reprojection_error(), 3)
    stats["sparse_dir"] = str(sparse_dir)

    # In log tổng kết kết quả đăng ký ảnh và chất lượng mô hình
    log(
        f"  -> đăng ký: {stats['images_registered']}/{stats['images_input']} ảnh, "
        f"{stats['points3d']} điểm 3D, reproj {stats['mean_reprojection_error_px']} px"
    )

    # --- 4. Dense (tùy chọn, ~10 phút cho 47 ảnh trên 4GB VRAM) ---
    if dense:
        t = time.time()
        log("[dense] undistort -> patch-match stereo -> fusion...")
        pycolmap.undistort_images(mvs_dir, sparse_dir, image_dir)
        pycolmap.patch_match_stereo(mvs_dir)
        ply_path = mvs_dir / "dense.ply"
        # BẮT BUỘC output_type="ply": mặc định "bin" sẽ cố ghi thư mục model vào dense.ply
        pycolmap.stereo_fusion(ply_path, mvs_dir, output_type="ply")

        stats["timings"]["dense"] = round(time.time() - t, 1)
        stats["dense_ply"] = str(ply_path)
        stats["dense_ply_mb"] = round(ply_path.stat().st_size / 1e6, 1)
        log(f"  -> {ply_path} ({stats['dense_ply_mb']} MB)")

    stats["success"] = True
    stats["timings"]["total"] = round(sum(stats["timings"].values()), 1)
    (work_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main():
    # --matcher superpoint: lấy giá trị rồi loại khỏi argv để không bị coi là đường dẫn ảnh
    matcher = "sift"
    if "--matcher" in sys.argv:
        i = sys.argv.index("--matcher")
        matcher = sys.argv[i + 1]
        del sys.argv[i:i + 2]

    # --mask: bật loại nền SAM 2; --point u,v: chọn vật thể bằng 1 điểm (tỉ lệ 0–1)
    mask = "--mask" in sys.argv
    mask_point = None
    if "--point" in sys.argv:
        i = sys.argv.index("--point")
        mask_point = tuple(float(v) for v in sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]

    # Lấy tham số không bắt đầu bằng "-" làm đường dẫn (cho phép gõ --dense ở bất kỳ đâu)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("Cach dung: python src/reconstruction/colmap_pipeline.py <thu_muc_anh> "
              "[--dense] [--matcher superpoint|aliked] [--mask] [--point 0.5,0.5]")
        return 2

    image_dir = Path(args[0])
    dense = "--dense" in sys.argv
    work_dir = Path("data/output") / image_dir.name

    try:
        # flush=True: COLMAP ghi log ra stderr nên nếu không flush, log của ta sẽ bị đẩy xuống cuối
        stats = run_colmap(image_dir, work_dir, dense=dense, matcher=matcher,
                           mask=mask, mask_point=mask_point,
                           log=lambda m: print(m, flush=True))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"LOI: {exc}")
        return 1

    print("\n=== TOM TAT ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
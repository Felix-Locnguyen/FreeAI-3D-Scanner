"""
Smoke test: SuperPoint + LightGlue -> point cloud 3D thưa bằng COLMAP.

Thay vì để COLMAP tự trích SIFT, ta tự trích điểm đặc trưng bằng PyTorch
(SuperPoint) và tự ghép cặp ảnh (LightGlue), rồi GHI KẾT QUẢ VÀO DATABASE của COLMAP.
Sau đó gọi COLMAP như bình thường để tái tạo 3D.

Chạy:
    python src/scripts/smoke_lightglue.py data/input/sample_temple
    python src/scripts/smoke_lightglue.py data/input/sample_temple --limit 8
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_KEYPOINTS = 2048      # mỗi ảnh tối đa bao nhiêu điểm đặc trưng


def list_images(image_dir, limit=None):
    """Danh sách ảnh hợp lệ, đã sắp xếp theo tên."""
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return paths[:limit] if limit else paths


def load_models(device):
    """Tạo 2 mạng nơ-ron:
    - SuperPoint: tìm điểm đặc trưng (keypoint) trên mỗi ảnh.
    - LightGlue: ghép keypoint giữa 2 ảnh (nếu hai ảnh nhìn cùng vật thể).
    """
    extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def extract_all(extractor, paths, device):
    """Trích điểm đặc trưng cho từng ảnh. Trả về dict {tên_ảnh: feats}."""
    feats_all = {}
    for i, path in enumerate(paths, 1):
        with torch.inference_mode():                  # tắt đạo hàm -> nhanh, tốn RAM ít
            feats = extractor.extract(load_image(str(path)).to(device))
        feats_all[path.name] = feats
        print(f"  [{i}/{len(paths)}] {path.name}: {feats['keypoints'].shape[1]} điểm", flush=True)
    return feats_all


def match_all(matcher, paths, feats_all):
    """Ghép mọi cặp ảnh (exhaustive: chậm nhưng chắc, đúng như COLMAP mặc định).
    Trả về list[(tên_ảnh_0, tên_ảnh_1, matches (M, 2))].
    """
    results = []
    total = len(paths) * (len(paths) - 1) // 2
    done = 0
    for i, path0 in enumerate(paths):
        for path1 in paths[i + 1:]:
            with torch.inference_mode():
                out = matcher({"image0": feats_all[path0.name],
                               "image1": feats_all[path1.name]})
            matches = rbd(out)["matches"].cpu().numpy().astype(np.uint32)   # (M, 2)
            done += 1
            if len(matches):
                results.append((path0.name, path1.name, matches))
            if done % 50 == 0:
                print(f"  {done}/{total} cặp đã ghép", flush=True)
    return results


def write_database(db_path, image_dir, feats_all, matches):
    """Ghi keypoints và các cặp match vào database COLMAP."""
    with pycolmap.Database.open(db_path):        # tạo file database rỗng trước
        pass

    # Nhập thông tin ảnh (tên, kích thước, camera) vào database
    pycolmap.import_images(db_path, image_dir, pycolmap.CameraMode.AUTO, list(feats_all))

    with pycolmap.Database.open(db_path) as db:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        for name, feats in feats_all.items():
            # SuperPoint trả tâm pixel 0; COLMAP định nghĩa tâm pixel 0.5 -> cộng 0.5
            db.write_keypoints(ids[name], feats["keypoints"][0].cpu().numpy() + 0.5)
        for name0, name1, m in matches:
            db.write_matches(ids[name0], ids[name1], m)


def db_stats(db_path):
    """Số liệu trong database: match thô và match còn lại sau RANSAC."""
    with pycolmap.Database.open(db_path) as db:
        raw = sum(len(m) for m in db.read_all_matches()[1])
        inliers = sum(db.read_two_view_geometry_num_inliers()[1])
    return {
        "raw_matches": raw,
        "inlier_matches": inliers,
        "inlier_ratio": round(inliers / raw, 3) if raw else None,
    }


def run_mapping(db_path, image_dir, sparse_dir):
    """COLMAP tái tạo 3D từ database. Trả về reconstruction, hoặc None nếu thất bại."""
    maps = pycolmap.incremental_mapping(db_path, image_dir, str(sparse_dir))
    if not maps:
        return None
    # COLMAP có thể trả nhiều model (các cụm ảnh tách nhau) -> chọn model nhiều điểm 3D nhất
    return max(maps.values(), key=lambda r: r.num_points3D())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("Chạy: python src/scripts/smoke_lightglue.py <thu_muc_anh> [--limit N]")
        return 2
    image_dir = Path(args[0])
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}", flush=True)

    paths = list_images(image_dir, limit)
    print(f"Ảnh đầu vào: {len(paths)} ({image_dir})", flush=True)
    if len(paths) < 3:
        print("LOI: can it nhat 3 anh.")
        return 1

    # Dọn kết quả cũ rồi tạo workspace mới
    work_dir = Path("data/output/smoke_lightglue")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    db_path, sparse_dir = work_dir / "database.db", work_dir / "sparse"

    stats = {"images_input": len(paths), "device": str(device), "timings": {}}
    t = time.time()

    # --- Bước 1: trích điểm đặc trưng ---
    extractor, matcher = load_models(device)
    print("[1/3] Trich diem dac trung SuperPoint...", flush=True)
    feats_all = extract_all(extractor, paths, device)
    stats["timings"]["extract"] = round(time.time() - t, 1)
    t = time.time()

    # --- Bước 2: ghép cặp ảnh ---
    print("[2/3] Ghep cap anh LightGlue...", flush=True)
    matches = match_all(matcher, paths, feats_all)
    stats["timings"]["match"] = round(time.time() - t, 1)
    t = time.time()

    # --- Bước 3: ghi database -> RANSAC -> tái tạo 3D ---
    print("[3/3] Ghi database + tao mo hinh 3D...", flush=True)
    write_database(db_path, image_dir, feats_all, matches)
    pycolmap.geometric_verification(str(db_path))     # RANSAC: loại match sai
    rec = run_mapping(db_path, image_dir, sparse_dir)
    stats["timings"]["sparse"] = round(time.time() - t, 1)

    if rec is None:
        print("LOI: khong tao duoc mo hinh 3D.")
        return 1

    stats["pairs_matched"] = len(matches)
    stats.update(db_stats(db_path))
    stats.update({
        "images_registered": len(rec.images),
        "points3d": rec.num_points3D(),
        "reprojection_error_px": round(rec.compute_mean_reprojection_error(), 4),
    })
    stats["timings"]["total"] = round(sum(stats["timings"].values()), 1)

    # Xuất point cloud để xem ngay trên MeshLab / Blender
    ply = work_dir / "sparse.ply"
    rec.export_PLY(str(ply))
    stats["ply_mb"] = round(ply.stat().st_size / 1e6, 2)

    (work_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n=== TOM TAT ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

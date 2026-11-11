"""Smoke test Giai đoạn 0: chạy COLMAP trên bộ ảnh mẫu.

Cách dùng:
    python src/scripts/smoke_colmap.py data/input/sample_temple
"""
import sys
import time
from pathlib import Path

import pycolmap


def main() -> int:
    # --- Nhận thư mục ảnh từ dòng lệnh ---
    if len(sys.argv) < 2:
        print("Cach dung: python src/scripts/smoke_colmap.py <thu_muc_anh>")
        return 2
    image_dir = Path(sys.argv[1]).resolve()
    out_dir = Path("data/output/smoke_test").resolve()

    # Đếm số ảnh
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Anh dau vao: {len(images)} ({image_dir})", flush=True)
    if len(images) < 3:
        print("LOI: can it nhat 3 anh")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    database_path = out_dir / "database.db"
    mvs_path = out_dir / "mvs"
    if database_path.exists():
        database_path.unlink()

    t0 = time.time()

    # --- 4 buoc co ban cua COLMAP ---
    print("\n[1/4] Trich xuat dac trung SIFT...", flush=True)
    pycolmap.extract_features(database_path, image_dir)

    print("[2/4] Ghep cap anh (exhaustive matching)...", flush=True)
    pycolmap.match_exhaustive(database_path)

    print("[3/4] Tao mo hinh 3D (sparse reconstruction)...", flush=True)
    maps = pycolmap.incremental_mapping(database_path, image_dir, out_dir)
    if not maps:
        print("LOI: khong tao duoc mo hinh")
        return 1
    # Luu y: num_points3D la PHUONG THUC (goi bang ()), khong phai property
    rec = max(maps.values(), key=lambda r: r.num_points3D())
    rec.write(out_dir)
    print(f"  -> Anh dang ky   : {len(rec.images)}/{len(images)}", flush=True)
    print(f"  -> Diem 3D sparse: {rec.num_points3D()}", flush=True)

    print("[4/4] Dense: undistort -> patch-match stereo (GPU) -> fusion...", flush=True)
    pycolmap.undistort_images(mvs_path, out_dir, image_dir)
    pycolmap.patch_match_stereo(mvs_path)
    ply_path = mvs_path / "dense.ply"
    # output_type="ply" bat buoc: mac dinh "bin" se thu ghi thu muc .bin vao dense.ply -> loi
    pycolmap.stereo_fusion(ply_path, mvs_path, output_type="ply")

    # --- Ket qua ---
    dt = time.time() - t0
    print("\n=== KET QUA ===", flush=True)
    print(rec.summary())
    print(f"Thoi gian tong: {dt:.1f}s")
    if ply_path.exists():
        size_mb = ply_path.stat().st_size / 1e6
        print(f"Dense point cloud: {ply_path} ({size_mb:.1f} MB)")
        print("SMOKE TEST: PASS")
        return 0
    print("SMOKE TEST: FAIL (khong co dense.ply)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Xử lý point cloud: lọc nhiễu, giảm điểm, ước lượng normal, cắt vật thể (Giai đoạn 3.1).

Chạy thử:
    python src/pointcloud/pointcloud_processor.py <duong_dan.ply>
    python src/pointcloud/pointcloud_processor.py <duong_dan.ply> --voxel 0.01 --normals
    python src/pointcloud/pointcloud_processor.py <duong_dan.ply> --mask \
        --sparse <work_dir>/sparse --mask-dir <work_dir>/images_masked

Cách có `images_masked/` (cần cho `--mask`):
    - Chạy pipeline với cờ `--mask` (SAM 2 tự ghi `<work_dir>/images_masked/`), hoặc
    - Chạy riêng SAM 2: `python src/ai/segmentation/sam2_model.py <thu_muc_anh>`
      -> kết quả ở `data/output/mask_test/images_masked/`

Đầu vào : PLY do `app.py` (`sparse.ply`) hoặc `colmap_pipeline.py --dense` (`mvs/dense.ply`) tạo.
Đầu ra  : PLY đã xử lý (mặc định `<ten_goc>_processed.ply`) + in số điểm trước/sau.

Thứ tự xử lý: cắt vật thể -> lọc nhiễu -> giảm điểm -> ước lượng normal.
(Cắt trước để điểm nền không làm sai phép đo nhiễu; normal làm sau cùng vì
 radius phụ thuộc vào độ thưa sau khi đã giảm điểm.)

Ghi chú nhanh:
- SOR (Statistical Outlier Removal): mỗi điểm so với K láng giềng gần nhất; điểm
  nào lệch trung bình nhiều lần so với trung vị -> loại. Chặn "chấm bay lơ lửng".
- Voxel downsampling: chia không gian thành lưới vuông cạnh voxel, mỗi ô giữ 1 điểm
  (trung bình) -> giảm số điểm mà giữ nguyên hình dạng.
- Normal (mặt phẳng tiếp tuyến tại mỗi điểm): BẮT BUỘC cho Poisson mesh (Giai đoạn 3.2).
- Cắt theo mask: chiếu điểm 3D vào từng ảnh đã biết pose, so với ảnh đã tô nền đen
  (`images_masked/` của SAM 2) -> bỏ điểm rơi trên nền.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image


# ------------------------------------------------------------------ đọc / ghi

def load_ply(path):
    """Đọc file PLY thành point cloud Open3D; báo lỗi nếu rỗng hoặc hỏng."""
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Point cloud rong hoac khong doc duoc: {path}")
    return pcd


def save_ply(pcd, path):
    """Ghi point cloud ra PLY (giữ nguyên màu và normal nếu có)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    return path


# ---------------------------------------------------- lọc nhiễu / giảm điểm

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0, log=print):
    """Lọc nhiễu thống kê (SOR): bỏ điểm lơ lửng nằm cách xa đám chính.

    nb_neighbors: số láng giềng gần nhất dùng để so sánh với mỗi điểm.
    std_ratio    : ngưỡng lọc — càng lớn thì càng ít điểm bị loại.
    """
    n_before = len(pcd.points)
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    pcd = pcd.select_by_index(idx)
    log(f"  [SOR nb={nb_neighbors}, std={std_ratio}] {n_before} -> "
        f"{len(pcd.points)} diem (loai {n_before - len(pcd.points)})")
    return pcd


def downsample(pcd, voxel_size=0.01, log=print):
    """Giảm điểm bằng voxel: mỗi ô vuông cạnh `voxel_size` giữ 1 điểm (trung bình)."""
    n_before = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    log(f"  [voxel {voxel_size}] {n_before} -> {len(pcd.points)} diem")
    return pcd


# -------------------------------------------------------------------- normal

def estimate_normals(pcd, radius=0.05, max_nn=30, log=print):
    """Ước lượng vector pháp tuyến tại mỗi điểm — bắt buộc cho Poisson (Giai đoạn 3.2).

    radius: bán kính tìm láng giềng; nên bằng khoảng 3× khoảng cách trung bình
            giữa 2 điểm (to quá -> chậm & mờ, nhỏ quá -> thiếu láng giềng).
    max_nn: số láng giềng tối đa dùng trong bán kính đó.
    """
    n = len(pcd.points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius,
                                                           max_nn=max_nn))
    log(f"  [normals] da uoc luong cho {n} diem (radius={radius}, max_nn={max_nn})")
    return pcd


# --------------------------------------------------------- cắt vùng vật thể

def crop_bbox(pcd, min_bound, max_bound, log=print):
    """Chỉ giữ điểm nằm trong hộp giới hạn (bounding box) 6 số: x0,y0,z0,x1,y1,z1."""
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,
                                               max_bound=max_bound)
    n_before = len(pcd.points)
    pcd = pcd.crop(box)
    log(f"  [bbox] {n_before} -> {len(pcd.points)} diem")
    return pcd


def crop_by_mask(pcd, sparse_dir, mask_dir, bg_threshold=16, min_ratio=0.5,
                 log=print):
    """Bỏ điểm nằm trên nền: chiếu điểm 3D vào từng ảnh rồi so với ảnh đã mask.

    sparse_dir : thư mục COLMAP `sparse/` — cung cấp pose và intrinsics của camera.
    mask_dir   : thư mục ảnh đã tô nền đen (vd. `images_masked/` của SAM 2),
                 tên file khớp tên ảnh gốc. Nền = màu 0 nên chỉ cần so độ sáng.
    bg_threshold: độ sáng <= ngưỡng này coi là NỀN (JPEG nén lại làm nền không
                 còn đúng 0, nên phải có ngưỡng chứ không so bằng 0).
    min_ratio  : giữ điểm nếu >= tỷ lệ này số ảnh thấy nó nằm trên VẬT THỂ.

    Trả về point cloud đã cắt (bản mới — không sửa bản gốc).
    """
    sparse_dir, mask_dir = Path(sparse_dir), Path(mask_dir)
    rec = pycolmap.Reconstruction(str(sparse_dir))

    # Nạp trước mask từng ảnh (đen = nền, sáng = vật thể)
    masks = {}
    for image in rec.images.values():
        path = mask_dir / image.name
        if path.exists():
            masks[image.image_id] = np.array(Image.open(path).convert("L"))
    if not masks:
        raise ValueError(f"Khong co anh mask nao trong {mask_dir} "
                         f"khop voi anh trong {sparse_dir}")

    xyz = np.asarray(pcd.points)                    # (N, 3)
    n_views = np.zeros(len(xyz), dtype=np.int64)    # số ảnh lọt vào khung hình
    n_object = np.zeros(len(xyz), dtype=np.int64)   # số ảnh thấy điểm trên vật thể

    log(f"  [mask] chieu {len(xyz)} diem vao {len(masks)} anh...")
    for image_id, gray in masks.items():
        image = rec.images[image_id]
        h, w = gray.shape
        for i, pt in enumerate(xyz):
            uv = image.project_point(pt)            # None nếu nằm ngoài khung hình
            if uv is None:
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            if not (0 <= u < w and 0 <= v < h):
                continue
            n_views[i] += 1
            if gray[v, u] > bg_threshold:
                n_object[i] += 1

    # Điểm chưa từng lọt vào khung hình nào -> không quyết định được, loại luôn
    ratio = np.divide(n_object, n_views,
                      out=np.zeros(len(xyz), dtype=float), where=n_views > 0)
    keep = (n_views > 0) & (ratio >= min_ratio)

    out = pcd.select_by_index(np.nonzero(keep)[0])
    log(f"  [mask] {len(xyz)} -> {len(out.points)} diem "
        f"(bo {len(xyz) - len(out.points)} diem nen, min_ratio={min_ratio})")
    return out


# ---------------------------------------------------------------------- main

def parse_args(argv=None):
    """Đọc đối số dòng lệnh."""
    p = argparse.ArgumentParser(
        description="Xu ly point cloud: loc nhau, giam diem, normal, cat vat the")
    p.add_argument("input", help="file PLY dau vao")
    p.add_argument("--out", help="file PLY dau ra (mac dinh: <ten_goc>_processed.ply)")
    p.add_argument("--voxel", type=float, default=0.0,
                   help="do lon voxel de giam diem; 0 = khong giam (mac dinh)")
    p.add_argument("--sor-neighbors", type=int, default=20,
                   help="loc nhau SOR - so lang gieng; 0 = tat (mac dinh 20)")
    p.add_argument("--sor-std", type=float, default=2.0,
                   help="nguong loc nhau SOR (mac dinh 2.0)")
    p.add_argument("--normals", action="store_true", help="uoc luong normal")
    p.add_argument("--normal-radius", type=float, default=0.05,
                   help="ban kinh uoc luong normal (mac dinh 0.05)")
    p.add_argument("--bbox", help="cat theo bbox: x0,y0,z0,x1,y1,z1")
    p.add_argument("--mask", action="store_true",
                   help="cat theo mask (can --sparse va --mask-dir)")
    p.add_argument("--sparse", help="thu muc COLMAP sparse (dung voi --mask)")
    p.add_argument("--mask-dir", help="thu muc anh da mask, vi du images_masked")
    p.add_argument("--min-ratio", type=float, default=0.5,
                   help="ty le anh thay vat the de giu diem (mac dinh 0.5)")
    return p.parse_args(argv)


def main(argv=None):
    """Chạy pipeline xử lý theo thứ tự: cắt -> lọc nhiễu -> giảm điểm -> normal."""
    args = parse_args(argv)
    src = Path(args.input)

    pcd = load_ply(src)
    print(f"[doc] {src} — {len(pcd.points)} diem")
    step = "doc"    # tên bước gần nhất — dùng để báo lỗi dễ hiểu nếu cloud bị rỗng

    # 1) Cắt vùng vật thể (làm trước để điểm nền không làm sai phép đo nhiễu)
    if args.bbox:
        bound = [float(x) for x in args.bbox.split(",")]
        if len(bound) != 6:
            raise ValueError("--bbox can 6 so: x0,y0,z0,x1,y1,z1")
        step = "cat bbox"
        pcd = crop_bbox(pcd, np.array(bound[:3]), np.array(bound[3:]))
    if args.mask:
        if not (args.sparse and args.mask_dir):
            raise ValueError("--mask can --sparse <thu_muc> va --mask-dir <thu_muc>")
        step = "cat mask"
        pcd = crop_by_mask(pcd, args.sparse, args.mask_dir,
                           min_ratio=args.min_ratio)

    # 2) Lọc nhiễu
    if args.sor_neighbors:
        step = "loc nhau SOR"
        pcd = remove_outliers(pcd, nb_neighbors=args.sor_neighbors,
                              std_ratio=args.sor_std)

    # 3) Giảm điểm
    if args.voxel > 0:
        step = "giam diem voxel"
        pcd = downsample(pcd, voxel_size=args.voxel)

    # 4) Normal (làm sau cùng vì radius phụ thuộc độ thưa sau khi giảm điểm)
    if args.normals:
        step = "uoc luong normal"
        pcd = estimate_normals(pcd, radius=args.normal_radius)

    # Nếu để rỗng thì Open3D sẽ KHÔNG ghi được file -> báo ngay, đừng để crash
    # ở bước `stat()` với lỗi FileNotFoundError khó hiểu.
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud rong sau buoc '{step}' — "
                         f"hay doi tham so (bbox / min-ratio / voxel / SOR) va chay lai")

    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_processed.ply")
    save_ply(pcd, out)
    print(f"[xuat] {out} — {len(pcd.points)} diem, "
          f"{out.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

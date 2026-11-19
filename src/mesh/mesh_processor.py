"""
Tạo mesh từ point cloud bằng Poisson surface reconstruction (Giai đoạn 3.2).

Chạy thử:
    python src/mesh/mesh_processor.py <duong_dan.ply>
    python src/mesh/mesh_processor.py <duong_dan.ply> --depth 6 --decimate 0.5
    python src/mesh/mesh_processor.py <duong_dan.ply> --out model.glb

Đầu vào : PLY đã có normal. Nếu chưa có thì chạy trước (Poisson bắt buộc có normal):
    python src/pointcloud/pointcloud_processor.py <duong_dan.ply> --normals
    (chọn --normal-radius xấp xỉ 3x khoang cach trung binh giua 2 diem)
Đầu ra  : mesh .ply / .obj / .glb. KHÔNG nhận .stl — Open3D 0.20 ghi .stl thất bại
          (trả False, file 0 byte) nên bị từ chối ngay từ đầu.

Thứ tự xử lý: Poisson -> bỏ vùng mật độ thấp -> giữ cụm lớn nhất
              -> decimate -> làm mượt -> kiểm tra chất lượng.

Ghi chú nhanh:
- Poisson: dựng mặt từ đám điểm CÓ normal, chia không gian bằng cây octree.
  `depth` càng lớn càng chi tiết nhưng càng chậm và càng dễ sinh cụm rác
  (thực nghiệm: depth=8 -> 73 cụm, depth=10 -> 82 cụm, depth=6 -> 1 cụm nhưng thô).
- densities: độ dày điểm ảnh chụp tại mỗi đỉnh. Poisson luôn trả về cả phần
  "bọt" xung quanh đám điểm -> phải cắt bỏ. Đây là thang log2 nên KHÔNG có ngưỡng
  tuyệt đối dùng chung, chỉ cắt theo percentile.
- `decimate()` và `smooth()` TRẢ VỀ mesh mới, KHÔNG sửa tại chỗ -> bắt buộc gán
  biến. Ngược lại `remove_*` thì sửa tại chỗ (API của Open3D không đồng nhất).
- Mesh từ ảnh chụp MỘT PHÍA thường KHÔNG watertight (mặt hở ở rìa vùng chụp)
  -> đó là bình thường, `check_mesh()` chỉ báo trung thực chứ không coi là lỗi.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# Đuôi file Open3D 0.20 ghi được (đã kiểm chứng). .stl bị loại có chủ đích.
ALLOWED_EXTS = {".ply", ".obj", ".glb"}


# ------------------------------------------------------------------ đọc / ghi

def load_pcd(path):
    """Đọc PLY thành point cloud; bắt buộc có normal vì Poisson không có normal."""
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Point cloud rong hoac khong doc duoc: {path}")
    if not pcd.has_normals():
        # Báo đúng lệnh cần chạy thay vì để Poisson ném RuntimeError khó hiểu
        raise ValueError(
            f"Point cloud chua co normal: {path}\n"
            "  Hay chay truoc: python src/pointcloud/pointcloud_processor.py "
            f"{path} --normals")
    return pcd


def save_mesh(mesh, path):
    """Ghi mesh ra file; chỉ nhận .ply/.obj/.glb và kiểm tra file có thật hay không.

    Bắt buộc kiểm tra kết quả: Open3D 0.20 ghi .stl trả về False kèm file 0 byte,
    nếu không kiểm tra thì bước `stat()` sau đó sẽ crash FileNotFoundError khó hiểu.
    """
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTS:
        raise ValueError(
            f"Dinh dang '{path.suffix}' khong ho tro — chi nhan "
            f"{'/'.join(sorted(ALLOWED_EXTS))} (ghi .stl bi loi trong Open3D 0.20)")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(path), mesh)
    if not ok or not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Khong ghi duoc mesh ra {path} — hay doi ten thu muc va chay lai")
    return path


# ------------------------------------------------------------------- Poisson

def poisson_reconstruct(pcd, depth=8, scale=1.1, point_weight=2.0, log=print):
    """Dựng mặt Poisson từ point cloud đã có normal.

    depth       : độ sâu cây octree — càng lớn càng chi tiết, càng chậm (mặc định 8).
    scale       : tỉ lệ hộp bao quanh đám điểm (mặc định 1.1).
    point_weight: mức "buộc" mesh đi qua các điểm — lớn thì sát điểm hơn (mặc định 2.0).

    Trả về (mesh, densities) — densities có đúng bằng số đỉnh của mesh.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, point_weight=point_weight)
    if len(mesh.triangles) == 0:
        raise ValueError("Poisson khong tao duoc tam giac nao — hay doi --depth")
    densities = np.asarray(densities)
    if len(densities) != len(mesh.vertices):
        raise ValueError("So densities khong khop so dinh — khong trim duoc")
    log(f"  [poisson depth={depth}, scale={scale}] {len(mesh.vertices)} dinh, "
        f"{len(mesh.triangles)} tam giac")
    return mesh, densities


# ------------------------------------------------------------ làm sạch mesh

def trim_density(mesh, densities, percentile=10, log=print):
    """Bỏ vùng mật độ thấp — phần "bọt" Poisson tạo ra xung quanh đám điểm.

    densities là thang log2 nên không có ngưỡng tuyệt đối, chỉ cắt theo percentile:
    10 = bỏ 10% đỉnh mỏng nhất.
    """
    if not 0 < percentile < 100:
        raise ValueError("--trim phai nam trong khoang (0, 100)")
    if len(densities) != len(mesh.vertices):
        raise ValueError("So densities khong khop so dinh — khong trim duoc")
    n_before = len(mesh.vertices)
    threshold = float(np.percentile(densities, percentile))
    mesh.remove_vertices_by_mask(densities < threshold)   # True = xóa (sửa tại chỗ)
    mesh.remove_unreferenced_vertices()
    log(f"  [trim density p{percentile:g}] {n_before} -> {len(mesh.vertices)} dinh")
    return mesh                     # trả về để main gán biến thống nhất


def keep_largest_component(mesh, log=print):
    """Giữ cụm tam giác lớn nhất, bỏ các thành phần rời (mảnh vụn bay lơ lửng)."""
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if len(counts) <= 1:
        log("  [cum] 1 cum — khong co thanh phan roi")
        return mesh
    keep = labels == np.argmax(counts)
    n_drop = int((~keep).sum())
    mesh.remove_triangles_by_mask(~keep)                 # True = xóa (sửa tại chỗ)
    mesh.remove_unreferenced_vertices()
    log(f"  [cum] {len(counts)} cum -> 1 (bo {n_drop} tam giac ruot)")
    return mesh


def decimate(mesh, ratio=0.3, log=print):
    """Giảm số tam giác bằng Quadric Error Metric — giữ lại `ratio` (0.3 = 30%).

    LƯU Ý: hàm này TRẢ VỀ mesh mới, KHÔNG sửa `mesh` tại chỗ.
    Gọi không gán biến thì kết quả bị mất âm thầm, không báo lỗi.
    """
    if not 0 < ratio < 1:
        raise ValueError("--decimate phai nam trong khoang (0, 1)")
    n_before = len(mesh.triangles)
    target = max(4, int(n_before * ratio))               # 4 = khối tứ diện, tránh rỗng
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    log(f"  [decimate giu {ratio:g}] {n_before} -> {len(mesh.triangles)} tam giac")
    return mesh


def smooth(mesh, iterations=3, log=print):
    """Làm mượt Taubin (2 Laplacian mỗi lần: `lambda_filter` rồi `mu`) — chỉ dịch
    vị trí đỉnh, không đổi số tam giác.

    LƯU Ý: cũng TRẢ VỀ mesh mới như `decimate()`.
    """
    n_before = len(mesh.vertices)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    log(f"  [taubin x{iterations}] {n_before} dinh (chi dich vi tri dinh)")
    return mesh


# --------------------------------------------------------------- kiểm tra

def count_open_edges(mesh):
    """Đếm cạnh mở — cạnh chỉ có 1 tam giác liền kề, tức là lỗ hổng trên bề mặt."""
    tri = np.asarray(mesh.triangles)
    if len(tri) == 0:
        return 0
    # Gom 3 cạnh của mỗi tam giác, rồi sắp xếp từng cặp để (a,b) và (b,a) trùng nhau
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def check_mesh(mesh, log=print):
    """In báo cáo chất lượng: watertight, manifold, tự giao, lỗ hổng, thành phần rời.

    Mesh từ ảnh chụp MỘT PHÍA thường KHÔNG watertight — đó là bình thường,
    không phải lỗi nên hàm này chỉ báo cáo, không raise.
    """
    watertight = mesh.is_watertight()
    try:
        volume = f"{mesh.get_volume():.4f}"
    except RuntimeError:
        volume = "khong tinh duoc (can watertight)"

    _, counts, _ = mesh.cluster_connected_triangles()
    log(f"[kiem tra mesh] {len(mesh.vertices)} dinh, {len(mesh.triangles)} tam giac")
    log(f"  watertight       : {'co' if watertight else 'khong'}"
        + ("" if watertight else " (binh thuong voi anh chup mot phia)"))
    log(f"  edge manifold    : {'co' if mesh.is_edge_manifold() else 'khong'}")
    log(f"  vertex manifold  : {'co' if mesh.is_vertex_manifold() else 'khong'}")
    log(f"  tu giao nhau     : {'co' if mesh.is_self_intersecting() else 'khong'}")
    log(f"  canh mo (lo hong): {count_open_edges(mesh)}")
    log(f"  thanh phan roi   : {len(np.asarray(counts))} cum")
    log(f"  Euler (V-E+F)    : {mesh.euler_poincare_characteristic()}")
    log(f"  the tich         : {volume}")


# ---------------------------------------------------------------------- main

def parse_args(argv=None):
    """Đọc đối số dòng lệnh."""
    p = argparse.ArgumentParser(
        description="Tao mesh tu point cloud bang Poisson surface reconstruction")
    p.add_argument("input", help="file PLY dau vao (da co normal)")
    p.add_argument("--out",
                   help="file dau ra .ply/.obj/.glb (mac dinh: <ten_goc>_mesh.ply)")
    p.add_argument("--depth", type=int, default=8,
                   help="do sau octree — lon = chi tiet hon, cham hon (mac dinh 8)")
    p.add_argument("--scale", type=float, default=1.1,
                   help="ti le hop bao quanh diem (mac dinh 1.1)")
    p.add_argument("--point-weight", type=float, default=2.0,
                   help="muc bu mesh di qua diem, lon = sat diem hon (mac dinh 2.0)")
    # Lưu ý: argparse format chuỗi help bằng `%` -> muốn in dấu % phải viết `%%`
    p.add_argument("--trim", type=float, default=10,
                   help="bo %% dinh co density thap nhat; 0 = tat (mac dinh 10)")
    p.add_argument("--decimate", type=float, default=0.3,
                   help="giu %% so tam giac sau decimate; 0 = tat (mac dinh 0.3)")
    p.add_argument("--smooth", type=int, default=3,
                   help="so lan lam muot Taubin; 0 = tat (mac dinh 3)")
    return p.parse_args(argv)


def main(argv=None):
    """Chạy pipeline: Poisson -> trim density -> cụm lớn nhất -> decimate -> mượt -> kiểm tra."""
    args = parse_args(argv)
    src = Path(args.input)

    pcd = load_pcd(src)
    print(f"[doc] {src} — {len(pcd.points)} diem")
    step = "poisson"    # tên bước gần nhất — dùng để báo lỗi nếu mesh bị rỗng

    # 1) Dựng mặt Poisson (trả về cả densities để bước sau cắt)
    mesh, densities = poisson_reconstruct(pcd, depth=args.depth, scale=args.scale,
                                          point_weight=args.point_weight)

    # 2) Bỏ phần "bọt" (density thấp) — làm trước khi tách cụm vì điểm rìa
    #    density thấp thường nằm ở cụm con.
    if args.trim > 0:
        step = "trim density"
        mesh = trim_density(mesh, densities, percentile=args.trim)

    # 3) Loại thành phần rời
    step = "loai thanh phan roi"
    mesh = keep_largest_component(mesh)

    # 4) Giảm số tam giác
    if args.decimate > 0:
        step = "decimate"
        mesh = decimate(mesh, ratio=args.decimate)

    # 5) Làm mượt — SAU decimate vì QEM để lại mặt thô
    if args.smooth > 0:
        step = "lam muot taubin"
        mesh = smooth(mesh, iterations=args.smooth)

    # Tương tự 3.1: mesh rỗng thì Open3D KHÔNG ghi được file -> báo ngay,
    # đừng để crash ở bước `stat()` với FileNotFoundError khó hiểu.
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh rong sau buoc '{step}' — "
                         f"hay doi tham so (--trim / --decimate / --depth) va chay lai")

    # 6) Báo cáo chất lượng trước khi ghi
    check_mesh(mesh)

    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_mesh.ply")
    save_mesh(mesh, out)
    print(f"[xuat] {out} — {len(mesh.triangles)} tam giac, "
          f"{out.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

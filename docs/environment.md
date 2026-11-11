# Môi trường phát triển (Giai đoạn 0)

Ghi ngày: 2026-11-12

## Phần cứng

- CPU: 20 luồng
- RAM: 22 GB
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU — 4 GB VRAM
- Driver: 595.99.02

## Phần mềm

- OS: Ubuntu 26.04.1 LTS
- Python: 3.14.4 (venv `.venv`)
- FFmpeg: 8.0.1 (đã cài sẵn)
- CUDA toolkit hệ thống: không cài (dùng CUDA runtime kèm wheel)
- COLMAP: pycolmap-cuda12 4.2.1 (CUDA 12, has_cuda=True)
- torch: 2.10.0 (CUDA 12.8)

## Kết quả kiểm tra

- `torch.cuda.is_available()`: True
- GPU name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- pycolmap.has_cuda: True

## Smoke test COLMAP (bộ ảnh mẫu Middlebury `sample_temple`, 47 ảnh)

Chạy: `python src/scripts/smoke_colmap.py data/input/sample_temple`
Ngày: 2026-11-12 — kết quả chạy sạch từ đầu (xóa toàn bộ output cũ trước khi chạy):

| Bước | Kết quả |
|---|---|
| [1/4] SIFT feature extraction (GPU) | ~0.7 giây |
| [2/4] Exhaustive matching (1081 cặp) | ~0.4 giây |
| [3/4] Sparse reconstruction | 47/47 ảnh đăng ký, 7.630 điểm 3D, reprojection error 0.31 px |
| [4/4] Dense (undistort → patch-match GPU → fusion) | patch-match **10.4 phút**, fusion 3 giây |
| **Tổng** | **634 giây (~10.6 phút)** |
| Dense point cloud | `data/output/smoke_test/mvs/dense.ply` — 4.2 MB |

Script thoát với exit code 0 và in `SMOKE TEST: PASS`.

**Lưu ý kỹ thuật (bug đã fix):** `pycolmap.stereo_fusion()` mặc định `output_type="bin"` —
nó ghi một thư mục model COLMAP, nên truyền path `.ply` sẽ lỗi
`ExistsDir(dense.ply)`. Bắt buộc truyền `output_type="ply"` để ghi file PLY.
Ngoài ra `Reconstruction.num_points3D` là **phương thức** (gọi `num_points3D()`), không phải property.

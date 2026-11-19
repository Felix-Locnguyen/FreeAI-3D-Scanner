# FreeAI 3D Scanner: Giai đoạn và Task

Quy ước: mỗi giai đoạn chỉ bắt đầu khi giai đoạn trước đạt **Điều kiện hoàn thành (DoD)**. Mỗi giai đoạn kết thúc bằng một commit/tag.

---

## Giai đoạn 0: Chuẩn bị môi trường và dữ liệu

- [x] Tạo repo GitHub, thêm `README.md`, `.gitignore`, chọn license (MIT)
- [x] Tạo cấu trúc thư mục theo README
- [x] Tạo môi trường Python (venv `.venv`, Python 3.14) và `requirements.txt`
- [x] Kiểm tra GPU: `nvidia-smi`, PyTorch nhận CUDA, ghi lại VRAM (RTX 3050 Ti 4GB — xem `docs/environment.md`)
- [x] Cài FFmpeg (8.0.1 — đã có sẵn trên máy)
- [x] Cài COLMAP bản có CUDA (`pycolmap-cuda12`), chạy thử trên bộ ảnh mẫu — PASS end-to-end, 47/47 ảnh, exit code 0 (`src/scripts/smoke_colmap.py`)
- [ ] Quay/chụp bộ dữ liệu thử: vật thể nhiều texture, ít texture, bóng, trong suốt, hình dạng phức tạp *(demo — **không chặn G0**, làm khi nào cần demo; hướng dẫn trong `benchmarks/README.md`)*
- [ ] Ghi nhãn mô tả từng bộ dữ liệu (điều kiện sáng, số ảnh, thiết bị) vào `benchmarks/` *(demo — template sẵn trong `benchmarks/README.md`)*

**DoD:** COLMAP chạy tay thành công trên bộ ảnh mẫu ✅ *(đạt 2026-11-12: 47/47 ảnh, PASS, exit 0)*.
*Điều chỉnh 2026-11-14: 3 bộ dữ liệu thử chuyển thành hạng mục demo, **không chặn G0** (quyết định của chủ dự án).*

---

## Giai đoạn 1: Baseline (SIFT + COLMAP)

**1.1 Tiền xử lý** ✅
- [x] `video_processor.py`: trích frame từ video bằng FFmpeg
- [x] `quality_checker.py`: phát hiện mờ (Laplacian variance, chuẩn hóa về 640px)
- [x] Phát hiện thiếu sáng/quá sáng (histogram độ sáng)
- [x] Loại frame trùng lặp (pHash 256-bit) / chọn frame theo khoảng cách góc nhìn
- [x] Xuất báo cáo chất lượng từng ảnh (Good / Blurry / Overexposed / Duplicate) ra JSON

**1.2 Tái tạo cơ bản** ✅
- [x] `colmap_pipeline.py`: chạy `feature_extractor → matcher → mapper` (SIFT) bằng Python
- [x] Đọc kết quả: số ảnh đăng ký, số điểm sparse, reprojection error (ghi `stats.json`)
- [x] Dense reconstruction (undistort → patch match stereo → fusion) — tùy chọn `--dense`
- [x] Xuất point cloud PLY (dense: `mvs/dense.ply`; sparse: `sparse.ply`)

**1.3 Giao diện và đo lường** ✅
- [x] `app.py`: Gradio nhận ảnh/video, gọi pipeline, hiển thị tiến trình
- [x] Hiển thị point cloud/model trên Gradio Model3D
- [x] `src/utils/metrics.py`: ghi số liệu mỗi lần chạy ra JSONL (`data/output/runs.jsonl`)
- [x] Chạy baseline trên toàn bộ bộ dữ liệu thử và lưu bảng kết quả (`src/scripts/run_baseline.py`)

**DoD:** upload video → nhận point cloud xem được trên web; có số liệu baseline cho mọi bộ dữ liệu ✅ *(đạt 2026-11-15, tag `phase-1-done`)*.

**Kết quả baseline** (sparse-only, RTX 3050 Ti 4GB, 2026-11-15):

| Bộ dữ liệu | Ảnh vào | Đăng ký | Điểm 3D | Reprojection | PLY | Thời gian |
|---|---|---|---|---|---|---|
| `sample_temple` (47 ảnh) | 47 | 47 | 7.630 | 0.307 px | 0.11 MB | 7.7 s |
| `test_video` (2 fps) | 24 | 24 | 1.053 | 1.085 px | 0.02 MB | 2.1 s |

*Số điểm 3D dao động ±0.1% giữa các lần chạy (SIFT/matcher GPU không deterministic tuyệt đối).*
*Video test hiện 13/24 frame bị Blurry (video gốc hơi mềm) — quality_checker vẫn giữ lại vì baseline dùng toàn bộ ảnh; app mặc định lọc ảnh Good.*

---

## Giai đoạn 2: Tích hợp Deep Learning

**2.1 Feature matching** ✅ *(đạt 2026-11-16)*
- [x] Tích hợp SuperPoint/ALIKED + LightGlue vào pipeline — **không dùng hloc**: tự trích điểm bằng PyTorch rồi ghi keypoints + matches thẳng vào database COLMAP (`src/ai/feature_matching/matching_model.py`), COLMAP chạy RANSAC + tái tạo như bình thường
- [x] Cho phép chọn matcher trong UI: `gr.Dropdown` SIFT / SuperPoint / ALIKED (`app.py`) và `--matcher` trên dòng lệnh
- [x] Ghi lại inlier ratio, số ảnh đăng ký, thời gian (`stats.json` + `runs.jsonl`, cột `matcher` và `inlier_ratio`)

*Kết quả thử (`sample_temple`, 47 ảnh, RTX 3050 Ti):*

| Matcher | Đăng ký | Điểm 3D | Reprojection | Inlier ratio | Thời gian |
|---|---|---|---|---|---|
| SIFT | 47/47 | 7.628 | 0.307 px | 0.964 | 7.4 s |
| SuperPoint+LightGlue | 45/47 | 4.350 | 1.095 px | 0.904 | 57.9 s |
| ALIKED+LightGlue | — (thử 8 ảnh) | — | — | 0.972 | — |

**2.2 Segmentation** ✅ *(đạt 2026-11-17)*
- [x] Tích hợp SAM 2 (`sam2_model.py`): tạo mask vật thể trên ảnh — `load_model()`, `biggest_mask()` (tự động), `point_mask()` (click), `apply_mask()` (tô nền đen → `work_dir/images_masked/`)
- [x] Hỗ trợ video: **1 điểm click (tỷ lệ u,v) áp cho mọi frame** — thay cho "theo dõi vật thể" (phương án 1 đã chọn: đơn giản, không cần tracker)
- [x] Cách chọn vật thể trong UI: click điểm trên ảnh xem trước `gr.Image` (box chưa làm)
- [x] Áp mask khi trích đặc trưng để loại nền — thêm nhánh `if mask:` để đổi `image_dir`, **không sửa** SIFT/LightGlue
- [x] Công tắc bật/tắt mask trong UI: `gr.Checkbox` "Loại nền bằng SAM 2"

*Kết quả (`test_video`, 24 frame 1920×1080, RTX 3050 Ti, 2026-11-17):*

| Cấu hình | Đăng ký | Điểm 3D | Reprojection | Inlier | Thời gian mask | Tổng |
|---|---|---|---|---|---|---|
| SIFT (baseline) | 24/24 | 1.053 | 1,085 px | — | — | 2,1 s |
| SIFT + mask tự động | 20/24 | 873 | 1,030 px | 0,814 | 111,7 s | 113,9 s |
| SIFT + mask click | 24/24 | 901 | 1,042 px | 0,604 | 4,5 s | 7,5 s |
| SuperPoint (baseline) | 24/24 | 5.200 | 1,380 px | 0,768 | — | 16,3 s |
| SuperPoint + mask tự động | 21/24 | 2.343 | 1,452 px | 0,823 | 111,4 s | 120,2 s |
| SuperPoint + mask click | 24/24 | 2.509 | 1,430 px | 0,794 | 4,5 s | 14,5 s |

*Đọc bảng: mask **cắt bớt điểm nền** (SuperPoint 5.200 → 2.509 điểm) nhưng giữ được 24/24 ảnh đăng ký; **click nhanh gấp ~25×** so với tự động (4,5 s vs 111,7 s).*
***`sample_temple` không dùng được để test mask** — ảnh Middlebury vốn đã có nền đen, mask chỉ cắt mất phần đền → vài ảnh chỉ còn 3–6 điểm SIFT.*

**2.3 So sánh thực nghiệm**
- [x] Chạy 4 cấu hình: SIFT, SIFT + mask, LightGlue, LightGlue + mask
- [x] Điền bảng so sánh trên toàn bộ bộ dữ liệu
- [x] Viết phân tích ngắn: cấu hình nào tốt hơn, ở loại vật thể nào, vì sao

**DoD:** bảng so sánh 4 cấu hình có số liệu thật; UI chọn được matcher và mask. ✅

> Bảng + phân tích đầy đủ trong `benchmarks/README.md`. Số liệu chính: **SIFT thắng về đăng ký (47/47, 24/24) và tốc độ (~8× nhanh LightGlue)**; **LightGlue thắng về số điểm (5×)**; **mask chỉ có ích với vật thể tách được khỏi nền** — trên `sample_temple` nó phá (47/47 → 6/47), trên `test_video` cắt 54% điểm nền nhưng mất 3–4/24 ảnh.

---

## Giai đoạn 3: Mesh, Texture và Xuất file

**3.1 Xử lý point cloud**
- [ ] `pointcloud_processor.py`: lọc nhiễu (statistical outlier), downsample
- [ ] Ước lượng normal
- [ ] Cắt vùng vật thể (dùng mask hoặc bounding box)

**3.2 Mesh**
- [ ] `mesh_processor.py`: Poisson surface reconstruction (Open3D)
- [ ] Loại vùng mật độ thấp, decimate, làm mượt
- [ ] Kiểm tra mesh: số lỗ hổng, thành phần rời, watertight

**3.3 Texture**
- [ ] `texture_processor.py`: texture mapping bằng OpenMVS (TextureMesh) hoặc phương án tương đương
- [ ] Kiểm tra UV và texture có được giữ đúng

**3.4 Xuất file**
- [ ] Xuất GLB (trimesh)
- [ ] Xuất OBJ + MTL + texture
- [ ] Xuất PLY
- [ ] Mở thử từng file trong Blender/MeshLab để xác nhận texture còn nguyên
- [ ] Nút tải file trong UI

**DoD:** ít nhất 2 vật thể ra được mesh có texture, mở đúng trong phần mềm 3D.
*Nếu texture bị kẹt: giao mesh không texture trước, texture để làm sau.*

---

## Giai đoạn 4: Video 360°, Báo cáo và Hoàn thiện

**4.1 Video 360°**
- [ ] `video_renderer.py`: đặt camera quỹ đạo tròn quanh mô hình
- [ ] Render từng khung (Open3D offscreen hoặc pyrender)
- [ ] Ghép MP4/WebM bằng FFmpeg
- [ ] Tùy chọn độ phân giải và số khung hình trong UI

**4.2 Báo cáo tái tạo**
- [ ] Trang tóm tắt: số ảnh đầu vào/đăng ký, số điểm, số mặt mesh, thời gian từng bước, dung lượng file
- [ ] Hiển thị ảnh bị loại và lý do
- [ ] Lưu lịch sử các lần chạy (SQLite) và xem lại được

**4.3 Chất lượng mã và tài liệu**
- [ ] Unit test cho các hàm tiền xử lý, metrics, xuất file
- [ ] Xử lý lỗi: pipeline thất bại → thông báo nguyên nhân gợi ý (thiếu overlap, ảnh mờ, ít texture)
- [ ] Hoàn thiện README: ảnh/GIF demo, bảng benchmark thật, hạn chế đã biết
- [ ] `docs/installation.md`: hướng dẫn cài COLMAP, OpenMVS
- [ ] Kiểm tra lại license từng thư viện và trọng số, ghi vào README
- [ ] Gắn tag phiên bản v1.0

**DoD:** người lạ clone repo, làm theo README, chạy được từ video đến GLB và video 360°.

---

## Giai đoạn 5 (tùy chọn): Nghiên cứu mở rộng

- [ ] Nhánh thử nghiệm MASt3R hoặc 3D Gaussian Splatting, so sánh với pipeline chính trên cùng bộ dữ liệu
- [ ] Module đánh giá chất lượng ảnh bằng CNN (EfficientNet), so với baseline Laplacian; cần tự gán nhãn dữ liệu
- [ ] Thí nghiệm Depth Anything V2 như tín hiệu hình học phụ
- [ ] Đóng gói Docker
- [ ] Demo công khai (Hugging Face Spaces/Colab) với giới hạn lượt và độ phân giải
- [ ] Hỗ trợ xuất FBX, STL
- [ ] Viết SRS/TDD ngắn mô tả kiến trúc cuối cùng
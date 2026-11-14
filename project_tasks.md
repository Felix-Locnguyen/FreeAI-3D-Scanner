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

**1.1 Tiền xử lý**
- [ ] `video_processor.py`: trích frame từ video bằng FFmpeg
- [ ] `quality_checker.py`: phát hiện mờ (Laplacian variance)
- [ ] Phát hiện thiếu sáng/quá sáng (histogram độ sáng)
- [ ] Loại frame trùng lặp / chọn frame theo khoảng cách góc nhìn
- [ ] Xuất báo cáo chất lượng từng ảnh (Good / Blurry / Overexposed / Duplicate)

**1.2 Tái tạo cơ bản**
- [ ] `colmap_pipeline.py`: chạy `feature_extractor → matcher → mapper` (SIFT) bằng Python
- [ ] Đọc kết quả: số ảnh đăng ký, số điểm sparse, reprojection error
- [ ] Dense reconstruction (undistort → patch match stereo → fusion)
- [ ] Xuất point cloud PLY

**1.3 Giao diện và đo lường**
- [ ] `app.py`: Gradio nhận ảnh/video, gọi pipeline, hiển thị tiến trình
- [ ] Hiển thị point cloud/model trên Gradio Model3D
- [ ] `metrics.py`: ghi số liệu mỗi lần chạy ra JSON/SQLite
- [ ] Chạy baseline trên toàn bộ bộ dữ liệu thử và lưu bảng kết quả

**DoD:** upload video → nhận point cloud xem được trên web; có số liệu baseline cho mọi bộ dữ liệu.

---

## Giai đoạn 2: Tích hợp Deep Learning

**2.1 Feature matching**
- [ ] Tích hợp ALIKED (hoặc SuperPoint) + LightGlue vào pipeline (COLMAP có hỗ trợ hoặc dùng hloc rồi import)
- [ ] Cho phép chọn matcher trong UI: SIFT / ALIKED+LightGlue
- [ ] Ghi lại inlier ratio, số ảnh đăng ký, thời gian

**2.2 Segmentation**
- [ ] Tích hợp SAM 2 (`sam2_model.py`): tạo mask vật thể trên ảnh
- [ ] Hỗ trợ video: theo dõi vật thể qua các frame
- [ ] Cách chọn vật thể trong UI (click điểm hoặc box)
- [ ] Áp mask khi trích đặc trưng để loại nền
- [ ] Công tắc bật/tắt mask trong UI

**2.3 So sánh thực nghiệm**
- [ ] Chạy 4 cấu hình: SIFT, SIFT + mask, LightGlue, LightGlue + mask
- [ ] Điền bảng so sánh trên toàn bộ bộ dữ liệu
- [ ] Viết phân tích ngắn: cấu hình nào tốt hơn, ở loại vật thể nào, vì sao

**DoD:** bảng so sánh 4 cấu hình có số liệu thật; UI chọn được matcher và mask.

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
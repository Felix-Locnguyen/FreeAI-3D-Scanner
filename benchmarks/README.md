# Bộ dữ liệu thử (Benchmarks)

## Danh sách tối thiểu (DoD Giai đoạn 0: >= 3 bộ)

| Mã | Loại | Vật thể gợi ý |
|---|---|---|
| ds01_texture | Nhiều texture | Sách, hộp có nhãn, đồ chơi có hoa văn |
| ds02_lowtex | Ít texture / bóng | Cốc trắng, hộp carton, bóng gương |
| ds03_complex | Hình dạng phức tạp | Giày, xe đồ chơi, giá sách nhỏ |
| ds04_clear (tùy chọn) | Trong suốt | Chai nước trong, ly thủy tinh |

## Quy tắc chụp

- 30-60 ảnh, quanh vật thể đủ 360°, overlap giữa 2 ảnh >= 60%
- 2-3 độ cao khác nhau (ngang, cao hơn, thấp hơn)
- Ảnh sắc nét, đủ sáng, tránh ngược sáng mạnh
- Vật thể cố định — chỉ di chuyển điện thoại, không zoom
- Nền càng ít chi tiết càng tốt (sau này tách vật thể bằng SAM 2)
- Nếu quay video: quay chậm 1 vòng quanh vật thể, 20-40 giây

## Cấu trúc thư mục

- Ảnh: `data/input/<ma_bo>/`  (ví dụ: `data/input/ds01_texture/`)
- Nhãn: `benchmarks/<ma_bo>/metadata.md`

## Template metadata

```markdown
# <ma_bo> — <ten vat the>

- **Vat the:** <vd: hop banh quy co hoa van>
- **Loai:** <nhieu texture / it texture / phuc tap / trong suot>
- **Thiet bi chup:** <vd: iPhone 13 / Samsung A54>
- **Ngay chup:** <YYYY-MM-DD>
- **Dieu kien sang:** <vd: trong nha, anh sang tu nhien, khong flash>
- **So anh:** <n>  (hoac video <n>s, fps=2)
- **Khoang cach:** <vd: 0.5-0.8 m quanh vat the>
- **Ghi chu:** <ghi chu them>
```

## Ghi chú

- Bộ ảnh mẫu `sample_temple` (47 ảnh từ Middlebury) dùng để test COLMAP, không phải bộ dữ liệu sản phẩm.

---

# Benchmark 4 cấu hình (Giai đoạn 2.3)

## 4 cấu hình so sánh

| Cấu hình | Feature | Matcher | Loại nền SAM 2 | Mô tả |
|---|---|---|---|---|
| `sift` | SIFT | SIFT (COLMAP) | không | Mặc định, nhanh nhất |
| `sift+mask` | SIFT | SIFT (COLMAP) | có | Tách vật thể khỏi nền trước khi khớp |
| `lightglue` | SuperPoint | LightGlue | không | Mạng nơ-ron — nhiều điểm hơn nhưng nặng hơn |
| `lightglue+mask` | SuperPoint | LightGlue | có | Kết hợp cả hai |

## Bộ dữ liệu dùng

| Bộ | Số ảnh | Đặc điểm | Vì sao dùng |
|---|---|---|---|
| `sample_temple` | 47 (640×480) | Toàn cảnh, nền đen (Middlebury) | Cực "ảnh chụp thật, nhiều texture" |
| `test_video` | 24 (1920×1080) | Vật thể trên nền trơn | Cực "vật thể tách khỏi nền" — đúng loại ảnh SAM 2 sinh ra để xử lý |

> `ds01`–`ds03` theo DoD Giai đoạn 0 là **bộ demo, không chặn** — chưa chụp nên chưa đưa vào bảng.

## Cách chạy

```bash
python src/scripts/run_benchmark.py            # chạy nốt phần chưa chạy (resume)
python src/scripts/run_benchmark.py --only test_video
python src/scripts/run_benchmark.py --force    # chạy lại từ đầu
python src/scripts/run_benchmark.py --render   # in bảng từ dữ liệu đã có
```

- Mỗi (bộ, cấu hình) ghi 1 dòng vào `benchmarks/results.jsonl`, chạy ở thư mục riêng `data/output/bench/<bộ>_<cấu hình>` để không ghi đè nhau.
- Mask dùng **chế độ tự động** (mặc định của app): chọn mask chứa tâm ảnh.
- Chỉ chạy COLMAP **sparse** — cùng chế độ với Giai đoạn 2.1/2.2 nên số liệu so được trực tiếp.

## Bảng kết quả

| Bộ dữ liệu | Cấu hình | Đăng ký | Điểm 3D | Reproj (px) | Inlier | Mask (s) | Tổng (s) |
|---|---|---|---|---|---|---|---|
| sample_temple | sift | 47/47 | 7.625 | 0,306 | 0,964 | — | 7,7 |
| sample_temple | sift+mask | 6/47 | 766 | 0,251 | 0,966 | 208,3 | 210,5 |
| sample_temple | lightglue | 46/47 | 4.279 | 1,112 | 0,904 | — | 59,8 |
| sample_temple | lightglue+mask | 13/47 | 615 | 0,992 | 0,788 | 210,6 | 224,8 |
| test_video | sift | 24/24 | 1.049 | 1,074 | 0,817 | — | 2,2 |
| test_video | sift+mask | 20/24 | 875 | 1,035 | 0,816 | 109,3 | 111,3 |
| test_video | lightglue | 24/24 | 5.151 | 1,374 | 0,767 | — | 18,3 |
| test_video | lightglue+mask | 21/24 | 2.357 | 1,432 | 0,828 | 115,2 | 124,1 |

*Chạy 2026-11-17, RTX 3050 Ti 4 GB, COLMAP sparse-only. In lại bảng bất cứ lúc nào bằng `python src/scripts/run_benchmark.py --render`.*

## Phân tích

**Không có cấu hình thắng tuyệt đối — mỗi cấu hình thắng 1 tiêu chí:**

| Tiêu chí | Cấu hình thắng | Số liệu |
|---|---|---|
| Đăng ký nhiều ảnh + nhanh nhất | **SIFT** | 47/47 và 24/24; 7,7 s và 2,2 s — nhanh LightGlue ~8× |
| Nhiều điểm 3D nhất | **LightGlue** | 5.151 vs 1.049 điểm trên `test_video` (~5×) |
| Khớp chính xác nhất | **SIFT** | 0,306 px trên `sample_temple` |
| Inlier tốt nhất khi bật mask | **LightGlue + mask** | 0,828 vs 0,767 (+8%) |

**Khuyến nghị cho app:** mặc định **SIFT, không mask** — đăng ký đủ ảnh, nhanh, chính xác. Chỉ bật mask khi cần point cloud sạch nền.

**Mask giúp ở loại vật thể nào?** Chỉ khi vật thể tách được khỏi nền:

- `sample_temple` (nền đã đen sẵn) → **phá**: 47/47 → 6/47 với SIFT, 46/47 → 13/47 với LightGlue.
- `test_video` (vật thể trên nền trơn) → **cắt 54% điểm nền** (5.151 → 2.357) nhưng **mất 3–4/24 ảnh** và tốn thêm ~115 s.

**Vì sao?** Mask cắt feature points nên mỗi cặp ảnh còn ít điểm để khớp → một số cặp rơi dưới ngưỡng → ảnh không đăng ký. Chọn mask **tự động** (chứa tâm ảnh) cũng hay trúng sai: ở Giai đoạn 2.2 chế độ **click** cho 24/24 trong khi tự động chỉ 20/24.

**Bù lại, mask làm khớp "sạch" hơn:** inlier LightGlue 0,767 → 0,828, reproj SIFT 1,074 → 1,035 px — vì điểm nền gây nhiễu đã bị dọn đi.

**Hạn chế của bảng này:** chỉ đo COLMAP **sparse** — lợi ích thật của mask (giảm điểm nền cho MVS) chưa được đo; mới có 2 bộ dữ liệu; mask dùng chế độ tự động chứ không phải click.

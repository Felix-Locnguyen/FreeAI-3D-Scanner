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
- Nền尽量 ít chi tiết (sau này tách vật thể bằng SAM 2)
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

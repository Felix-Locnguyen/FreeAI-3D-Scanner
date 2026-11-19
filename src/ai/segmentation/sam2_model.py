"""
Phân đoạn vật thể bằng SAM 2 để loại nền trước khi trích đặc trưng (Giai đoạn 2.2).

Vì sao cần: ảnh có nền (bàn, tường, giấy) làm COLMAP bám vào nền thay vì vật thể.
Bước này tô nền thành màu ĐEN, giữ lại vật thể -> điểm 3D tập trung vào vật thể.

2 cách chọn vật thể:
  - TỰ ĐỘNG (mặc định): SAM 2 tự tìm vật thể ở giữa khung hình trong từng ảnh.
  - CLICK: người dùng bấm 1 điểm trên ảnh xem trước -> dùng điểm đó (theo tỉ lệ)
    cho mọi ảnh, không phải click từng ảnh.

Chạy thử độc lập:
    python src/ai/segmentation/sam2_model.py data/input/sample_temple --limit 3
    python src/ai/segmentation/sam2_model.py data/input/sample_temple --point 0.5,0.5
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Trọng số SAM 2 (tải 1 lần, 149 MB) — xem docs/environment.md
CHECKPOINT = Path("models/segmentation/sam2.1_hiera_tiny.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Model + predictor dùng chung cho cả phiên (tải 1 lần mất ~2 giây)
_MODEL = None
_PREDICTOR = None


def load_model():
    """Tải SAM 2 một lần rồi dùng lại cho mọi ảnh (mỗi lần load ~2 giây)."""
    global _MODEL, _PREDICTOR
    if _MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[sam2] tai model tren {device}...", flush=True)
        _MODEL = build_sam2(MODEL_CFG, str(CHECKPOINT), device=device)
        _PREDICTOR = SAM2ImagePredictor(_MODEL)
    return _MODEL, _PREDICTOR


def biggest_mask(model, image):
    """TỰ ĐỘNG: tìm vật thể ở giữa khung hình -> mask (H, W) kiểu bool.

    Ảnh to (1080p) làm SAM 2 hết VRAM 4 GB -> resize về tối đa 1024px trước, vì
    SAM 2 bên trong cũng chỉ nén ảnh về ~1024px nên chất lượng không đổi.
    """
    h, w = image.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    if scale < 1.0:
        small = np.array(Image.fromarray(image)
                         .resize((round(w * scale), round(h * scale))))
    else:
        small = image

    masks = SAM2AutomaticMaskGenerator(model).generate(small)
    if not masks:
        return None
    # Sắp giảm dần theo diện tích để tiện chọn
    masks.sort(key=lambda m: m["area"], reverse=True)

    # Chọn mask chứa TÂM ẢNH — vật thể thường nằm ở giữa khung hình.
    # (Chọn mask LỚN NHẤT sẽ trúng NỀN nếu nền bao quanh vật thể.)
    cy, cx = small.shape[0] // 2, small.shape[1] // 2
    mask = None
    for m in masks:                     # masks đã sắp theo diện tích giảm dần
        if m["segmentation"][cy, cx]:
            mask = m["segmentation"]
            break
    if mask is None:                    # không mask nào chứa tâm -> lấy lớn nhất
        mask = masks[0]["segmentation"]

    if scale < 1.0:                     # phóng mask về kích thước gốc
        mask = np.array(Image.fromarray((mask.astype(np.uint8)) * 255)
                        .resize((w, h), Image.Resampling.NEAREST)) > 0
    return mask


def point_mask(predictor, image, u, v):
    """CLICK: tìm vật thể tại điểm (u, v) — toạ độ TỈ LỆ 0–1 -> mask (H, W) bool."""
    h, w = image.shape[:2]
    x, y = int(u * w), int(v * h)
    with torch.inference_mode():               # tắt đạo hàm -> nhanh, tốn RAM ít
        predictor.set_image(image)
        masks, _scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),        # 1 = "đây là vật thể"
            multimask_output=True,             # SAM trả 3 mask ứng viên
        )

    # Chọn mask LỚN NHẤT trong 3 ứng viên (cùng chứa điểm click).
    # Chọn theo score (IoU dự đoán) thường trúng MẢNH CON của vật thể:
    # đã đo trên 3 ảnh, 2 lần score cao nhất chỉ giữ 1.9–2.1% khung hình
    # trong khi mask lớn giữ ~26% (cả áo). SAM 2 trả mask float32 (0.0/1.0).
    best = int(np.argmax(masks.sum(axis=(1, 2))))
    return masks[best].astype(bool)


def apply_mask(src_dir, dst_dir, point=None, limit=None, log=print):
    """Tạo bản ảnh đã tô nền đen cho mọi ảnh trong src_dir -> dst_dir.

    point: (u, v) tỉ lệ 0–1 nếu người dùng đã click; None = tự động.
    limit: chỉ xử lý N ảnh đầu (dùng cho chạy thử nhanh).
    Trả về dict số liệu.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)

    # Xóa bản mask cũ (nếu có) để không lẫn ảnh của lần chạy trước
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"Khong co anh nao trong {src_dir}")

    model, predictor = load_model()
    log(f"[mask] SAM 2 tren {len(paths)} anh, cach chon vat the: "
        + ("click" if point else "tu dong (vat the o giua khung hinh)"))

    t = time.time()
    for i, path in enumerate(paths, 1):
        image = np.array(Image.open(path).convert("RGB"))
        mask = point_mask(predictor, image, *point) if point else biggest_mask(model, image)

        out = image.copy()
        if mask is None:                        # SAM không thấy gì -> giữ ảnh gốc
            log(f"  [{i}/{len(paths)}] {path.name}: khong thay vat the, giu anh goc")
        else:
            out[~mask] = 0                      # tô nền = màu đen, giữ lại vật thể
            if i % 10 == 0 or i == len(paths):
                log(f"  [{i}/{len(paths)}] vat the chiem {mask.mean():.0%} khung anh")

        # JPEG nén lại ở chất lượng cao; PNG bỏ qua tham số quality
        save_kw = {"quality": 95} if path.suffix.lower() in {".jpg", ".jpeg"} else {}
        Image.fromarray(out).save(dst_dir / path.name, **save_kw)

    stats = {
        "mask": True,
        "mask_point": list(point) if point else None,
        "mask_images": len(paths),
        "mask_s": round(time.time() - t, 1),
    }
    log(f"[mask] xong {len(paths)} anh trong {stats['mask_s']} giay")

    # BẮT BUỘC: nhả bộ nhớ GPU đã dùng cho SAM 2 trước khi COLMAP chạy.
    # torch giữ lại vùng VRAM đã cấp (dù không dùng nữa) -> COLMAP hết VRAM trên
    # card 4 GB và SIFT GPU báo "illegal memory access" -> 0 điểm đặc trưng.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def main():
    # --point u,v và --limit N: lấy giá trị rồi loại khỏi argv để không bị coi là đường dẫn
    point = None
    if "--point" in sys.argv:
        i = sys.argv.index("--point")
        point = tuple(float(v) for v in sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]

    limit = None
    if "--limit" in sys.argv:
        i = sys.argv.index("--limit")
        limit = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("Chạy: python src/ai/segmentation/sam2_model.py <thu_muc_anh> "
              "[--limit N] [--point 0.5,0.5]")
        return 2

    out_dir = Path("data/output/mask_test")
    stats = apply_mask(Path(args[0]), out_dir / "images_masked",
                       point=point, limit=limit)
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n=== TOM TAT ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

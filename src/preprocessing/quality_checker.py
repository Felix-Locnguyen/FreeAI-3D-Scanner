"""
Kiểm tra chất lượng ảnh đầu vào: mờ, thiếu sáng, quá sáng, trùng lặp.

Lệnh chạy:
    python src/preprocessing/quality_checker.py data/input/sample_temple
    python src/preprocessing/quality_checker.py data/frames/test_video data/output/test_video/quality.json
"""
import json 
import sys
from pathlib import Path 
from PIL import Image

import cv2
import imagehash # chức năng xóa các frame bị trùng lặp

# --- Ngưỡng: hiệu chuẩn trên ảnh thật, đừng copy máy móc ---
BLUR_WIDTH = 640             # đưa mọi ảnh về cùng độ rộng rồi mới đo độ nét
BLUR_THRESHOLD = 70.0        # blur < ngưỡng -> mờ
                             # (47 ảnh temple nét: min 79.3, median 144.7)
DARK_THRESHOLD = 15.0        # độ sáng trung bình < ngưỡng -> thiếu sáng
                             # (temple: min 19.5, median 33.3 — nền tối nhưng chủ thể rõ)
BRIGHT_THRESHOLD = 235.0     # độ sáng trung bình > ngưỡng -> quá sáng
HASH_SIZE = 16               # phash 16x16 = 256 bit (8x8/64 bit quá thô, dễ báo trùng nhầm)
DUPLICATE_THRESHOLD = 8      # khoảng cách Hamming <= ngưỡng -> trùng lặp
                             # (copy/resize lại = 0; 2 góc nhìn khác nhau gần nhất = 10
                             #  -> phải để ngưỡng dưới 10, nếu không sẽ bỏ mất view cho SfM)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(image_dir):
    """
    Liệt kê các ảnh trong thư mục, sắp xếp theo thứ tự tên 
    """
    return sorted(pic for pic in image_dir.iterdir() if pic.suffix.lower() in IMAGE_EXTS)


def check_images(image_dir):
    """kiểm tra chất lượng từng ảnh"""
    results = []
    kept_hashes = []  # chỉ giữ hash của ảnh Good: ảnh mờ/tối đã bị loại nên không cần so trùng
                      # (nếu 0 ảnh Good -> hash_distance luôn = 999, không phải lỗi phash)

    for path in list_images(image_dir):
        img = cv2.imread(str(path))
        if img is None:
            print(f'Bo qua : {path.name}')
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Độ nét: ảnh nét thì biến đổi cường độ mạnh -> phương sai Laplacian cao
        #    Phải resize về cùng độ rộng: cùng 1 ảnh, blur@1080p có thể nhỏ hơn blur@640 tới 15 lần
        h, w = gray.shape
        if w != BLUR_WIDTH:
            small = cv2.resize(
                gray,
                (BLUR_WIDTH, round(BLUR_WIDTH * h / w)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = gray
        blur_score = cv2.Laplacian(small, cv2.CV_64F).var()

        # 2. Độ sáng trung bình (0 = đen, 255 = trắng)
        brightness = float(gray.mean())

        # 3. Perceptual hash: mở bằng PIL để tránh lỗi PosixPath, đóng file sau khi dùng
        with Image.open(path) as img_pil:
            phash = imagehash.phash(img_pil, hash_size=HASH_SIZE)

        distance = min((phash - prev for prev in kept_hashes), default=999)

        if blur_score < BLUR_THRESHOLD:
            status = "Blurry"
        elif brightness < DARK_THRESHOLD:
            status = "Underexposed"
        elif brightness > BRIGHT_THRESHOLD:
            status = "Overexposed"
        elif distance <= DUPLICATE_THRESHOLD:
            status = "Duplicate"
        else:
            status = "Good"
            kept_hashes.append(phash)

        results.append({
            "name": path.name,
            "status": status,
            "blur": round(float(blur_score), 1),
            "brightness": round(brightness, 1),
            "hash_distance": int(distance),
        })

    return results


def print_report(results):
    """In bang ket qua + tong ket theo trang thai."""
    print(f"\n{'Anh':<22}{'Trang thai':<15}{'Blur':>10}{'Sang':>8}{'Hash':>6}")
    print("-" * 61)
    for r in results:
        print(f"{r['name']:<22}{r['status']:<15}"
              f"{r['blur']:>10.1f}{r['brightness']:>8.1f}{r['hash_distance']:>6}")

    print("\nTong ket:")
    for status in ["Good", "Blurry", "Underexposed", "Overexposed", "Duplicate"]:
        count = sum(1 for r in results if r["status"] == status)
        if count:
            print(f"  {status:<15}{count:>4}")

    # Nếu quá nửa số frame bị mờ thì thường là cả video mềm, không phải lỗi ngưỡng
    n_blur = sum(1 for r in results if r["status"] == "Blurry")
    if results and n_blur / len(results) > 0.5:
        print(f"\nCANH BAO: {n_blur}/{len(results)} frame bi mo")
        print("  -> nen quay lai video net hon (hoac kiem tra lai nguong BLUR_THRESHOLD).")


def main(image_dir, out_path=None):
    image_dir = Path(image_dir)
    out_path = Path(out_path) if out_path else None

    if not image_dir.is_dir():
        print(f"LOI: khong thay thu muc {image_dir}")
        return 1

    results = check_images(image_dir)
    print_report(results)

    # Ghi file nếu có chỉ định out_path
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nDa ghi bao cao: {out_path}")
    return 0


if __name__ == "__main__":
    # Terminal: python src/preprocessing/quality_checker.py <thu_muc_anh> [bao_cao.json]
    # Bam Run (khong tham so): dung duong dan mac dinh duoi day
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
    sys.exit(main("data/frames/test_video", "data/output/test_video/quality_report.json"))
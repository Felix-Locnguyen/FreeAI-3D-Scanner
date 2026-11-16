"""
Trích điểm đặc trưng + ghép cặp ảnh bằng mạng nơ-ron (Giai đoạn 2.1).

Khác với SIFT (để COLMAP tự làm), ở đây ta tự làm 3 bước:
  1. SuperPoint / ALIKED tìm điểm đặc trưng trên từng ảnh   (PyTorch)
  2. LightGlue ghép điểm giữa mọi cặp ảnh                   (PyTorch)
  3. Ghi keypoints + matches vào database của COLMAP         (pycolmap)
Rồi COLMAP chạy RANSAC và tái tạo 3D như bình thường.

Chạy thử độc lập:
    python src/ai/feature_matching/matching_model.py data/input/sample_temple
    python src/ai/feature_matching/matching_model.py data/input/sample_temple --limit 8 --extractor aliked
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch
from lightglue import ALIKED, LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_KEYPOINTS = 2048        # mỗi ảnh tối đa bao nhiêu điểm đặc trưng


def load_models(name, device):
    """Tạo 2 mạng: trích điểm (SuperPoint/ALIKED) + ghép điểm (LightGlue)."""
    if name == "aliked":
        extractor = ALIKED(max_num_keypoints=MAX_KEYPOINTS)
    else:
        extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS)
    # features= phải trùng tên extractor để LightGlue tự tải đúng trọng số
    matcher = LightGlue(features=name).eval().to(device)
    return extractor.eval().to(device), matcher


def extract_all(extractor, paths, device, log=print):
    """Trích điểm đặc trưng cho từng ảnh -> dict {tên_ảnh: feats}."""
    feats_all = {}
    for i, path in enumerate(paths, 1):
        with torch.inference_mode():                  # tắt đạo hàm -> nhanh, tốn RAM ít
            feats = extractor.extract(load_image(str(path)).to(device))
        feats_all[path.name] = feats
        log(f"  [{i}/{len(paths)}] {path.name}: {feats['keypoints'].shape[1]} điểm")
    return feats_all


def match_all(matcher, paths, feats_all, log=print):
    """Ghép mọi cặp ảnh -> list[(tên_ảnh_0, tên_ảnh_1, matches (M, 2))]."""
    results, done = [], 0
    total = len(paths) * (len(paths) - 1) // 2
    for i, path0 in enumerate(paths):
        for path1 in paths[i + 1:]:
            with torch.inference_mode():
                out = matcher({"image0": feats_all[path0.name],
                               "image1": feats_all[path1.name]})
            matches = rbd(out)["matches"].cpu().numpy().astype(np.uint32)
            done += 1
            if len(matches):
                results.append((path0.name, path1.name, matches))
            if done % 50 == 0:
                log(f"  {done}/{total} cặp đã ghép")
    return results


def write_database(db_path, image_dir, feats_all, matches, log=print):
    """Ghi keypoints và các cặp match vào database COLMAP."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with pycolmap.Database.open(db_path):        # tạo file database rỗng trước
        pass
    pycolmap.import_images(db_path, image_dir, pycolmap.CameraMode.AUTO, list(feats_all))

    with pycolmap.Database.open(db_path) as db:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        for name, feats in feats_all.items():
            # Trích điểm trả tâm pixel 0; COLMAP định nghĩa tâm pixel 0.5 -> cộng 0.5
            db.write_keypoints(ids[name], feats["keypoints"][0].cpu().numpy() + 0.5)
        for name0, name1, m in matches:
            db.write_matches(ids[name0], ids[name1], m)
    log(f"  da ghi {len(feats_all)} anh + {len(matches)} cap match vao {db_path.name}")


def db_match_stats(db_path):
    """Số liệu match trong database: match thô và match còn lại sau RANSAC.
    Dùng chung cho cả SIFT và matcher học sâu.
    """
    with pycolmap.Database.open(db_path) as db:
        raw = sum(len(m) for m in db.read_all_matches()[1])
        inliers = sum(db.read_two_view_geometry_num_inliers()[1])
    return {"raw_matches": raw, "inlier_matches": inliers,
            "inlier_ratio": round(inliers / raw, 3) if raw else None}


def run_deep_matching(image_dir, db_path, extractor="superpoint", limit=None, log=print):
    """Trích điểm + ghép cặp + ghi database COLMAP. Trả về dict số liệu."""
    image_dir = Path(image_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if limit:
        paths = paths[:limit]
    if len(paths) < 2:
        raise ValueError(f"Can it nhat 2 anh, dang co {len(paths)}")

    stats = {"matcher": extractor, "device": str(device), "timings": {}}
    log(f"[matcher] {extractor} + LightGlue tren {device}, {len(paths)} anh")

    # --- 1. Trích điểm đặc trưng ---
    t = time.time()
    log("[1/3] Trich diem dac trung...")
    net_extractor, net_matcher = load_models(extractor, device)
    feats_all = extract_all(net_extractor, paths, device, log)
    stats["timings"]["extract"] = round(time.time() - t, 1)

    # --- 2. Ghép cặp ảnh ---
    t = time.time()
    log("[2/3] Ghep cap anh...")
    matches = match_all(net_matcher, paths, feats_all, log)
    stats["timings"]["match"] = round(time.time() - t, 1)

    # --- 3. Ghi database + RANSAC loại match sai ---
    t = time.time()
    log("[matcher] Ghi database + RANSAC...")
    write_database(db_path, image_dir, feats_all, matches, log)
    pycolmap.geometric_verification(str(db_path))
    stats["timings"]["verify"] = round(time.time() - t, 1)

    stats["pairs_matched"] = len(matches)
    stats.update(db_match_stats(db_path))
    log(f"  match: {stats['raw_matches']} th -> {stats['inlier_matches']} inlier "
        f"(ty le {stats['inlier_ratio']})")
    return stats


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("Chạy: python src/ai/feature_matching/matching_model.py <thu_muc_anh> "
              "[--limit N] [--extractor superpoint|aliked]")
        return 2

    image_dir = Path(args[0])
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    extractor = (sys.argv[sys.argv.index("--extractor") + 1]
                 if "--extractor" in sys.argv else "superpoint")

    work_dir = Path("data/output/matching_test")
    stats = run_deep_matching(image_dir, work_dir / "database.db",
                              extractor=extractor, limit=limit)
    stats["timings"]["total"] = round(sum(stats["timings"].values()), 1)
    (work_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n=== TOM TAT ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

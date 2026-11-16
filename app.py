"""Gradio app: video/ảnh -> point cloud 3D (COLMAP).

Chạy:
    python app.py      # http://127.0.0.1:7860
"""
from datetime import datetime
from pathlib import Path
import shutil

import gradio as gr
import pandas as pd
import pycolmap

from src.preprocessing.quality_checker import check_images
from src.preprocessing.video_processor import extract_frames
from src.reconstruction.colmap_pipeline import run_colmap
from src.utils.metrics import count_status, make_record, save_run

# Định nghĩa các thư mục cốt lõi của dự án
INPUT_DIR = Path("data/input")      # Nơi lưu trữ ảnh tải lên thủ công
FRAMES_DIR = Path("data/frames")    # Nơi lưu trữ các frame cắt ra từ video
OUTPUT_DIR = Path("data/output")    # Nơi lưu kết quả tái tạo 3D và log báo cáo


# CÁC HÀM XỬ LÝ DỮ LIỆU ĐẦU VÀO & TIỀN XỬ LÝ

def prepare_images(video, images, fps, run_id):
    """Chuẩn bị thư mục ảnh đầu vào cho pipeline.
    
    - Ưu tiên xử lý video nếu người dùng tải lên cả video lẫn ảnh.
    - Trả về: (đường dẫn thư mục chứa ảnh, loại nguồn dữ liệu 'video' hoặc 'images').
    """
    # Trường hợp 1: Người dùng upload video -> Trích xuất frame theo số FPS chỉ định
    if video:                                  
        image_dir = FRAMES_DIR / f"{Path(video).stem}_{run_id}"
        extract_frames(video, image_dir, fps)
        return image_dir, "video"

    # Trường hợp 2: Người dùng upload tập hợp nhiều ảnh tĩnh
    if images:
        image_dir = INPUT_DIR / f"upload_{run_id}"
        image_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(images):
            name = Path(src).name
            if (image_dir / name).exists():     # 2 ảnh trùng tên -> thêm số thứ tự để không ghi đè
                name = f"{i:03d}_{name}"
            shutil.copy(src, image_dir / name)
        return image_dir, "images"

    # Trường hợp không đủ dữ liệu đầu vào -> Bắn lỗi Gradio thông báo cho người dùng
    raise gr.Error("Cần upload 1 video hoặc ít nhất 3 ảnh.")


def keep_good_images(results, src_dir, work_dir):
    """Lọc và copy các ảnh đạt chất lượng 'Good' sang thư mục riêng.
    
    Giúp COLMAP chỉ xử lý những ảnh đạt tiêu chuẩn (loại bỏ ảnh mờ, tối, trùng lặp).
    """
    good_dir = work_dir / "images_good"
    good_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if r["status"] == "Good":
            shutil.copy(src_dir / r["name"], good_dir / r["name"])
    return good_dir


def export_sparse_ply(sparse_dir, out_ply):
    """Xuất trực tiếp mô hình mây điểm thưa (Sparse Point Cloud) ra file định dạng .PLY.
    
    Giúp người dùng xem kết quả 3D ngay lập tức trên UI mà không phải chờ đợi quá trình 
    Dense Reconstruction mất nhiều thời gian (~10 phút).
    """
    if out_ply.exists():
        out_ply.unlink()
    pycolmap.Reconstruction(sparse_dir).export_PLY(str(out_ply))
    return out_ply


# HÀM ĐIỀU PHỐI CHÍNH (PIPELINE CALLBACK)

def run_pipeline(video, images, fps, dense, good_only, matcher, progress=gr.Progress()):
    """Callback chính của ứng dụng: 
    Video/Ảnh đầu vào -> Kiểm tra chất lượng -> Chạy COLMAP SfM -> Xuất file PLY + Lưu số liệu.

    matcher: "sift" | "superpoint" | "aliked" — cách ghép cặp ảnh.
    """
    logs = []

    def log(msg):
        """Hàm ghi log nội bộ, đồng thời ép đẩy dữ liệu ra terminal ngay lập tức (flush).
        
        Lưu ý: COLMAP ghi log trực tiếp qua luồng stderr, việc dùng flush giúp log không bị tráo đổi thứ tự.
        """
        logs.append(str(msg))
        print(msg, flush=True)      
        if "[dense]" in str(msg):   # dense ~10 phút: cập nhật thanh tiến trình để UI không "đứng"
            progress(0.5, desc="Dense: patch-match stereo (chậm nhất)")

    # Tạo định danh phiên chạy dựa trên thời gian thực (YYYYMMDD-HHMMSS-mmm)
    # (thêm mili-giây để 2 request trong cùng 1 giây không ghi đè work_dir của nhau)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    work_dir = OUTPUT_DIR / run_id

    try:
        # Bước 1: Chuẩn bị ảnh từ video hoặc file upload

        progress(0.02, desc="Chuẩn bị ảnh đầu vào")
        image_dir, source = prepare_images(video, images, fps, run_id)
        log(f"[input] {source}: {image_dir}")


        # Bước 2: Kiểm tra chất lượng ảnh tự động (Blur, Brightness, Duplicate)
 
        progress(0.12, desc="Kiểm tra chất lượng ảnh")
        results = check_images(image_dir)
        quality = count_status(results)
        log("[quality] " + ", ".join(f"{k}={v}" for k, v in quality.items() if v))

        colmap_input = image_dir
        # Nếu bật tùy chọn chỉ dùng ảnh Good -> Lọc và kiểm tra số lượng tối thiểu
        if good_only:
            colmap_input = keep_good_images(results, image_dir, work_dir)
            n_good = len(list(colmap_input.iterdir()))
            log(f"[quality] dùng {n_good}/{len(results)} ảnh Good")
            if n_good < 3:
                raise gr.Error("Còn dưới 3 ảnh Good — không đủ để tái tạo. Hãy quay video nét hơn.")

        
        
        # Bước 3: Thực thi thuật toán COLMAP (Trích điểm -> Khớp cặp -> Tái tạo Sparse/Dense)

        progress(0.2, desc=f"COLMAP: {matcher} -> matching -> sparse")
        stats = run_colmap(colmap_input, work_dir, dense=dense, matcher=matcher, log=log)
        if stats.get("inlier_ratio") is not None:
            log(f"[matcher] {stats['matcher']}: inlier {stats['inlier_ratio']:.1%} "
                f"({stats['inlier_matches']}/{stats['raw_matches']})")

        # Cảnh báo sớm khi quá nửa số ảnh không đăng ký được (thiếu overlap / ít texture)
        rate = stats["images_registered"] / max(stats["images_input"], 1)
        if rate < 0.5:
            log(f"[canh bao] chỉ {rate:.0%} ảnh đăng ký được")


        # Bước 4: Xuất file định dạng 3D PLY để hiển thị trực quan

        progress(0.85, desc="Xuất point cloud")
        if dense:
            ply = Path(stats["dense_ply"])
        else:
            ply = export_sparse_ply(stats["sparse_dir"], work_dir / "sparse.ply")
        log(f"[ply] {ply} ({ply.stat().st_size / 1e6:.1f} MB)")


        # Bước 5: Lưu trữ số liệu thống kê phiên chạy vào file JSONL lịch sử

        record = save_run(make_record(stats, quality, source, run_id, ply))

        # Hoàn tất pipeline, chuẩn bị dữ liệu trả về cho giao diện Gradio
        progress(1.0, desc="Xong")
        stats_df = pd.DataFrame(record.items(), columns=["Chỉ số", "Giá trị"])
        return str(ply), stats_df, pd.DataFrame(results), "\n".join(logs), str(work_dir / "stats.json")

    except gr.Error:
        raise
    except Exception as exc:
        log(f"[LOI] {exc}")
        return None, pd.DataFrame(), pd.DataFrame(), "\n".join(logs), None


# THIẾT KẾ GIAO DIỆN WEB (GRADIO BLOCKS UI)

with gr.Blocks(title="FreeAI 3D Scanner") as demo:
    gr.Markdown("## FreeAI 3D Scanner\nUpload **video** hoặc **nhiều ảnh** -> point cloud 3D bằng COLMAP.")

    with gr.Row():
        # Cột bên trái: Bảng điều khiển, cấu hình thông số đầu vào và nút bấm kích hoạt
        with gr.Column(scale=1):
            video_in = gr.Video(label="Video (ưu tiên)", sources=["upload"])
            images_in = gr.File(label="Hoặc chọn nhiều ảnh",
                                file_count="multiple", file_types=["image"])
            with gr.Row():
                fps_in = gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="FPS trích frame")
                dense_in = gr.Checkbox(value=False, label="Dense (~10 phút)")
            good_in = gr.Checkbox(value=True, label="Chỉ dùng ảnh Good")
            # Matcher ghép cặp: SIFT (mặc định, nhanh) hoặc mạng nơ-ron (chậm hơn nhưng bền hơn)
            matcher_in = gr.Dropdown(
                choices=["sift", "superpoint", "aliked"],
                value="sift",
                label="Matcher",
                info="SIFT: nhanh | SuperPoint/ALIKED + LightGlue: mạng nơ-ron (2.1)",
            )
            btn = gr.Button("Tạo point cloud", variant="primary")

        # Cột bên phải: Hiển thị kết quả trực quan (Mô hình 3D, Bảng số liệu, Log hệ thống)
        with gr.Column(scale=2):
            model_out = gr.Model3D(label="Point cloud", height=420)
            file_out = gr.File(label="stats.json")
            with gr.Accordion("Số liệu", open=True):
                stats_out = gr.Dataframe(interactive=False)
            with gr.Accordion("Chất lượng ảnh", open=False):
                quality_out = gr.Dataframe(interactive=False)
            log_out = gr.Textbox(label="Log", lines=10, max_lines=10, autoscroll=True)

    # Đăng ký sự kiện (Event Listener): Khi bấm nút, gọi hàm run_pipeline và truyền nhận dữ liệu giữa UI
    btn.click(
        fn=run_pipeline,
        inputs=[video_in, images_in, fps_in, dense_in, good_in, matcher_in],
        outputs=[model_out, stats_out, quality_out, log_out, file_out],
    )

if __name__ == "__main__":
    # Khởi chạy ứng dụng Web Server local (mặc định tại http://127.0.0.1:7860)
    demo.launch()

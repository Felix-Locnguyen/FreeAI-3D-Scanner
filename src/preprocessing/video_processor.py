"""Trich frame tu video bang FFmpeg.

Chay thu:
    python src/preprocessing/video_processor.py input.mp4
    python src/preprocessing/video_processor.py input.mp4 data/frames/test 2
"""
import subprocess
import sys 
from pathlib import Path 

DEFAULT_FPS = 2.0  # moi giay lay 2 khung hinh

def extract_frames(video_path, out_dir, fps = DEFAULT_FPS):
    # tach video thanh jpg
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"khong tim thay video: {video_path}")

    out_dir.mkdir(parents=True, exist_ok= True)

    # Xoa frame cu: doi fps ma khong xoa se lam sai so frame
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()

    # -vf fps=2  : cu 1 giay lay 2 khung
    # -q:v 2     : chat luong JPG (2 la rat cao, 31 la thap nhat)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(out_dir / "frame_%04d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output= True, text= True)

    if result.returncode != 0: 
        raise RuntimeError("FFmpeg loi:\n" + result.stderr[-800:])
    
    return sorted(out_dir.glob("frame_*.jpg"))



# < == Main == >
def main(video_path, out_dir=None, fps=DEFAULT_FPS):
    video_path = Path(video_path)
    # Nếu không truyền out_dir, tự động lưu vào: data/frames/<tên_video>
    out_dir = Path(out_dir) if out_dir else Path("data/frames") / video_path.stem

    print(f"Đang xử lý video: {video_path}")
    print(f"Thư mục đầu ra  : {out_dir}")
    print(f"Tốc độ FPS      : {fps}\n")

    try:
        frames = extract_frames(video_path, out_dir, fps)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"LỖI: {exc}")
        return 1

    # In thông tin tổng kết
    print("----------------------------------------")
    print("Trích xuất thành công!")
    print(f"Tổng số frame : {len(frames)}")
    if frames:
        print(f"Frame đầu tiên: {frames[0].name}")
        print(f"Frame cuối cùng: {frames[-1].name}")

    return 0


if __name__ == "__main__":
    # Terminal: python src/preprocessing/video_processor.py <video> [thu_muc_ra] [fps]
    # Bam Run (khong tham so): dung duong dan mac dinh duoi day
    if len(sys.argv) > 1:
        sys.exit(main(
            sys.argv[1],
            sys.argv[2] if len(sys.argv) > 2 else None,
            float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_FPS,
        ))
    sys.exit(main("data/input/test_video.mp4", "data/frames/test_video"))
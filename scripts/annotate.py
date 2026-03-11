"""Extract frames from a video file at a given FPS for annotation."""

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FRAME_FPS)
    frame_interval = max(1, int(round(video_fps / fps)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    saved = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            filename = out / f"{stem}_{idx:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {total} total (every {frame_interval} frames) → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for annotation")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("-f", "--fps", type=float, default=2.0, help="Extraction rate in FPS (default: 2.0)")
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps)

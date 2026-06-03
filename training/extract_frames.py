"""
Pre-saves labeled video frames as JPEG images for fast fine-tuning.
Run once before finetune_backbone.py:
    python3 -m training.extract_frames
"""
import os
import cv2
import torch

VIDEO_DIR = "data/raw_videos"
LABELS_DIR = "data/labels"
FRAMES_DIR = "data/frames"

os.makedirs(FRAMES_DIR, exist_ok=True)

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
print(f"Found {len(video_files)} videos\n")

for vid_idx, fname in enumerate(video_files, 1):
    stem = fname.replace(".mp4", "")
    label_path = os.path.join(LABELS_DIR, stem + ".pt")
    if not os.path.exists(label_path):
        print(f"[{vid_idx}/{len(video_files)}] {fname}  no labels, skipping")
        continue

    out_dir = os.path.join(FRAMES_DIR, stem)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"[{vid_idx}/{len(video_files)}] {fname}  already extracted, skipping")
        continue

    os.makedirs(out_dir, exist_ok=True)
    labels = torch.load(label_path)

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, fname))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(fps)))

    i = 0
    frame_idx = 0
    saved = 0
    success, frame = cap.read()

    while success:
        if i % step == 0:
            if frame_idx < labels.shape[0] and labels[frame_idx].item() != -100:
                out_path = os.path.join(out_dir, f"{frame_idx:06d}_{labels[frame_idx].item()}.jpg")
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved += 1
            frame_idx += 1
        success, frame = cap.read()
        i += 1

    cap.release()
    print(f"[{vid_idx}/{len(video_files)}] {fname}  saved {saved} frames")

print("\nDone.")

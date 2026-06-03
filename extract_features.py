import argparse
import os
import cv2
import torch
from torchvision import transforms
from models.cnn_backbone import FrameBackbone

VIDEO_DIR = "data/raw_videos"
OUT_DIR = "data/features"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default=VIDEO_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--cnn_checkpoint", default=None,
                        help="Trained FrameBackbone weights (e.g. checkpoints/baseline_cnn.pt)")
    parser.add_argument("--num_classes", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cudnn.enabled = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = FrameBackbone(num_classes=args.num_classes).to(device)
    if args.cnn_checkpoint:
        model.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device))
        print(f"Loaded CNN checkpoint: {args.cnn_checkpoint}")
    else:
        print("Using ImageNet ResNet-50 (no --cnn_checkpoint)")
    model.eval()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    label_stems = {f.replace(".pt", "") for f in os.listdir(args.labels_dir) if f.endswith(".pt")}
    video_files = [
        f for f in os.listdir(args.video_dir)
        if f.endswith(".mp4") and f.replace(".mp4", "") in label_stems
    ]
    print(f"Found {len(video_files)} videos with matching labels\n")

    with torch.no_grad():
        for vid_idx, fname in enumerate(video_files, 1):
            path = os.path.join(args.video_dir, fname)
            cap = cv2.VideoCapture(path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            step = max(1, int(round(fps)))
            expected = total_frames // step

            print(f"[{vid_idx}/{len(video_files)}] {fname}  ({expected} frames to extract)")

            feats = []
            logits_list = []
            i = 0
            extracted = 0
            success, frame = cap.read()

            while success:
                if i % step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    x = tfm(frame_rgb).unsqueeze(0).to(device)
                    logit, feat = model(x, return_features=True)
                    feats.append(feat.cpu())
                    logits_list.append(logit.cpu())
                    extracted += 1

                    if extracted % 100 == 0:
                        print(f"  {extracted}/{expected} frames", flush=True)

                success, frame = cap.read()
                i += 1

            cap.release()

            features = torch.cat(feats, dim=0)
            cnn_logits = torch.cat(logits_list, dim=0)
            out_name = fname.replace(".mp4", ".pt")
            out_path = os.path.join(args.out_dir, out_name)
            torch.save({"features": features, "cnn_logits": cnn_logits}, out_path)
            print(f"  Done -> {out_name}  features={tuple(features.shape)}  logits={tuple(cnn_logits.shape)}\n")

    print("All videos processed.")


if __name__ == "__main__":
    main()

"""
Add frozen CNN logits to legacy feature .pt files (tensor-only format).

Usage:
    python train_baseline.py --epochs 20
    python precompute_cnn_logits.py --cnn_checkpoint checkpoints/baseline_cnn.pt
"""

import argparse
import os
import torch

from models.cnn_backbone import FrameBackbone
from utils.attention_reg import load_feature_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--cnn_checkpoint", required=True)
    parser.add_argument("--num_classes", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrameBackbone(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    files = [f for f in os.listdir(args.features_dir) if f.endswith(".pt")]
    print(f"Processing {len(files)} feature files...")

    for fname in files:
        path = os.path.join(args.features_dir, fname)
        features, existing_logits = load_feature_bundle(path)

        if existing_logits is not None:
            print(f"  {fname}: already has cnn_logits, skipping")
            continue

        with torch.no_grad():
            logits = model.classifier(features.to(device)).cpu()

        torch.save(
            {"features": features, "cnn_logits": logits},
            path,
        )
        print(f"  {fname}: features {tuple(features.shape)} + logits {tuple(logits.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()

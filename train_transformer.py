import argparse
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import FeatureDataset, collate_variable_length
from data.splits import build_samples, train_val_split
from models.cnn_backbone import FrameBackbone
from models.opera_transformer import NeuroOperA
from utils.attention_reg import (
    attention_regularization_loss,
    cnn_logits_from_features,
    compute_cee,
    load_feature_bundle,
    normalized_frame_attention,
)
from utils.class_weights import compute_class_weights


CLASS_NAMES = [
    "Brain Exposure",
    "Parent Vessel ID",
    "Neck ID",
    "Dome ID",
    "Clipping",
]


def _acc(logits, labels):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100
    correct = (flat_logits[mask].argmax(1) == flat_labels[mask]).sum().item()
    return correct, mask.sum().item()


def _per_class_stats(logits, labels, num_classes):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    preds = flat_logits.argmax(1)
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for c in range(num_classes):
        tp[c] = ((preds == c) & (flat_labels == c)).sum()
        fp[c] = ((preds == c) & (flat_labels != c)).sum()
        fn[c] = ((preds != c) & (flat_labels == c)).sum()
    return tp, fp, fn


def _resolve_cnn_logits(features, cnn_logits, frozen_classifier, device):
    if cnn_logits is not None:
        return cnn_logits.to(device)
    if frozen_classifier is not None:
        return cnn_logits_from_features(features.to(device), frozen_classifier)
    return None


def _compute_losses(
    logits,
    all_attn,
    labels,
    padding_mask,
    class_weights,
    cnn_logits,
    lambda_reg,
):
    lc = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        weight=class_weights,
        ignore_index=-100,
    )

    lreg = torch.tensor(0.0, device=logits.device)
    if lambda_reg > 0 and cnn_logits is not None and len(all_attn) > 0:
        n = normalized_frame_attention(all_attn[0], padding_mask)
        cee = compute_cee(cnn_logits, labels)
        lreg = attention_regularization_loss(n, cee, labels)

    loss = lc + lambda_reg * lreg
    return loss, lc, lreg


def train_one_epoch(
    model, loader, optimizer, device, class_weights=None, lambda_reg=1.0, frozen_classifier=None,
):
    model.train()
    total_loss, total_lc, total_lreg = 0, 0, 0
    correct, total = 0, 0
    for features, labels, padding_mask, cnn_logits in loader:
        features = features.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)
        cnn_logits = _resolve_cnn_logits(features, cnn_logits, frozen_classifier, device)

        logits, all_attn = model(features, padding_mask)
        loss, lc, lreg = _compute_losses(
            logits, all_attn, labels, padding_mask, class_weights, cnn_logits, lambda_reg,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_lc += lc.item()
        total_lreg += lreg.item()
        c, n = _acc(logits, labels)
        correct += c
        total += n

    n_batches = len(loader)
    return (
        total_loss / n_batches,
        total_lc / n_batches,
        total_lreg / n_batches,
        correct / total,
    )


@torch.no_grad()
def eval_one_epoch(model, loader, device, class_weights=None, lambda_reg=0.0, frozen_classifier=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    num_classes = model.classifier.out_features
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for features, labels, padding_mask, cnn_logits in loader:
        features = features.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)
        cnn_logits = _resolve_cnn_logits(features, cnn_logits, frozen_classifier, device)

        logits, all_attn = model(features, padding_mask)
        loss, _, _ = _compute_losses(
            logits, all_attn, labels, padding_mask, class_weights, cnn_logits, lambda_reg=0.0,
        )
        total_loss += loss.item()
        c, n = _acc(logits, labels)
        correct += c
        total += n
        btp, bfp, bfn = _per_class_stats(logits.cpu(), labels.cpu(), num_classes)
        tp += btp
        fp += bfp
        fn += bfn

    per_class = []
    for c in range(num_classes):
        acc = tp[c].item() / (tp[c] + fn[c]).item() if (tp[c] + fn[c]) > 0 else float("nan")
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1 = (2 * tp[c] / denom).item() if denom > 0 else float("nan")
        per_class.append((acc, f1))

    return total_loss / len(loader), correct / total, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--lambda_reg", type=float, default=1.0,
                        help="OperA attention regularization weight (0 to disable)")
    parser.add_argument("--cnn_checkpoint", default=None,
                        help="Frozen CNN for CEE when feature files lack cnn_logits")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    frozen_classifier = None
    if args.cnn_checkpoint:
        cnn = FrameBackbone(num_classes=args.num_classes)
        cnn.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device))
        cnn.eval()
        for p in cnn.parameters():
            p.requires_grad = False
        frozen_classifier = cnn.classifier.to(device)
        print(f"Loaded frozen CNN for CEE: {args.cnn_checkpoint}")

    samples = build_samples(args.features_dir, args.labels_dir)
    print(f"Found {len(samples)} videos with features+labels")

    train_samples, val_samples = train_val_split(samples)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")
    print(f"lambda_reg={args.lambda_reg}")
    if args.lambda_reg > 0:
        has_logits_in_files = any(
            load_feature_bundle(s["features"])[1] is not None for s in samples[:3]
        )
        if not has_logits_in_files and not args.cnn_checkpoint:
            print(
                "WARNING: lambda_reg > 0 but no cnn_logits in features and no --cnn_checkpoint. "
                "L_reg will be 0. Run precompute_cnn_logits.py or pass --cnn_checkpoint."
            )

    class_weights = compute_class_weights(train_samples, args.num_classes, device)

    train_loader = DataLoader(
        FeatureDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
    )
    val_loader = DataLoader(
        FeatureDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
    )

    model = NeuroOperA(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_lc, train_lreg, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            class_weights=class_weights,
            lambda_reg=args.lambda_reg,
            frozen_classifier=frozen_classifier,
        )
        val_loss, val_acc, per_class = eval_one_epoch(model, val_loader, device)

        reg_str = f"  lreg={train_lreg:.4f}" if args.lambda_reg > 0 else ""
        print(
            f"Epoch {epoch:03d}  train loss={train_loss:.4f} (lc={train_lc:.4f}{reg_str}) acc={train_acc:.3f}  "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )
        for name, (acc, f1) in zip(CLASS_NAMES, per_class):
            acc_str = f"{acc:.3f}" if not math.isnan(acc) else " n/a"
            f1_str = f"{f1:.3f}" if not math.isnan(f1) else " n/a"
            print(f"    {name:<22} acc={acc_str}  f1={f1_str}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "best_transformer.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()

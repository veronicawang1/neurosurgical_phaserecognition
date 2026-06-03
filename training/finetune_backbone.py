"""
Fine-tunes the ResNet50 backbone on pre-saved frames (run extract_frames.py first).
After running this, re-extract features with the fine-tuned checkpoint:
    python3 extract_features.py --checkpoint checkpoints/finetuned_backbone.pt
"""
import argparse
import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from models.cnn_backbone import FrameBackbone


class FrameImageDataset(Dataset):
    def __init__(self, frames_dir, transform):
        self.samples = []
        self.transform = transform
        for video_dir in os.listdir(frames_dir):
            full_dir = os.path.join(frames_dir, video_dir)
            if not os.path.isdir(full_dir):
                continue
            for fname in os.listdir(full_dir):
                if not fname.endswith(".jpg"):
                    continue
                label = int(fname.split("_")[-1].replace(".jpg", ""))
                self.samples.append((os.path.join(full_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frame = cv2.imread(path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame), torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", default="data/frames")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading dataset...")
    full_dataset = FrameImageDataset(args.frames_dir, train_tfm)
    print(f"Total frames: {len(full_dataset)}")

    n_val = max(1, int(len(full_dataset) * 0.2))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    val_set.dataset.transform = val_tfm

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Train: {n_train}  Val: {n_val}")

    model = FrameBackbone(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args.lr_backbone},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ])

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            logits, _ = model(frames, return_features=True)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                logits, _ = model(frames, return_features=True)
                val_loss += F.cross_entropy(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss/len(train_loader):.4f} acc={correct/total:.3f} | "
            f"val loss={val_loss/len(val_loader):.4f} acc={val_correct/val_total:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "finetuned_backbone.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()

"""Train and evaluate a small CNN on CIFAR-10.

Designed to run on a laptop (CPU) for a quick smoke test, or inside a
Kubernetes pod/job on Nautilus where the data and outputs live on a PVC.

Examples
--------
Quick smoke test (small subset, 1 epoch):
    python main.py --epochs 1 --subset 2000

Full run with data and results on the PVC:
    python main.py --data-dir /data/datasets --out-dir /data/outputs --epochs 10
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


class SmallCNN(nn.Module):
    """Three conv blocks followed by a two-layer classifier."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8   -> 4x4
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def get_loaders(data_dir: str, batch_size: int, subset: int, workers: int):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    if subset:
        train_set = Subset(train_set, range(min(subset, len(train_set))))
        test_set = Subset(test_set, range(min(subset // 5 or 1, len(test_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 CNN training")
    p.add_argument("--data-dir", default="./data", help="where CIFAR-10 is downloaded/stored")
    p.add_argument("--out-dir", default="./outputs", help="where the model and metrics are saved")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset", type=int, default=0, help="use only N training images (0 = all)")
    p.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.subset, args.workers)
    model = SmallCNN(len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "test_loss": te_loss, "test_acc": te_acc})
        print(f"Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| test loss {te_loss:.4f} acc {te_acc:.3f}")
    elapsed = time.time() - start

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "cifar10_cnn.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"device": str(device), "seconds": round(elapsed, 1),
                   "args": vars(args), "history": history}, f, indent=2)
    print(f"Done in {elapsed:.1f}s. Saved model and metrics to {out_dir.resolve()}")


if __name__ == "__main__":
    main()

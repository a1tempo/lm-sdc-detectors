"""Train the ResNet-18 EuroSAT classifier and save the checkpoint and test indices.

Follows Reem's notebook (notebooks/reem_TFLite_EuroSAT_ResNet18.ipynb): ImageNet
weights, the seed-42 80/10/10 split, Adam at lr 1e-4, batch size 32, 5 epochs.

Two differences, both deliberate:
  * The training transform (random horizontal flip) really is applied. In the notebook
    all subsets share one dataset object, so training ran with the eval transform.
  * Training is seeded, so the shuffle order is repeatable. On a GPU the result can
    still differ slightly between runs.

Refuses to replace an existing checkpoint: every later result depends on it.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models

from sdc.config import load_config
from sdc.data.eurosat import CLASS_NAMES, EVAL_TRANSFORM, TRAIN_TRANSFORM
from sdc.model.resnet18 import NUM_CLASSES, checkpoint_sha256

BATCH_SIZE = 32
LEARNING_RATE = 1e-4


def split_indices(num_images: int, fractions, seed: int) -> tuple[list[int], list[int], list[int]]:
    """The notebook's train/val/test indices; they depend only on the count and seed."""
    train_size = int(fractions[0] * num_images)
    val_size = int(fractions[1] * num_images)
    parts = random_split(
        range(num_images), [train_size, val_size, num_images - train_size - val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return tuple(list(part.indices) for part in parts)


def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((preds == labels).sum())
            total += len(labels)
    return correct / total


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main(config_path: str | None, epochs: int, device: str, limit: int | None, workers: int) -> None:
    config = load_config(config_path)
    paths, data = config["paths"], config["data"]
    checkpoint, indices_path = Path(paths["checkpoint"]), Path(paths["test_indices"])
    for path in (checkpoint, indices_path):
        if path.exists():
            raise SystemExit(f"{path} already exists; delete it to retrain")

    train_images = datasets.ImageFolder(paths["eurosat_dir"], transform=TRAIN_TRANSFORM)
    eval_images = datasets.ImageFolder(paths["eurosat_dir"], transform=EVAL_TRANSFORM)
    if train_images.classes != CLASS_NAMES:
        raise ValueError(f"expected classes {CLASS_NAMES}, found {train_images.classes}")
    train_idx, val_idx, test_idx = split_indices(len(train_images), data["split_fractions"], data["split_seed"])
    if limit is not None:   # quick checks only: never save weights trained this way for real use
        train_idx, val_idx, test_idx = train_idx[:limit], val_idx[:limit], test_idx[:limit]

    torch.manual_seed(config["seed"])
    loader_args = {"batch_size": BATCH_SIZE, "num_workers": workers, "persistent_workers": workers > 0}
    train_loader = DataLoader(
        Subset(train_images, train_idx), shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]), **loader_args,
    )
    val_loader = DataLoader(Subset(eval_images, val_idx), **loader_args)
    test_loader = DataLoader(Subset(eval_images, test_idx), **loader_args)

    device = pick_device(device)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"training on {device}: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test images")

    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        val_acc = accuracy(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_idx), "val_accuracy": val_acc})
        print(f"epoch {epoch}/{epochs}: loss {history[-1]['train_loss']:.4f}, "
              f"val accuracy {val_acc:.4f} ({time.perf_counter() - start:.0f}s)")
    test_acc = accuracy(model, test_loader, device)
    print(f"test accuracy {test_acc:.4f}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.cpu() for name, value in model.state_dict().items()}
    _write_atomic(checkpoint, lambda f: torch.save(state, f))
    _write_atomic(indices_path, lambda f: f.write(json.dumps(test_idx).encode()))
    info = {
        "checkpoint_sha256": checkpoint_sha256(str(checkpoint)),
        "epochs": epochs, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "seed": config["seed"], "split_seed": data["split_seed"], "limit": limit,
        "augmentation": "random horizontal flip", "init": "ImageNet (ResNet18_Weights.DEFAULT)",
        "device": device, "torch": torch.__version__, "platform": platform.platform(),
        "history": history, "test_accuracy": test_acc,
        "training_seconds": round(time.perf_counter() - start),
    }
    (checkpoint.parent / "training_info.json").write_text(json.dumps(info, indent=2))
    print(f"saved {checkpoint}, {indices_path}, and training_info.json")


def _write_atomic(path: Path, write) -> None:
    temp = path.with_name(f"{path.name}.tmp")
    with open(temp, "wb") as f:
        write(f)
    os.replace(temp, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--limit", type=int, default=None, help="use only this many images per split (quick checks)")
    parser.add_argument("--workers", type=int, default=4, help="data loading processes")
    args = parser.parse_args()
    main(args.config, args.epochs, args.device, args.limit, args.workers)

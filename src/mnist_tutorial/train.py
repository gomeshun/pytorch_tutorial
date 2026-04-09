from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from mnist_tutorial.data import build_dataloaders
from mnist_tutorial.model import MnistClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple MNIST classifier with PyTorch.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for MNIST downloads.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_metrics(prefix: str, loss: float, accuracy: float) -> str:
    return f"{prefix}: loss={loss:.4f} accuracy={accuracy:.2%}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device()
    train_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MnistClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print(f"Using device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs}")
        print(format_metrics("  train", train_loss, train_accuracy))
        print(format_metrics("  test ", test_loss, test_accuracy))


if __name__ == "__main__":
    main()
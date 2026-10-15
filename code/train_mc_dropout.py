#!/usr/bin/env python3
"""Run the PneumoniaMNIST MC Dropout reproducibility experiments.

This script is intended to be executed on a Linux workstation or GPU server.
It downloads PneumoniaMNIST through MedMNIST on the first run, trains the
specified CNN, evaluates deterministic and MC Dropout predictions, and writes
metrics, checkpoints, and plots under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

import medmnist
from medmnist import INFO


@dataclass(frozen=True)
class Config:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 4
    min_delta: float = 1e-3
    mc_passes: int = 30
    c1: int = 32
    c2: int = 64
    c3: int = 128
    hidden_dim: int = 128
    dropout: float = 0.3
    train_size: int = 4700
    validation_size: int = 500
    test_size: int = 600


class PneumoniaCNN(nn.Module):
    """CNN specified in the experiment plan, returning one logit per image."""

    def __init__(
        self,
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * c3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def seed_everything(seed: int) -> None:
    """Set all available RNGs and request deterministic CUDA algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def build_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def load_datasets(
    root: Path,
    config: Config,
) -> tuple[Dataset, Dataset, Dataset]:
    root = root.expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"Cannot create MedMNIST data directory '{root}'. "
            "Choose a directory you can write to with --data-dir."
        ) from error
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
    )
    dataset_class = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])
    datasets = {
        split: dataset_class(split=split, root=str(root), transform=transform, download=True)
        for split in ("train", "val", "test")
    }
    requested = {
        "train": config.train_size,
        "val": config.validation_size,
        "test": config.test_size,
    }
    for split, size in requested.items():
        available = len(datasets[split])
        if size > available:
            raise ValueError(
                f"{split} split has {available} samples, but {size} were requested."
            )
    return tuple(
        Subset(datasets[split], range(requested[split]))
        for split in ("train", "val", "test")
    )  # type: ignore[return-value]


def set_dropout_train_only(model: nn.Module) -> None:
    """Enable Dropout modules while preserving BatchNorm evaluation mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float().view(-1)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * len(targets)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.cpu().numpy())
    y_probability = np.concatenate(probabilities)
    y_true = np.concatenate(labels).astype(np.int64)
    prediction = (y_probability >= 0.5).astype(np.int64)
    accuracy = float(np.mean(prediction == y_true))
    return y_probability, y_true, total_loss / len(y_true)


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    passes: int,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    set_dropout_train_only(model)
    pass_probabilities: list[np.ndarray] = []
    labels: np.ndarray | None = None
    for pass_index in range(1, passes + 1):
        probabilities: list[np.ndarray] = []
        current_labels: list[np.ndarray] = []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            probabilities.append(torch.sigmoid(model(images)).cpu().numpy())
            current_labels.append(targets.view(-1).numpy())
        pass_probabilities.append(np.concatenate(probabilities))
        if labels is None:
            labels = np.concatenate(current_labels).astype(np.int64)
        if progress_label is not None and (
            pass_index == 1 or pass_index == passes or pass_index % 5 == 0
        ):
            print(
                f"[{progress_label}] MC pass {pass_index}/{passes}",
                flush=True,
            )
    assert labels is not None
    return np.stack(pass_probabilities, axis=0), labels


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))


def expected_calibration_error(
    probability: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    prediction = (probability >= 0.5).astype(np.int64)
    confidence = np.maximum(probability, 1.0 - probability)
    correctness = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        in_bin = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1]
        )
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(
                np.mean(confidence[in_bin]) - np.mean(correctness[in_bin])
            )
    return float(ece)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def evaluate_mc(
    pass_probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    mean_probability = np.mean(pass_probabilities, axis=0)
    predictive_entropy = binary_entropy(mean_probability)
    mutual_information = predictive_entropy - np.mean(
        binary_entropy(pass_probabilities), axis=0
    )
    prediction = (mean_probability >= 0.5).astype(np.int64)
    errors = (prediction != labels).astype(np.int64)
    metrics = {
        "accuracy": float(np.mean(prediction == labels)),
        "predictive_entropy": float(np.mean(predictive_entropy)),
        "mutual_information": float(np.mean(mutual_information)),
        "ece": expected_calibration_error(mean_probability, labels),
        "error_detection_auroc_predictive_entropy": safe_auroc(errors, predictive_entropy),
        "error_detection_auroc_mutual_information": safe_auroc(errors, mutual_information),
    }
    return metrics, predictive_entropy, mutual_information


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: Config,
    progress_label: str = "training",
) -> list[dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float().view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
            train_correct += int(((torch.sigmoid(logits) >= 0.5) == targets).sum().item())
            train_count += len(targets)

        validation_probability, validation_labels, validation_loss = predict(
            model, validation_loader, device
        )
        validation_accuracy = float(
            np.mean((validation_probability >= 0.5).astype(np.int64) == validation_labels)
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / train_count,
            "train_accuracy": train_correct / train_count,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            improved = True
        else:
            epochs_without_improvement += 1
            improved = False
        print(
            f"[{progress_label}] epoch {epoch}/{config.epochs} "
            f"train_loss={row['train_loss']:.5f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['validation_loss']:.5f} "
            f"val_acc={row['validation_accuracy']:.4f} "
            f"best_val_loss={best_loss:.5f} "
            f"patience={epochs_without_improvement}/{config.patience}"
            f"{' *best*' if improved else ''}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            print(
                f"[{progress_label}] early stopping at epoch {epoch}; "
                f"best_val_loss={best_loss:.5f}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(
        f"[{progress_label}] training finished after {len(history)} epoch(s); "
        f"restored_best_val_loss={best_loss:.5f}",
        flush=True,
    )
    return history


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(
    output_dir: Path,
    history: list[dict[str, float]],
    uncertainty: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="validation")
    axes[1].set(xlabel="epoch", ylabel="accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(uncertainty["correct"], bins=20, alpha=0.7, label="correct")
    axes[0].hist(uncertainty["incorrect"], bins=20, alpha=0.7, label="incorrect")
    axes[0].set(xlabel="predictive entropy", ylabel="count")
    axes[0].legend()
    axes[1].hist(
        uncertainty["correct_mi"], bins=20, alpha=0.7, label="correct"
    )
    axes[1].hist(
        uncertainty["incorrect_mi"], bins=20, alpha=0.7, label="incorrect"
    )
    axes[1].set(xlabel="mutual information", ylabel="count")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "uncertainty_histograms.png", dpi=150)
    plt.close(figure)

    if np.unique(uncertainty["errors"]).size == 2:
        figure, axis = plt.subplots(figsize=(6, 5))
        for score, label in (
            (uncertainty["entropy_all"], "predictive entropy"),
            (uncertainty["mi_all"], "mutual information"),
        ):
            false_positive_rate, true_positive_rate, _ = roc_curve(
                uncertainty["errors"], score
            )
            axis.plot(false_positive_rate, true_positive_rate, label=label)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set(xlabel="false positive rate", ylabel="true positive rate")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "error_detection_auroc.png", dpi=150)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[5, 38, 42, 84, 112])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.epochs < 1 or args.mc_passes < 1:
        raise ValueError("--repeats, --epochs, and --mc-passes must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    config = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        mc_passes=args.mc_passes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": asdict(config), "device": str(device)}, handle, indent=2)
    print(
        f"Starting experiment on device={device}; "
        f"seeds={args.seeds}; repeats_per_seed={args.repeats}; "
        f"epochs={config.epochs}; mc_passes={config.mc_passes}",
        flush=True,
    )
    print(f"Loading PneumoniaMNIST data from {args.data_dir}...", flush=True)
    train_set, validation_set, test_set = load_datasets(args.data_dir, config)
    print(
        f"Loaded train={len(train_set)}, validation={len(validation_set)}, "
        f"test={len(test_set)} samples.",
        flush=True,
    )
    all_metrics: list[dict[str, object]] = []
    seed_representatives: list[dict[str, object]] = []

    for seed_index, seed in enumerate(args.seeds, start=1):
        seed_metrics: list[dict[str, object]] = []
        for repeat in range(1, args.repeats + 1):
            progress_label = (
                f"seed {seed} ({seed_index}/{len(args.seeds)}), "
                f"repeat {repeat}/{args.repeats}"
            )
            print(f"[{progress_label}] starting training", flush=True)
            seed_everything(seed)
            train_loader = build_loader(train_set, config.batch_size, seed, shuffle=True)
            validation_loader = build_loader(validation_set, config.batch_size, seed, shuffle=False)
            test_loader = build_loader(test_set, config.batch_size, seed, shuffle=False)
            model = PneumoniaCNN(
                c1=config.c1,
                c2=config.c2,
                c3=config.c3,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            ).to(device)
            history = train_one(
                model,
                train_loader,
                validation_loader,
                device,
                config,
                progress_label=progress_label,
            )
            test_probability, labels, test_loss = predict(model, test_loader, device)
            print(
                f"[{progress_label}] deterministic test evaluation: "
                f"loss={test_loss:.5f} "
                f"accuracy={np.mean((test_probability >= 0.5).astype(np.int64) == labels):.4f}",
                flush=True,
            )
            seed_everything(seed)
            pass_probabilities, labels = mc_predict(
                model,
                test_loader,
                device,
                config.mc_passes,
                progress_label=progress_label,
            )
            metrics, predictive_entropy, mutual_information = evaluate_mc(
                pass_probabilities, labels
            )
            metrics.update(
                {
                    "seed": seed,
                    "repeat": repeat,
                    "loss": test_loss,
                    "deterministic_accuracy": float(
                        np.mean((test_probability >= 0.5).astype(np.int64) == labels)
                    ),
                }
            )
            print(
                f"[{progress_label}] completed: "
                f"accuracy={metrics['accuracy']:.4f} "
                f"ece={metrics['ece']:.5f} "
                f"error_auroc={metrics['error_detection_auroc_predictive_entropy']:.4f}",
                flush=True,
            )
            all_metrics.append(metrics)
            seed_metrics.append(metrics)
            run_dir = args.output_dir / f"seed_{seed}" / f"repeat_{repeat}"
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), run_dir / "model.pt")
            write_csv(run_dir / "history.csv", history)
            np.savez_compressed(
                run_dir / "mc_predictions.npz",
                probabilities=pass_probabilities,
                labels=labels,
                predictive_entropy=predictive_entropy,
                mutual_information=mutual_information,
            )
            if not args.skip_plots:
                correct = (test_probability >= 0.5).astype(np.int64) == labels
                plot_outputs(
                    run_dir,
                    history,
                    {
                        "correct": predictive_entropy[correct],
                        "incorrect": predictive_entropy[~correct],
                        "correct_mi": mutual_information[correct],
                        "incorrect_mi": mutual_information[~correct],
                        "errors": (~correct).astype(np.int64),
                        "entropy_all": predictive_entropy,
                        "mi_all": mutual_information,
                    },
                )
        representative: dict[str, object] = {"seed": seed}
        for metric in all_metrics[0]:
            if metric in {"seed", "repeat"}:
                continue
            values = np.asarray([float(row[metric]) for row in seed_metrics], dtype=float)
            representative[f"{metric}_mean"] = float(np.nanmean(values))
            representative[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        seed_representatives.append(representative)
        print(
            f"[seed {seed}] repeats completed: "
            f"accuracy_mean={representative['accuracy_mean']:.4f} "
            f"accuracy_std={representative['accuracy_std']:.4f}",
            flush=True,
        )

    write_csv(args.output_dir / "run_metrics.csv", all_metrics)
    write_csv(args.output_dir / "seed_representatives.csv", seed_representatives)
    experiment_summary: list[dict[str, object]] = []
    for metric in seed_representatives[0]:
        if metric == "seed" or not metric.endswith("_mean"):
            continue
        values = np.asarray([float(row[metric]) for row in seed_representatives], dtype=float)
        experiment_summary.append(
            {
                "metric": metric.removesuffix("_mean"),
                "mean_across_seeds": float(np.nanmean(values)),
                "variance_across_seeds": float(np.nanvar(values)),
                "std_across_seeds": float(np.nanstd(values)),
            }
        )
    write_csv(args.output_dir / "experiment_summary.csv", experiment_summary)
    print(
        f"Completed {len(all_metrics)} runs on {device}. Results: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

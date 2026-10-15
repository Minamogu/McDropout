#!/usr/bin/env python3
"""Run one training and evaluation run of the MC Dropout experiment."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from train_mc_dropout import (
    Config,
    PneumoniaCNN,
    build_loader,
    evaluate_mc,
    load_datasets,
    mc_predict,
    plot_outputs,
    predict,
    seed_everything,
    train_one,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("single_run"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def train_with_progress(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
    label: str,
) -> list[dict[str, float]]:
    """Support both the original and progress-enabled training modules."""
    if "progress_label" in inspect.signature(train_one).parameters:
        return train_one(
            model,
            train_loader,
            validation_loader,
            device,
            config,
            progress_label=label,
        )
    print(
        f"[{label}] train_mc_dropout.py is an older version; "
        "epoch progress may be unavailable.",
        flush=True,
    )
    return train_one(model, train_loader, validation_loader, device, config)


def mc_predict_with_progress(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    passes: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Support both the original and progress-enabled MC prediction modules."""
    if "progress_label" in inspect.signature(mc_predict).parameters:
        return mc_predict(
            model,
            test_loader,
            device,
            passes,
            progress_label=label,
        )
    print(
        f"[{label}] train_mc_dropout.py is an older version; "
        "MC pass progress may be unavailable.",
        flush=True,
    )
    return mc_predict(model, test_loader, device, passes)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.mc_passes < 1:
        raise ValueError("--epochs and --mc-passes must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    config = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        mc_passes=args.mc_passes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config.__dict__,
                "device": str(device),
                "seed": args.seed,
            },
            handle,
            indent=2,
        )

    label = f"single run, seed {args.seed}"
    print(f"[{label}] device={device}", flush=True)
    print(f"[{label}] loading data from {args.data_dir}", flush=True)
    train_set, validation_set, test_set = load_datasets(args.data_dir, config)
    print(
        f"[{label}] loaded train={len(train_set)}, validation={len(validation_set)}, "
        f"test={len(test_set)}",
        flush=True,
    )

    seed_everything(args.seed)
    train_loader = build_loader(train_set, config.batch_size, args.seed, shuffle=True)
    validation_loader = build_loader(
        validation_set, config.batch_size, args.seed, shuffle=False
    )
    test_loader = build_loader(test_set, config.batch_size, args.seed, shuffle=False)
    model = PneumoniaCNN(
        c1=config.c1,
        c2=config.c2,
        c3=config.c3,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    history = train_with_progress(
        model,
        train_loader,
        validation_loader,
        device,
        config,
        label,
    )
    test_probability, labels, test_loss = predict(model, test_loader, device)
    deterministic_accuracy = float(
        np.mean((test_probability >= 0.5).astype(np.int64) == labels)
    )
    print(
        f"[{label}] deterministic test: loss={test_loss:.5f}, "
        f"accuracy={deterministic_accuracy:.4f}",
        flush=True,
    )

    seed_everything(args.seed)
    pass_probabilities, labels = mc_predict_with_progress(
        model,
        test_loader,
        device,
        config.mc_passes,
        label,
    )
    metrics, predictive_entropy, mutual_information = evaluate_mc(
        pass_probabilities, labels
    )
    metrics.update(
        {
            "seed": args.seed,
            "repeat": 1,
            "loss": test_loss,
            "deterministic_accuracy": deterministic_accuracy,
        }
    )
    write_csv(args.output_dir / "metrics.csv", [metrics])
    write_csv(args.output_dir / "history.csv", history)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    np.savez_compressed(
        args.output_dir / "mc_predictions.npz",
        probabilities=pass_probabilities,
        labels=labels,
        predictive_entropy=predictive_entropy,
        mutual_information=mutual_information,
    )

    if not args.skip_plots:
        correct = (test_probability >= 0.5).astype(np.int64) == labels
        plot_outputs(
            args.output_dir,
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
    print(
        f"[{label}] completed: accuracy={metrics['accuracy']:.4f}, "
        f"ece={metrics['ece']:.5f}, "
        f"error_auroc={metrics['error_detection_auroc_predictive_entropy']:.4f}",
        flush=True,
    )
    print(f"[{label}] results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

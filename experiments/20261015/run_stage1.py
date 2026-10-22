#!/usr/bin/env python3
"""Run stage 1 of the 2026-10-15 PneumoniaMNIST experiment.

Stage 1 runs every architecture/dropout condition once for each planned seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train_mc_dropout import (  # noqa: E402
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


CONDITIONS: tuple[dict[str, object], ...] = (
    {"experiment": "channels", "condition": "A", "c1": 16, "c2": 32, "c3": 64},
    {"experiment": "channels", "condition": "B_baseline", "c1": 32, "c2": 64, "c3": 128},
    {"experiment": "channels", "condition": "C", "c1": 64, "c2": 128, "c3": 256},
    {"experiment": "dropout", "condition": "A", "dropout": 0.1},
    {"experiment": "dropout", "condition": "B_baseline", "dropout": 0.3},
    {"experiment": "dropout", "condition": "C", "dropout": 0.5},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[5, 38, 42],
        help="Seeds to run. Defaults to all seeds in the experiment plan.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run only this seed (compatibility shortcut for a single-seed check).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def save_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def run_condition(
    condition: dict[str, object],
    seed: int,
    args: argparse.Namespace,
    base_config: Config,
    datasets: tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset],
    device: torch.device,
) -> dict[str, object]:
    experiment = str(condition["experiment"])
    name = str(condition["condition"])
    label = f"{experiment}/{name}, seed={seed}"
    run_dir = args.output_dir / experiment / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_values = asdict(base_config)
    config_values.update(
        {
            "c1": int(condition.get("c1", base_config.c1)),
            "c2": int(condition.get("c2", base_config.c2)),
            "c3": int(condition.get("c3", base_config.c3)),
            "dropout": float(condition.get("dropout", base_config.dropout)),
        }
    )
    config = Config(**config_values)
    print(
        f"[{label}] starting: C=({config.c1},{config.c2},{config.c3}), "
        f"dropout={config.dropout}, device={device}",
        flush=True,
    )

    seed_everything(seed)
    train_set, validation_set, test_set = datasets
    train_loader = build_loader(train_set, config.batch_size, seed, shuffle=True)
    validation_loader = build_loader(
        validation_set, config.batch_size, seed, shuffle=False
    )
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
        progress_label=label,
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

    seed_everything(seed)
    pass_probabilities, labels = mc_predict(
        model, test_loader, device, config.mc_passes, progress_label=label
    )
    metrics, predictive_entropy, mutual_information = evaluate_mc(
        pass_probabilities, labels
    )
    metrics.update(
        {
            "experiment": experiment,
            "condition": name,
            "seed": seed,
            "loss": test_loss,
            "deterministic_accuracy": deterministic_accuracy,
            "epochs_completed": len(history),
        }
    )
    write_csv(run_dir / "history.csv", history)
    write_csv(run_dir / "metrics.csv", [metrics])
    torch.save(model.state_dict(), run_dir / "model.pt")
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
    print(
        f"[{label}] completed: accuracy={metrics['accuracy']:.4f}, "
        f"ece={metrics['ece']:.5f}, "
        f"entropy_auroc={metrics['error_detection_auroc_predictive_entropy']:.4f}",
        flush=True,
    )
    return metrics


def main() -> None:
    args = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("At least one non-negative seed must be specified.")
    if min(args.epochs, args.batch_size, args.mc_passes, args.patience) < 1:
        raise ValueError("epochs, batch size, MC passes, and patience must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    base_config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        mc_passes=args.mc_passes,
        patience=args.patience,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "config.json",
        {
            "stage": 1,
            "seeds": seeds,
            "device": str(device),
            "conditions": list(CONDITIONS),
            "config": asdict(base_config),
        },
    )
    print(
        f"Stage 1: {len(CONDITIONS)} conditions x {len(seeds)} seed(s) "
        f"({', '.join(map(str, seeds))}), "
        f"device={device}, epochs={args.epochs}, mc_passes={args.mc_passes}",
        flush=True,
    )
    print(f"Loading PneumoniaMNIST from {args.data_dir}...", flush=True)
    datasets = load_datasets(args.data_dir, base_config)
    print(
        f"Loaded train={len(datasets[0])}, validation={len(datasets[1])}, "
        f"test={len(datasets[2])}.",
        flush=True,
    )

    metrics = [
        run_condition(condition, seed, args, base_config, datasets, device)
        for seed in seeds
        for condition in CONDITIONS
    ]
    fieldnames = list(metrics[0])
    with (args.output_dir / "stage1_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    print(
        f"Stage 1 completed: {len(metrics)} conditions saved under {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

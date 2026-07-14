#!/usr/bin/env python3
"""Train, evaluate, or apply an instrument-specific MS QC model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from model import (
    TrainConfig,
    best_youden_threshold,
    load_checkpoint,
    predict_scores,
    save_checkpoint,
    train_vae_mlp,
)


PREPROCESSING_PROFILES = ("dda_missingness_v1", "dia_missingness_v1")


def read_features(path: Path) -> pd.DataFrame:
    """Read a sample-by-feature CSV or compressed CSV matrix."""

    features = pd.read_csv(path, index_col=0)
    features.index = features.index.astype(str)
    if not features.index.is_unique:
        raise ValueError("Feature matrix sample identifiers must be unique.")
    values = features.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return features


def read_labels(path: Path, samples: pd.Index, instrument: str | None) -> np.ndarray:
    """Read explicit good/bad labels and align them to a feature matrix."""

    labels = pd.read_csv(path, dtype=str)
    required = {"sample_id", "quality_label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Label file is missing columns: {sorted(missing)}")
    if instrument is not None and "instrument" in labels.columns:
        labels = labels[labels["instrument"].str.lower() == instrument.lower()]
    labels = labels.set_index("sample_id")
    if not labels.index.is_unique:
        raise ValueError("Label file contains duplicate sample identifiers.")
    unavailable = samples.difference(labels.index)
    if len(unavailable):
        raise ValueError(f"Labels are missing for {len(unavailable)} samples.")
    normalized = labels.loc[samples, "quality_label"].str.lower()
    invalid = sorted(set(normalized).difference({"good", "bad"}))
    if invalid:
        raise ValueError(f"Unknown quality labels: {invalid}")
    return normalized.map({"good": 0, "bad": 1}).to_numpy(dtype=np.int64)


def select_device(name: str) -> torch.device:
    """Resolve a requested CPU, CUDA, or automatic device."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def metric_row(
    instrument: str,
    fold: int,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int | str]:
    """Build one fold-level metrics record."""

    predicted = (scores >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "instrument": instrument,
        "fold": fold,
        "n_samples": int(len(y_true)),
        "n_good": int((y_true == 0).sum()),
        "n_bad": int((y_true == 1).sum()),
        "auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision_bad": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_bad": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity_good": float(specificity),
        "f1_bad": float(f1_score(y_true, predicted, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
    }


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        latent_dim=args.latent_dim,
        vae_epochs=args.vae_epochs,
        mlp_epochs=args.mlp_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


def train_command(args: argparse.Namespace) -> None:
    features = read_features(args.input)
    labels = read_labels(args.labels, features.index, args.instrument)
    config = config_from_args(args)
    device = select_device(args.device)
    vae, mlp, train_scores = train_vae_mlp(
        features.to_numpy(dtype=np.float32),
        labels,
        device,
        config,
        log_prefix=f"[{args.instrument}]",
    )
    threshold = best_youden_threshold(labels, train_scores)
    save_checkpoint(
        args.output,
        vae,
        mlp,
        threshold,
        features.columns.astype(str).tolist(),
        args.instrument,
        args.preprocessing_profile,
        config,
    )
    print(f"Saved {args.output} with threshold={threshold:.6f}", flush=True)


def evaluate_command(args: argparse.Namespace) -> None:
    features = read_features(args.input)
    labels = read_labels(args.labels, features.index, args.instrument)
    x = features.to_numpy(dtype=np.float32)
    base_config = config_from_args(args)
    device = select_device(args.device)
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.split_seed,
    )
    metric_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x, labels), start=1):
        fold_config = replace(base_config, seed=base_config.seed + fold - 1)
        vae, mlp, _ = train_vae_mlp(
            x[train_index],
            labels[train_index],
            device,
            fold_config,
            log_prefix=f"[{args.instrument}:fold-{fold}]",
        )
        scores = predict_scores(vae, mlp, x[test_index], device)

        # This reproduces the threshold definition used in the manuscript analysis.
        threshold = best_youden_threshold(labels[test_index], scores)
        metric_rows.append(
            metric_row(args.instrument, fold, labels[test_index], scores, threshold)
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "sample_id": features.index[test_index],
                    "instrument": args.instrument,
                    "fold": fold,
                    "quality_label": np.where(labels[test_index] == 0, "good", "bad"),
                    "score_bad": scores,
                    "threshold": threshold,
                    "predicted_label": np.where(scores >= threshold, "bad", "good"),
                }
            )
        )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.metrics_output, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.predictions_output, index=False
    )
    print(f"Saved {args.metrics_output} and {args.predictions_output}", flush=True)


def predict_command(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    vae, mlp, checkpoint = load_checkpoint(args.model, device)
    features = read_features(args.input)
    expected_columns = checkpoint.get("feature_columns")
    if expected_columns is not None:
        missing = sorted(set(expected_columns).difference(features.columns))
        unexpected = sorted(set(features.columns).difference(expected_columns))
        if missing or unexpected:
            raise ValueError(
                "Input feature schema does not match the model: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        features = features.loc[:, expected_columns]
    elif features.shape[1] != checkpoint["input_dim"]:
        raise ValueError(
            f"Expected {checkpoint['input_dim']} features, found {features.shape[1]}."
        )

    scores = predict_scores(
        vae,
        mlp,
        features.to_numpy(dtype=np.float32),
        device,
    )
    threshold = float(checkpoint["threshold"])
    output = pd.DataFrame(
        {
            "sample_id": features.index,
            "score_bad": scores,
            "threshold": threshold,
            "predicted_label": np.where(scores >= threshold, "bad", "good"),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {args.output}", flush=True)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--vae-epochs", type=int, default=200)
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save one final model.")
    train_parser.add_argument("--instrument", required=True)
    train_parser.add_argument("--preprocessing-profile", choices=PREPROCESSING_PROFILES, required=True)
    train_parser.add_argument("--input", type=Path, required=True)
    train_parser.add_argument("--labels", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    add_training_arguments(train_parser)
    train_parser.set_defaults(function=train_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run stratified cross-validation.")
    evaluate_parser.add_argument("--instrument", required=True)
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--labels", type=Path, required=True)
    evaluate_parser.add_argument("--folds", type=int, default=5)
    evaluate_parser.add_argument("--split-seed", type=int, default=42)
    evaluate_parser.add_argument("--metrics-output", type=Path, required=True)
    evaluate_parser.add_argument("--predictions-output", type=Path, required=True)
    add_training_arguments(evaluate_parser)
    evaluate_parser.set_defaults(function=evaluate_command)

    predict_parser = subparsers.add_parser("predict", help="Apply a published model.")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict_parser.set_defaults(function=predict_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
All-in-one training script for SAM Audio embedding classification.

This script provides a complete pipeline for:
1. Loading the SAM Audio model and extracting embeddings
2. Training a CNN classifier on audio embeddings
3. Evaluating and saving the trained model

Usage:
    # Basic training with class folders
    python train_classifier.py \
        --train-dir data/train \
        --val-dir data/val \
        --output-dir checkpoints/

    # With precomputed embeddings (faster)
    python train_classifier.py \
        --train-dir data/train \
        --val-dir data/val \
        --precompute-embeddings \
        --embeddings-dir embeddings/ \
        --output-dir checkpoints/

    # Test a trained model
    python train_classifier.py \
        --test-dir data/test \
        --checkpoint checkpoints/best_model.pt \
        --eval-only

Directory structure expected:
    data/
    ├── train/
    │   ├── positive/
    │   │   ├── clip1.wav
    │   │   └── clip2.wav
    │   └── negative/
    │       ├── clip3.wav
    │       └── clip4.wav
    └── val/
        ├── positive/
        └── negative/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam_audio.embeddings.extractor import SAMAudioEmbeddingExtractor
from sam_audio.embeddings.classifier import EmbeddingCNN, EmbeddingClassifier
from sam_audio.embeddings.dataset import (
    AudioClassificationDataset,
    collate_variable_length,
    precompute_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an audio classifier on SAM Audio embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--train-dir",
        type=Path,
        help="Path to training data directory with class subfolders",
    )
    data.add_argument(
        "--val-dir",
        type=Path,
        help="Path to validation data directory",
    )
    data.add_argument(
        "--test-dir",
        type=Path,
        help="Path to test data directory (for evaluation)",
    )
    data.add_argument(
        "--precompute-embeddings",
        action="store_true",
        help="Precompute embeddings before training (recommended for large datasets)",
    )
    data.add_argument(
        "--embeddings-dir",
        type=Path,
        help="Directory for precomputed embeddings",
    )
    data.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="Cache embeddings in memory during training",
    )

    # Model arguments
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--sam-model",
        type=str,
        default="facebook/sam-audio-large",
        help="SAM Audio model ID or path",
    )
    model.add_argument(
        "--classifier",
        type=str,
        choices=["cnn", "residual"],
        default="cnn",
        help="Classifier architecture",
    )
    model.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 512, 256],
        help="Hidden dimensions for CNN layers",
    )
    model.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=[5, 5, 3],
        help="Kernel sizes for CNN layers",
    )
    model.add_argument(
        "--pooling",
        type=str,
        choices=["max", "avg", "attention", "max_avg"],
        default="attention",
        help="Temporal pooling strategy",
    )
    model.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability",
    )
    model.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Number of residual blocks (for residual classifier)",
    )

    # Training arguments
    training = parser.add_argument_group("Training")
    training.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    training.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    training.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    training.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    training.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    training.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "plateau", "none"],
        default="cosine",
        help="Learning rate scheduler",
    )

    # Output arguments
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save model checkpoints",
    )
    output.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name (auto-generated if not provided)",
    )
    output.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )

    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train",
    )
    eval_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint for evaluation",
    )

    # Other arguments
    other = parser.add_argument_group("Other")
    other.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    other.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    other.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(
    args,
    num_classes: int,
    input_dim: int = 128,
) -> nn.Module:
    """Create the classifier model based on arguments."""
    if args.classifier == "cnn":
        model = EmbeddingCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=tuple(args.hidden_dims),
            kernel_sizes=tuple(args.kernel_sizes),
            pooling=args.pooling,
            dropout=args.dropout,
        )
    else:  # residual
        model = EmbeddingClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=args.hidden_dims[0] if args.hidden_dims else 256,
            num_blocks=args.num_blocks,
            pooling=args.pooling,
            dropout=args.dropout,
        )

    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for embeddings, labels, lengths in pbar:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(embeddings, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * embeddings.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for embeddings, labels, lengths in tqdm(loader, desc="Evaluating", leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(embeddings, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * embeddings.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Compute metrics
    accuracy = correct / total
    avg_loss = total_loss / total

    # Compute per-class accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    per_class_acc = {
        cls: class_correct[cls] / class_total[cls]
        for cls in class_total
    }

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "predictions": all_preds,
        "labels": all_labels,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    args,
    path: Path,
):
    """Save a model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[int, Dict]:
    """Load a model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("metrics", {})


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")

    # Load SAM Audio embedding extractor
    print(f"\nLoading SAM Audio model: {args.sam_model}")
    extractor = SAMAudioEmbeddingExtractor.from_pretrained(
        args.sam_model,
        device=args.device,
    )
    print(f"  Embedding dim: {extractor.embedding_dim}")
    print(f"  Frame rate: {extractor.get_frame_rate():.2f} fps")

    # Precompute embeddings if requested
    embeddings_dir = args.embeddings_dir
    if args.precompute_embeddings:
        if embeddings_dir is None:
            embeddings_dir = output_dir / "embeddings"

        for split_name, split_dir in [
            ("train", args.train_dir),
            ("val", args.val_dir),
            ("test", args.test_dir),
        ]:
            if split_dir is not None:
                split_emb_dir = embeddings_dir / split_name
                print(f"\nPrecomputing embeddings for {split_name}...")
                precompute_embeddings(
                    split_dir,
                    split_emb_dir,
                    extractor,
                    verbose=True,
                )

    # Create datasets and dataloaders
    dataset_kwargs = {
        "cache_embeddings": args.cache_embeddings,
    }
    if embeddings_dir and not args.precompute_embeddings:
        # Use existing precomputed embeddings
        dataset_kwargs["precomputed_embeddings_dir"] = embeddings_dir / "train"

    if args.train_dir is not None:
        print(f"\nLoading training data from: {args.train_dir}")
        train_dataset = AudioClassificationDataset(
            root=args.train_dir,
            extractor=extractor if embeddings_dir is None else None,
            precomputed_embeddings_dir=(
                embeddings_dir / "train" if embeddings_dir and args.precompute_embeddings else None
            ),
            cache_embeddings=args.cache_embeddings,
        )
        print(f"  Samples: {len(train_dataset)}")
        print(f"  Classes: {train_dataset.class_to_idx}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_variable_length,
            pin_memory=True,
        )
    else:
        train_dataset = None
        train_loader = None

    val_loader = None
    if args.val_dir is not None:
        print(f"\nLoading validation data from: {args.val_dir}")
        val_dataset = AudioClassificationDataset(
            root=args.val_dir,
            extractor=extractor if embeddings_dir is None else None,
            precomputed_embeddings_dir=(
                embeddings_dir / "val" if embeddings_dir and args.precompute_embeddings else None
            ),
            class_to_idx=train_dataset.class_to_idx if train_dataset else None,
            cache_embeddings=args.cache_embeddings,
        )
        print(f"  Samples: {len(val_dataset)}")

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_variable_length,
            pin_memory=True,
        )

    test_loader = None
    if args.test_dir is not None:
        print(f"\nLoading test data from: {args.test_dir}")
        test_dataset = AudioClassificationDataset(
            root=args.test_dir,
            extractor=extractor if embeddings_dir is None else None,
            precomputed_embeddings_dir=(
                embeddings_dir / "test" if embeddings_dir and args.precompute_embeddings else None
            ),
            class_to_idx=train_dataset.class_to_idx if train_dataset else None,
            cache_embeddings=args.cache_embeddings,
        )
        print(f"  Samples: {len(test_dataset)}")

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_variable_length,
            pin_memory=True,
        )

    # Determine number of classes
    if train_dataset is not None:
        num_classes = train_dataset.num_classes
        class_to_idx = train_dataset.class_to_idx
    else:
        # Load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        num_classes = checkpoint["args"].get("num_classes", 2)
        class_to_idx = checkpoint.get("class_to_idx", {"negative": 0, "positive": 1})

    # Create model
    print(f"\nCreating {args.classifier} classifier...")
    model = create_model(args, num_classes, input_dim=extractor.embedding_dim)
    model = model.to(args.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        start_epoch, _ = load_checkpoint(args.checkpoint, model)
        print(f"  Loaded from epoch {start_epoch}")

    # Evaluation only mode
    if args.eval_only:
        if test_loader is not None:
            print("\nEvaluating on test set...")
            criterion = nn.CrossEntropyLoss()
            results = evaluate(model, test_loader, criterion, args.device)
            print(f"\nTest Results:")
            print(f"  Loss: {results['loss']:.4f}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Per-class accuracy: {results['per_class_accuracy']}")

            # Save results
            with open(output_dir / "test_results.json", "w") as f:
                json.dump({
                    "loss": results["loss"],
                    "accuracy": results["accuracy"],
                    "per_class_accuracy": results["per_class_accuracy"],
                }, f, indent=2)
        return

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
    else:
        scheduler = None

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validate
        if val_loader is not None:
            val_results = evaluate(model, val_loader, criterion, args.device)
            val_loss, val_acc = val_results["loss"], val_results["accuracy"]
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Update scheduler
            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, epoch, val_results, args,
                    output_dir / "best_model.pt",
                )
                print(f"  Saved best model (val_acc={val_acc:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        else:
            if scheduler is not None and args.scheduler != "plateau":
                scheduler.step()

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, {}, args,
                output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
            )

    # Save final model and history
    save_checkpoint(
        model, optimizer, epoch, {}, args,
        output_dir / "final_model.pt",
    )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")

    # Final evaluation on test set
    if test_loader is not None:
        print("\nEvaluating best model on test set...")
        load_checkpoint(output_dir / "best_model.pt", model)
        test_results = evaluate(model, test_loader, criterion, args.device)
        print(f"  Test Loss: {test_results['loss']:.4f}")
        print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Per-class accuracy: {test_results['per_class_accuracy']}")

        with open(output_dir / "test_results.json", "w") as f:
            json.dump({
                "loss": test_results["loss"],
                "accuracy": test_results["accuracy"],
                "per_class_accuracy": test_results["per_class_accuracy"],
            }, f, indent=2)


if __name__ == "__main__":
    main()

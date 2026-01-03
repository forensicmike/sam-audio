# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Dataset utilities for audio classification with SAM Audio embeddings.

Supports loading audio clips organized in folder structures for
binary or multi-class classification.
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

# Supported audio formats
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


class AudioClassificationDataset(Dataset):
    """
    Dataset for audio classification.

    Supports two folder structures:

    1. Class-per-folder (binary or multi-class):
        data/
        ├── positive/
        │   ├── clip1.wav
        │   └── clip2.wav
        └── negative/
            ├── clip3.wav
            └── clip4.wav

    2. Manifest file (JSON):
        {
            "samples": [
                {"path": "audio/clip1.wav", "label": 1},
                {"path": "audio/clip2.wav", "label": 0}
            ],
            "classes": ["negative", "positive"]
        }

    Example:
        >>> dataset = AudioClassificationDataset(
        ...     root="data/train",
        ...     extractor=extractor,
        ... )
        >>> embeddings, label = dataset[0]
    """

    def __init__(
        self,
        root: Union[str, Path],
        extractor: Optional["SAMAudioEmbeddingExtractor"] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        cache_embeddings: bool = False,
        precomputed_embeddings_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            root: Root directory containing class folders or audio files
            extractor: SAMAudioEmbeddingExtractor for computing embeddings on-the-fly
            manifest_path: Path to JSON manifest file (alternative to folder structure)
            transform: Optional transform to apply to embeddings
            class_to_idx: Dict mapping class names to indices (auto-detected if None)
            cache_embeddings: If True, cache computed embeddings in memory
            precomputed_embeddings_dir: Directory containing precomputed .pt embeddings
        """
        self.root = Path(root)
        self.extractor = extractor
        self.transform = transform
        self.cache_embeddings = cache_embeddings
        self.precomputed_dir = Path(precomputed_embeddings_dir) if precomputed_embeddings_dir else None
        self._cache: Dict[int, torch.Tensor] = {}

        if manifest_path is not None:
            self._load_from_manifest(manifest_path, class_to_idx)
        else:
            self._load_from_folders(class_to_idx)

    def _load_from_folders(self, class_to_idx: Optional[Dict[str, int]] = None):
        """Load samples from class-per-folder structure."""
        self.samples: List[Tuple[Path, int]] = []

        # Find all class directories
        class_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.root}")

        # Build class to index mapping
        if class_to_idx is None:
            class_names = [d.name for d in class_dirs]
            self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        # Collect all audio files
        for class_dir in class_dirs:
            if class_dir.name not in self.class_to_idx:
                continue

            class_idx = self.class_to_idx[class_dir.name]

            for audio_file in class_dir.iterdir():
                if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
                    self.samples.append((audio_file, class_idx))

        if not self.samples:
            raise ValueError(f"No audio files found in {self.root}")

    def _load_from_manifest(
        self,
        manifest_path: Union[str, Path],
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """Load samples from a JSON manifest file."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        if "classes" in manifest and class_to_idx is None:
            self.class_to_idx = {name: i for i, name in enumerate(manifest["classes"])}
        elif class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            # Infer from samples
            labels = set(s["label"] for s in manifest["samples"])
            if all(isinstance(l, int) for l in labels):
                self.class_to_idx = {str(l): l for l in sorted(labels)}
            else:
                self.class_to_idx = {str(l): i for i, l in enumerate(sorted(labels))}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        self.samples = []
        for sample in manifest["samples"]:
            path = self.root / sample["path"]
            label = sample["label"]
            if isinstance(label, str):
                label = self.class_to_idx[label]
            self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            embeddings: (T, C) tensor of embeddings
            label: integer class label
        """
        audio_path, label = self.samples[idx]

        # Check cache
        if self.cache_embeddings and idx in self._cache:
            embeddings = self._cache[idx]
        elif self.precomputed_dir is not None:
            # Load precomputed embeddings
            emb_path = self.precomputed_dir / f"{audio_path.stem}.pt"
            if emb_path.exists():
                embeddings = torch.load(emb_path)
            else:
                embeddings = self.extractor.extract(audio_path)
        else:
            # Compute embeddings on-the-fly
            if self.extractor is None:
                raise ValueError(
                    "Either extractor or precomputed_embeddings_dir must be provided"
                )
            embeddings = self.extractor.extract(audio_path)

        # Cache if requested
        if self.cache_embeddings and idx not in self._cache:
            self._cache[idx] = embeddings

        # Apply transform if any
        if self.transform is not None:
            embeddings = self.transform(embeddings)

        return embeddings, label


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for variable-length embeddings.

    Args:
        batch: List of (embeddings, label) tuples

    Returns:
        embeddings: (B, T_max, C) padded embeddings
        labels: (B,) labels
        lengths: (B,) actual lengths
    """
    embeddings_list, labels = zip(*batch)

    lengths = torch.tensor([e.shape[0] for e in embeddings_list])
    labels = torch.tensor(labels)

    # Find max length and embedding dim
    max_len = max(e.shape[0] for e in embeddings_list)
    embed_dim = embeddings_list[0].shape[1]

    # Pad embeddings
    padded = torch.zeros(len(batch), max_len, embed_dim)
    for i, e in enumerate(embeddings_list):
        padded[i, :e.shape[0]] = e

    return padded, labels, lengths


def create_data_loaders(
    train_dir: Union[str, Path],
    val_dir: Optional[Union[str, Path]] = None,
    extractor: Optional["SAMAudioEmbeddingExtractor"] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation data loaders.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory (optional)
        extractor: Embedding extractor
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for AudioClassificationDataset

    Returns:
        train_loader, val_loader (val_loader is None if val_dir not provided)
    """
    train_dataset = AudioClassificationDataset(
        root=train_dir,
        extractor=extractor,
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True,
    )

    val_loader = None
    if val_dir is not None:
        val_dataset = AudioClassificationDataset(
            root=val_dir,
            extractor=extractor,
            class_to_idx=train_dataset.class_to_idx,
            **dataset_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_variable_length,
            pin_memory=True,
        )

    return train_loader, val_loader


def precompute_embeddings(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    extractor: "SAMAudioEmbeddingExtractor",
    verbose: bool = True,
):
    """
    Precompute and save embeddings for all audio files in a directory.

    This can significantly speed up training by avoiding repeated
    embedding extraction.

    Args:
        data_dir: Directory containing audio files (with class subfolders)
        output_dir: Directory to save .pt embedding files
        extractor: SAMAudioEmbeddingExtractor instance
        verbose: Print progress
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(data_dir.rglob(f"*{ext}"))

    if verbose:
        print(f"Found {len(audio_files)} audio files")

    for i, audio_path in enumerate(audio_files):
        output_path = output_dir / f"{audio_path.stem}.pt"

        if output_path.exists():
            if verbose:
                print(f"[{i+1}/{len(audio_files)}] Skipping {audio_path.name} (exists)")
            continue

        try:
            embeddings = extractor.extract(audio_path)
            torch.save(embeddings, output_path)
            if verbose:
                print(f"[{i+1}/{len(audio_files)}] Processed {audio_path.name}: {embeddings.shape}")
        except Exception as e:
            print(f"[{i+1}/{len(audio_files)}] Error processing {audio_path.name}: {e}")

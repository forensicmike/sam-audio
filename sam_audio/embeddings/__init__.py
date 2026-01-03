# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM Audio Embeddings Module

This module provides utilities for extracting and using SAM Audio embeddings
for downstream tasks like audio classification.
"""

from .extractor import SAMAudioEmbeddingExtractor
from .classifier import EmbeddingCNN, EmbeddingClassifier
from .dataset import (
    AudioClassificationDataset,
    collate_variable_length,
    create_data_loaders,
    precompute_embeddings,
)

__all__ = [
    "SAMAudioEmbeddingExtractor",
    "EmbeddingCNN",
    "EmbeddingClassifier",
    "AudioClassificationDataset",
    "collate_variable_length",
    "create_data_loaders",
    "precompute_embeddings",
]

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
CNN Classifiers for SAM Audio Embeddings

Provides lightweight CNN architectures for classifying audio based on
SAM Audio embeddings. Designed to handle variable-length sequences.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPool(nn.Module):
    """
    Attention-based temporal pooling for variable-length sequences.

    Learns to weight different time steps, useful when the target sound
    may only appear in part of the audio clip.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input features
            mask: (B, T) boolean mask, True for valid positions

        Returns:
            (B, C) pooled features
        """
        # Compute attention scores: (B, T, 1)
        scores = self.attention(x)

        if mask is not None:
            # Mask out padding positions
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # Softmax over time dimension
        weights = F.softmax(scores, dim=1)  # (B, T, 1)

        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # (B, C)
        return pooled


class EmbeddingCNN(nn.Module):
    """
    1D CNN classifier for SAM Audio embeddings.

    Designed for binary or multi-class classification of audio clips
    based on their embeddings. Handles variable-length sequences through
    global pooling.

    Architecture:
        Input: (B, T, C) embeddings
        → Conv1D blocks with increasing channels
        → Global pooling (max, avg, or attention)
        → Classification head

    Example:
        >>> model = EmbeddingCNN(
        ...     input_dim=128,
        ...     num_classes=2,
        ...     pooling="attention",
        ... )
        >>> embeddings = torch.randn(4, 750, 128)  # 30-sec clips
        >>> logits = model(embeddings)
        >>> print(logits.shape)  # (4, 2)
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 2,
        hidden_dims: tuple = (256, 512, 256),
        kernel_sizes: tuple = (5, 5, 3),
        pooling: Literal["max", "avg", "attention", "max_avg"] = "attention",
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings (128 for SAM Audio)
            num_classes: Number of output classes
            hidden_dims: Tuple of channel sizes for conv layers
            kernel_sizes: Tuple of kernel sizes for conv layers
            pooling: Pooling strategy - "max", "avg", "attention", or "max_avg"
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pooling_type = pooling

        # Build convolutional layers
        layers = []
        in_channels = input_dim

        for i, (out_channels, kernel_size) in enumerate(
            zip(hidden_dims, kernel_sizes)
        ):
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.final_dim = hidden_dims[-1]

        # Pooling
        if pooling == "attention":
            self.pool = TemporalAttentionPool(self.final_dim)
            classifier_input = self.final_dim
        elif pooling == "max_avg":
            # Concatenate max and avg pooling
            classifier_input = self.final_dim * 2
        else:
            classifier_input = self.final_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, C) input embeddings
            lengths: (B,) actual sequence lengths (for masking in attention pooling)

        Returns:
            (B, num_classes) logits
        """
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)

        # Apply conv layers
        x = self.conv_layers(x)

        # Back to (B, T, C) for pooling
        x = x.transpose(1, 2)

        # Create mask if lengths provided
        mask = None
        if lengths is not None:
            B, T, _ = x.shape
            mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]

        # Apply pooling
        if self.pooling_type == "attention":
            pooled = self.pool(x, mask)
        elif self.pooling_type == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = x.max(dim=1)[0]
        elif self.pooling_type == "avg":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
                pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            else:
                pooled = x.mean(dim=1)
        elif self.pooling_type == "max_avg":
            if mask is not None:
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                max_pool = x_masked.max(dim=1)[0]
                x_avg = x.masked_fill(~mask.unsqueeze(-1), 0.0)
                avg_pool = x_avg.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            else:
                max_pool = x.max(dim=1)[0]
                avg_pool = x.mean(dim=1)
            pooled = torch.cat([max_pool, avg_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Classify
        logits = self.classifier(pooled)
        return logits


class EmbeddingClassifier(nn.Module):
    """
    A more sophisticated classifier with residual connections.

    Better suited for larger datasets or when maximum accuracy is needed.
    Uses a deeper architecture with residual blocks before pooling.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        pooling: Literal["max", "avg", "attention", "max_avg"] = "attention",
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for residual blocks
            num_blocks: Number of residual blocks
            pooling: Pooling strategy
            dropout: Dropout probability
        """
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        self.pooling_type = pooling
        if pooling == "attention":
            self.pool = TemporalAttentionPool(hidden_dim)
            classifier_input = hidden_dim
        elif pooling == "max_avg":
            classifier_input = hidden_dim * 2
        else:
            classifier_input = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, C) input embeddings
            lengths: (B,) actual sequence lengths

        Returns:
            (B, num_classes) logits
        """
        # Project input
        x = self.input_proj(x)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Create mask if lengths provided
        mask = None
        if lengths is not None:
            B, T, _ = x.shape
            mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]

        # Apply pooling
        if self.pooling_type == "attention":
            pooled = self.pool(x, mask)
        elif self.pooling_type == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = x.max(dim=1)[0]
        elif self.pooling_type == "avg":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
                pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            else:
                pooled = x.mean(dim=1)
        elif self.pooling_type == "max_avg":
            if mask is not None:
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                max_pool = x_masked.max(dim=1)[0]
                x_avg = x.masked_fill(~mask.unsqueeze(-1), 0.0)
                avg_pool = x_avg.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            else:
                max_pool = x.max(dim=1)[0]
                avg_pool = x.mean(dim=1)
            pooled = torch.cat([max_pool, avg_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        return self.classifier(pooled)


class ResidualBlock(nn.Module):
    """A simple residual block with 1D convolutions."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C)"""
        residual = x
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(F.relu(self.conv2(x)))
        x = x.transpose(1, 2)  # (B, T, C)
        return x + residual

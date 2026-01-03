# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM Audio Embedding Extractor

Extracts embeddings from SAM Audio's DACVAE encoder for downstream tasks.
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
import torchaudio

from sam_audio import SAMAudio


class SAMAudioEmbeddingExtractor:
    """
    Extracts audio embeddings using SAM Audio's DACVAE encoder.

    The DACVAE encoder produces features at ~25 frames/second (hop_length=1920 at 48kHz).
    Each frame is a 128-dimensional vector representing the audio content.

    Example:
        >>> extractor = SAMAudioEmbeddingExtractor.from_pretrained("facebook/sam-audio-large")
        >>> embeddings = extractor.extract("audio.wav")
        >>> print(embeddings.shape)  # (T, 128) where T depends on audio length
    """

    def __init__(
        self,
        model: SAMAudio,
        device: Optional[str] = None,
        duplicate_features: bool = False,
    ):
        """
        Initialize the embedding extractor.

        Args:
            model: A loaded SAMAudio model
            device: Device to run inference on (auto-detected if None)
            duplicate_features: If True, return duplicated features (B, T, 256)
                              as used by SAM Audio internally. If False (default),
                              return raw codec features (B, T, 128).
        """
        self.model = model
        self.duplicate_features = duplicate_features

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(device).eval()

        # Extract key parameters from the model
        self.sample_rate = model.sample_rate  # 48000 Hz
        self.hop_length = model.audio_codec.hop_length  # 1920
        self.embedding_dim = 256 if duplicate_features else 128

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/sam-audio-large",
        device: Optional[str] = None,
        duplicate_features: bool = False,
        **kwargs,
    ) -> "SAMAudioEmbeddingExtractor":
        """
        Load a pretrained SAM Audio model and create an extractor.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to run on
            duplicate_features: Whether to duplicate features (256-dim) or not (128-dim)
            **kwargs: Additional arguments passed to SAMAudio.from_pretrained

        Returns:
            SAMAudioEmbeddingExtractor instance
        """
        model = SAMAudio.from_pretrained(model_id, **kwargs)
        return cls(model, device=device, duplicate_features=duplicate_features)

    def _load_audio(self, audio: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        """Load and preprocess audio to the correct format."""
        if isinstance(audio, (str, Path)):
            waveform, sr = torchaudio.load(str(audio))
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = audio
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        return waveform

    @torch.inference_mode()
    def extract(
        self,
        audio: Union[str, Path, torch.Tensor],
        return_time_info: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Extract embeddings from an audio file or tensor.

        Args:
            audio: Path to audio file or waveform tensor (channels, samples)
            return_time_info: If True, also return frame timestamps

        Returns:
            embeddings: Tensor of shape (T, embedding_dim) where T = audio_samples / 1920
            timestamps: (optional) Tensor of frame center times in seconds
        """
        waveform = self._load_audio(audio).to(self.device)

        # Add batch dimension if needed
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # (1, channels, samples)

        # Extract features using the audio codec
        # Output shape: (B, T, 128)
        features = self.model.audio_codec(waveform).transpose(1, 2)

        if self.duplicate_features:
            # Duplicate as SAM Audio does internally: (B, T, 256)
            features = torch.cat([features, features], dim=2)

        # Remove batch dimension for single audio
        embeddings = features.squeeze(0).cpu()

        if return_time_info:
            num_frames = embeddings.shape[0]
            # Frame centers in seconds
            timestamps = torch.arange(num_frames) * self.hop_length / self.sample_rate
            return embeddings, timestamps

        return embeddings

    @torch.inference_mode()
    def extract_batch(
        self,
        audios: List[Union[str, Path, torch.Tensor]],
        pad_to_max: bool = True,
    ) -> tuple:
        """
        Extract embeddings from a batch of audio files.

        Args:
            audios: List of audio file paths or waveform tensors
            pad_to_max: If True, pad all embeddings to the max length in the batch

        Returns:
            embeddings: Tensor of shape (B, T_max, embedding_dim) if pad_to_max
                       else list of tensors with varying T
            lengths: Tensor of actual frame counts for each audio
        """
        all_embeddings = []
        lengths = []

        for audio in audios:
            emb = self.extract(audio)
            all_embeddings.append(emb)
            lengths.append(emb.shape[0])

        lengths = torch.tensor(lengths)

        if pad_to_max:
            max_len = max(lengths)
            padded = torch.zeros(len(audios), max_len, self.embedding_dim)
            for i, emb in enumerate(all_embeddings):
                padded[i, :emb.shape[0]] = emb
            return padded, lengths

        return all_embeddings, lengths

    def get_frame_rate(self) -> float:
        """Get the frame rate of embeddings in frames per second."""
        return self.sample_rate / self.hop_length

    def get_expected_frames(self, duration_seconds: float) -> int:
        """Calculate expected number of frames for a given duration."""
        samples = int(duration_seconds * self.sample_rate)
        return (samples + self.hop_length - 1) // self.hop_length

    def frames_to_seconds(self, num_frames: int) -> float:
        """Convert frame count to duration in seconds."""
        return num_frames * self.hop_length / self.sample_rate

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert duration in seconds to frame count."""
        return int(seconds * self.sample_rate / self.hop_length)

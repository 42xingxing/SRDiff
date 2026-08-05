"""Batch collation for the seven-frame satellite-to-radar task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _resize_video(video: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    batch, frames, channels, height, width = video.shape
    video = video.reshape(batch * frames, channels, height, width)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)
    return video.reshape(batch, frames, channels, *target_size)


def collate_fn_sequence(
    batch: list[dict],
    target_size: tuple[int, int] = (128, 128),
    input_keys: Sequence[str] = ("ir069", "ir107", "lght", "vil"),
    num_frames: int = 7,
) -> dict[str, torch.Tensor]:
    """Build the seven-frame satellite-to-radar translation batch."""
    stacked = {
        key: torch.stack([sample[key] for sample in batch], dim=0).permute(0, 2, 1, 3, 4)
        for key in input_keys
    }
    total_frames = stacked["vil"].shape[1]
    if total_frames < num_frames:
        raise ValueError(
            f"Expected at least {num_frames} sampled frames, got {total_frames}"
        )
    resized = {
        key: _resize_video(value[:, :num_frames], target_size)
        for key, value in stacked.items()
    }

    return {
        "condition": torch.cat(
            [resized["ir069"], resized["ir107"], resized["lght"]], dim=2
        ),
        "target": resized["vil"],
    }

"""Continuous-time rectified-flow objective used by SRDiff."""

from __future__ import annotations

import torch


class RFlowScheduler:
    def __init__(self, num_timesteps: int = 1000):
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be positive")
        self.num_timesteps = num_timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate from data at t=0 to Gaussian noise at t=T."""
        data_weight = 1.0 - timesteps.float() / self.num_timesteps
        data_weight = data_weight.reshape(
            data_weight.shape[0], *([1] * (original_samples.ndim - 1))
        )
        return data_weight * original_samples + (1.0 - data_weight) * noise

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,
        model_kwargs: dict | None = None,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fit the constant velocity from noise to data along a straight path."""
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        if timesteps is None:
            timesteps = torch.rand(x_start.shape[0], device=x_start.device)
            timesteps = timesteps * self.num_timesteps
        if noise.shape != x_start.shape:
            raise ValueError("noise and x_start must have the same shape")

        noisy_samples = self.add_noise(x_start, noise, timesteps)
        velocity = model(noisy_samples, timesteps, **model_kwargs)
        return torch.mean((velocity - (x_start - noise)).square())

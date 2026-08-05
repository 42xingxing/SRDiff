"""Paper-setting rectified-flow training and Euler sampling."""

from __future__ import annotations

import torch

from .rectified_flow import RFlowScheduler


class RFLOW:
    def __init__(self, num_sampling_steps: int = 10, num_timesteps: int = 1000):
        if num_sampling_steps < 1:
            raise ValueError("num_sampling_steps must be positive")
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.scheduler = RFlowScheduler(num_timesteps=num_timesteps)

    def sample(
        self,
        model,
        noise: torch.Tensor,
        additional_args: dict | None = None,
    ) -> torch.Tensor:
        """Integrate the learned velocity from noise to data with Euler steps."""
        model_args = dict(additional_args or {})
        samples = noise
        step_size = 1.0 / self.num_sampling_steps
        for index in range(self.num_sampling_steps):
            timestep = (1.0 - index * step_size) * self.num_timesteps
            timesteps = torch.full(
                (samples.shape[0],),
                timestep,
                device=samples.device,
                dtype=torch.float32,
            )
            velocity = model(samples, timesteps, **model_args)
            samples = samples + velocity * step_size
        return samples

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,
        model_kwargs: dict | None = None,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scheduler.training_losses(
            model,
            x_start,
            model_kwargs=model_kwargs,
            noise=noise,
            timesteps=timesteps,
        )


__all__ = ["RFLOW", "RFlowScheduler"]

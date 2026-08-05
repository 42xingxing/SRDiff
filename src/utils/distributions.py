"""Minimal diagonal-Gaussian posterior used by the frame-wise VAE.

Adapted from CompVis Stable Diffusion under CreativeML Open RAIL-M.
See THIRD_PARTY_NOTICES.md.
"""

from typing import Optional

import torch


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, log_variance = torch.chunk(parameters, 2, dim=1)
        self.log_variance = torch.clamp(log_variance, -30.0, 20.0)
        self.standard_deviation = torch.exp(0.5 * self.log_variance)

    def sample(
        self, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.standard_deviation * noise

    def mode(self) -> torch.Tensor:
        return self.mean

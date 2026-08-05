"""Paper-aligned SEVIR metrics for SRDiff evaluation."""

from __future__ import annotations

from collections.abc import Sequence
import re

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional.image import structural_similarity_index_measure

from .dataset import SEVIRDataset


class SEVIRMetrics(Metric):
    """Accumulate CSI-family scores, SSIM, and empirical ensemble CRPS."""

    full_state_update = False

    def __init__(
        self,
        layout: str = "NTHW",
        mode: str = "0",
        seq_len: int | None = None,
        preprocess_type: str = "ddim_pool1",
        threshold_list: Sequence[int] = (16, 74, 133, 160, 181, 219),
        metrics_list: Sequence[str] = ("csi", "bias", "sucr", "pod", "hss"),
        eps: float = 1e-4,
    ):
        super().__init__()
        if layout != "NTHW" or mode != "0":
            raise ValueError("The released evaluator supports layout=NTHW and mode=0")
        match = re.fullmatch(r"ddim_pool([124])", preprocess_type)
        if match is None:
            raise ValueError("preprocess_type must be ddim_pool1, ddim_pool2, or ddim_pool4")

        self.pool_size = int(match.group(1))
        self.threshold_list = tuple(threshold_list)
        self.metrics_list = tuple(metrics_list)
        self.eps = eps
        state_shape = (len(self.threshold_list),)
        for name in ("hits", "misses", "fas", "correct_negatives"):
            self.add_state(
                name,
                default=torch.zeros(state_shape, dtype=torch.float64),
                dist_reduce_fx="sum",
            )
        self.add_state(
            "ssim_sum", default=torch.zeros(1, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "crps_sum", default=torch.zeros(1, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "frame_count", default=torch.zeros(1, dtype=torch.float64), dist_reduce_fx="sum"
        )

    @staticmethod
    def _to_raw_vil(data: torch.Tensor) -> torch.Tensor:
        return SEVIRDataset.process_data_dict_back(
            {"vil": data.detach().float()}, rescale="ddim"
        )["vil"]

    def _pool(self, data: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 1:
            return data
        batch, frames, height, width = data.shape
        data = data.reshape(batch * frames, 1, height, width)
        data = F.max_pool2d(data, kernel_size=self.pool_size, stride=self.pool_size)
        return data.reshape(batch, frames, data.shape[-2], data.shape[-1])

    @staticmethod
    def _empirical_crps(target: torch.Tensor, ensemble: torch.Tensor) -> torch.Tensor:
        """Empirical CRPS; ensemble is [B, E, T, C, H, W]."""
        first_term = (ensemble - target.unsqueeze(1)).abs().mean()
        pairwise = 0.0
        members = ensemble.shape[1]
        for first in range(members):
            for second in range(members):
                pairwise = pairwise + (ensemble[:, first] - ensemble[:, second]).abs().mean()
        second_term = pairwise / (2 * members * members)
        return first_term - second_term

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ensemble: torch.Tensor | None = None,
    ) -> None:
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError("pred and target must have shape [B, T, C, H, W]")
        if ensemble is None:
            ensemble = pred.unsqueeze(1)
        if ensemble.ndim != 6:
            raise ValueError("ensemble must have shape [B, E, T, C, H, W]")

        pred_01 = pred.clamp(0, 1)
        target_01 = target.clamp(0, 1)
        frames = pred.shape[0] * pred.shape[1]
        pred_frames = pred_01.reshape(frames, *pred.shape[2:])
        target_frames = target_01.reshape(frames, *target.shape[2:])
        ssim = structural_similarity_index_measure(
            pred_frames,
            target_frames,
            data_range=1.0,
        )
        self.ssim_sum += ssim * frames
        self.crps_sum += self._empirical_crps(target_01, ensemble.clamp(0, 1)) * frames
        self.frame_count += frames

        pred_raw = self._pool(self._to_raw_vil(pred_01.squeeze(2)))
        target_raw = self._pool(self._to_raw_vil(target_01.squeeze(2)))
        for index, threshold in enumerate(self.threshold_list):
            predicted = pred_raw >= threshold
            observed = target_raw >= threshold
            self.hits[index] += torch.logical_and(predicted, observed).sum()
            self.misses[index] += torch.logical_and(~predicted, observed).sum()
            self.fas[index] += torch.logical_and(predicted, ~observed).sum()
            self.correct_negatives[index] += torch.logical_and(~predicted, ~observed).sum()

    def _scores(self, index: int) -> dict[str, torch.Tensor]:
        hits = self.hits[index]
        misses = self.misses[index]
        fas = self.fas[index]
        correct = self.correct_negatives[index]
        eps = self.eps
        return {
            "csi": hits / (hits + misses + fas + eps),
            "pod": hits / (hits + misses + eps),
            "sucr": hits / (hits + fas + eps),
            "bias": (hits + fas) / (hits + misses + eps),
            "hss": 2 * (hits * correct - misses * fas)
            / (
                (hits + misses) * (misses + correct)
                + (hits + fas) * (fas + correct)
                + eps
            ),
        }

    def compute(self):
        threshold_metrics = {}
        for index, threshold in enumerate(self.threshold_list):
            scores = self._scores(index)
            threshold_metrics[threshold] = {
                name: value for name, value in scores.items() if name in self.metrics_list
            }
        threshold_metrics["avg"] = {
            name: torch.stack(
                [threshold_metrics[threshold][name] for threshold in self.threshold_list]
            ).mean()
            for name in self.metrics_list
        }
        aggregate_metrics = {
            "ssim": self.ssim_sum / self.frame_count,
            "crps": self.crps_sum / self.frame_count,
            # FVD needs a separately distributed, provenance-documented I3D detector.
            "fvd": torch.tensor(float("nan"), device=self.frame_count.device),
        }
        return threshold_metrics, aggregate_metrics

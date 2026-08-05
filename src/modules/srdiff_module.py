"""SRDiff Lightning module shared by the base and CMCA variants."""

import copy
import math
from pathlib import Path

import einops
from omegaconf import OmegaConf
import pytorch_lightning as pl
from rich.console import Console
from tabulate import tabulate
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

from src.models.dit import SpatiotemporalDiT
from src.models.local_adapter import BaseConditionAdapter, CMCAConditionAdapter
from src.models.prediff.autoencoder_kl import AutoencoderKL
from src.schedulers.rf import RFLOW


class WarmupCosineAnnealingLR:
    """Linear warm-up followed by cosine decay to ``final_ratio``."""

    def __init__(self, total_iters: int, final_ratio: float, warmup_steps: int):
        if not 0 <= warmup_steps < total_iters:
            raise ValueError("warmup_steps must be in [0, total_iters)")
        self.total_iters = total_iters
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / (self.total_iters - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_ratio + (1.0 - self.final_ratio) * cosine_decay


class SRDiffLightningModule(pl.LightningModule):
    """Latent SRDiff trained with rectified-flow matching."""

    SUPPORTED_CONDITIONS = {"base", "cmca"}

    def __init__(self, config, metrics):
        super().__init__()
        self.config = config
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        self.console = Console()

        vae_config = OmegaConf.to_container(config.model.vae.model_cfg, resolve=True)
        self.vae = AutoencoderKL(**vae_config)
        checkpoint_path = Path(config.model.vae.from_pretrained)
        resume_path = Path(config.args.resume_run) if config.args.resume_run else None
        if checkpoint_path.is_file():
            self._load_vae_checkpoint(checkpoint_path)
        elif resume_path is not None and resume_path.is_file():
            self.console.log(
                "Frame-wise VAE .pth not found; restoring its registered weights "
                f"from Lightning checkpoint {resume_path}"
            )
        else:
            raise FileNotFoundError(
                "A compatible frame-wise VAE checkpoint is required at "
                f"{checkpoint_path}. See README.md."
            )
        self.vae.requires_grad_(False)
        self.vae.eval()

        dit_config = OmegaConf.to_container(config.model.dit.model_cfg, resolve=True)
        # Older run snapshots exposed this unused option; accept those snapshots
        # without advertising a decoder stack that the released DiT never had.
        dit_config.pop("dec_depth", None)
        if "split_num" in dit_config and "num_frames" not in dit_config:
            dit_config["num_frames"] = dit_config.pop("split_num")
        self.model = SpatiotemporalDiT(**dit_config)

        self.condition_type = config.pipeline.meta.condition_type
        if self.condition_type == "satellite_adapter":
            self.condition_type = "base"
        if self.condition_type not in self.SUPPORTED_CONDITIONS:
            supported = ", ".join(sorted(self.SUPPORTED_CONDITIONS))
            raise ValueError(
                f"Unsupported condition_type={self.condition_type!r}; choose {supported}"
            )
        if self.condition_type == "base":
            adapter_config = OmegaConf.to_container(config.model.adapter, resolve=True)
            self.model.condition_encoder = BaseConditionAdapter(**adapter_config)
        else:
            adapter_config = OmegaConf.to_container(config.model.cmca, resolve=True)
            self.model.condition_encoder = CMCAConditionAdapter(**adapter_config)

        self.pipeline_config = config.pipeline
        self.inference_steps = int(config.args.inference_steps)
        if "rectified_flow" in config.pipeline:
            self.flow_config = config.pipeline.rectified_flow
            train_timesteps = int(self.flow_config.num_train_timesteps)
        else:
            # Backward compatibility with snapshots made before the public cleanup.
            scheduler_config = config.pipeline.training_noise_scheduler
            train_timesteps = int(
                scheduler_config.DDPMScheduler.num_train_timesteps
            )
            self.flow_config = config.pipeline.inference_noise_scheduler
        self.scheduler = RFLOW(
            num_sampling_steps=self.inference_steps,
            num_timesteps=train_timesteps,
        )
        self.register_buffer(
            "scaling_factor",
            torch.tensor(float(config.model.vae.scaling_factor)),
        )
        # Fresh runs calibrate this from the first distributed training batch;
        # resumed/evaluation runs restore the authoritative value from the ckpt.
        self._scale_calibrated = bool(config.args.resume_run)
        self.val_metrics = copy.deepcopy(metrics)

    def _load_vae_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if "model" in checkpoint and "autoencoder_kl" in checkpoint["model"]:
            state_dict = checkpoint["model"]["autoencoder_kl"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        cleaned = {}
        for key, value in state_dict.items():
            for prefix in ("module.vae.", "vae.", "net."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            cleaned[key] = value
        self.vae.load_state_dict(cleaned)
        self.console.log(f"Loaded frame-wise VAE from {checkpoint_path}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    def configure_optimizers(self):
        optimizer_kwargs = OmegaConf.to_container(
            self.pipeline_config.optimizer.kwargs, resolve=True
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **optimizer_kwargs,
        )
        schedule = WarmupCosineAnnealingLR(
            **OmegaConf.to_container(self.pipeline_config.lr_scheduler.kwargs, resolve=True)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, lr_lambda=schedule),
                "interval": "step",
                "name": "warmup_cosine",
            },
        }

    @staticmethod
    def encode_vil(data: torch.Tensor) -> torch.Tensor:
        # The released prediff VAE is trained directly on VIL in [0, 1].
        return data

    @torch.no_grad()
    def encode_latent(self, images: torch.Tensor) -> torch.Tensor:
        moments = self.vae.quant_conv(self.vae.encoder(images))
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean

    @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.scaling_factor
        return self.vae.decoder(self.vae.post_quant_conv(latents))

    def _calibrate_scaling_factor(self, latents: torch.Tensor) -> None:
        values = latents.detach().double()
        stats = torch.stack(
            (
                values.sum(),
                values.square().sum(),
                values.new_tensor(values.numel()),
            )
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        count = stats[2]
        if count.item() <= 1:
            raise RuntimeError("Cannot calibrate latent scale from fewer than two values")
        variance = (stats[1] - stats[0].square() / count) / (count - 1)
        latent_std = variance.clamp_min(0).sqrt().to(self.scaling_factor.dtype)
        if not torch.isfinite(latent_std).item() or latent_std.item() <= 0:
            raise RuntimeError(f"Invalid first-batch latent standard deviation: {latent_std}")
        self.scaling_factor.copy_(latent_std.reciprocal())
        self.config.model.vae.scaling_factor = float(self.scaling_factor)
        self._scale_calibrated = True
        self.console.log(f"Calibrated latent scaling factor: {float(self.scaling_factor):.6f}")
        if self.global_rank == 0:
            OmegaConf.save(
                config=self.config,
                f=Path(self.config.paths.out_dir_run) / "config.yaml",
            )

    def encode_batch(
        self, batch: dict, calibrate_scale: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = batch["target"]
        batch_size, frames = target.shape[:2]
        target_2d = einops.rearrange(target, "b t c h w -> (b t) c h w")
        target_latent = self.encode_latent(self.encode_vil(target_2d))
        target_latent = einops.rearrange(
            target_latent,
            "(b t) c h w -> b t c h w",
            b=batch_size,
            t=frames,
        )
        if calibrate_scale:
            self._calibrate_scaling_factor(target_latent)
        target_latent = target_latent * self.scaling_factor
        condition_latent = self.model.condition_encoder(batch["condition"])
        return target_latent, condition_latent

    def training_step(self, batch, batch_idx):
        target_latent, condition_latent = self.encode_batch(
            batch,
            calibrate_scale=not self._scale_calibrated,
        )
        loss = self.scheduler.training_losses(
            model=self.model,
            x_start=target_latent,
            model_kwargs={"c": condition_latent},
        )
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    @staticmethod
    def _reduce_ensemble(samples: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "mean":
            return samples.mean(dim=0)
        if reduction == "median":
            return samples.median(dim=0).values
        raise ValueError(f"Unsupported ensemble reduction: {reduction}")

    @torch.no_grad()
    def inference_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        target_latent, condition_latent = self.encode_batch(batch)
        ensemble_size = int(self.config.args.ensemble_times)

        def sample_once() -> torch.Tensor:
            noise = torch.randn_like(target_latent)
            latents = self.scheduler.sample(
                self.model,
                noise,
                additional_args={"c": condition_latent},
            )
            batch_size, frames = latents.shape[:2]
            latents_2d = einops.rearrange(latents, "b t c h w -> (b t) c h w")
            prediction = self.decode_latent(latents_2d)
            return einops.rearrange(
                prediction,
                "(b t) c h w -> b t c h w",
                b=batch_size,
                t=frames,
            ).clamp(0, 1)

        if ensemble_size == 1:
            samples = sample_once()
            ensemble_for_metrics = samples.unsqueeze(1)
        else:
            ensemble = torch.stack([sample_once() for _ in range(ensemble_size)], dim=0)
            reduction = self.flow_config.reduction
            samples = self._reduce_ensemble(ensemble, reduction)
            ensemble_for_metrics = ensemble.transpose(0, 1)
        return {"samples": samples, "ensemble": ensemble_for_metrics}

    def on_validation_start(self) -> None:
        self.console.log(
            f"Validating {self.condition_type} with {self.inference_steps} RF steps"
        )

    def validation_step(self, batch, batch_idx):
        result = self.inference_batch(batch)
        self.val_metrics.update(
            pred=result["samples"].float(),
            target=batch["target"].float(),
            ensemble=result["ensemble"].float(),
        )

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self.val_metrics.reset()
            return
        threshold_metrics, aggregate_metrics = self.val_metrics.compute()
        avg_csi = float(threshold_metrics["avg"]["csi"])
        self.log(
            "val_avg_csi",
            avg_csi,
            logger=True,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        for name in ("ssim", "crps"):
            self.log(
                f"val_{name}",
                aggregate_metrics[name],
                logger=True,
                on_epoch=True,
                sync_dist=True,
            )

        if self.global_rank == 0:
            step = int(getattr(self, "val_step", self.global_step))
            postfix = self.config.args.eval_postfix or "_pool1"
            resume_run = self.config.args.resume_run
            checkpoint_name = Path(resume_run).stem if resume_run else "training"
            report_dir = Path(self.config.paths.out_dir_eval) / checkpoint_name
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"step_{step}{postfix}.txt"
            report_path.write_text(
                self._format_metrics(threshold_metrics, aggregate_metrics),
                encoding="utf-8",
            )
        self.val_metrics.reset()

    @staticmethod
    def _format_metrics(threshold_metrics: dict, aggregate_metrics: dict) -> str:
        rows = []
        for threshold, values in threshold_metrics.items():
            rows.append(
                [
                    threshold,
                    *[
                        float(values[name])
                        for name in ("csi", "bias", "sucr", "pod", "hss")
                    ],
                ]
            )
        text = "Evaluation metrics\n"
        text += tabulate(
            rows,
            headers=["Threshold", "CSI", "Bias", "SUCR", "POD", "HSS"],
            tablefmt="rounded_outline",
            floatfmt=".4f",
        )
        aggregate_row = [[
            float(aggregate_metrics["ssim"]),
            float(aggregate_metrics["crps"]),
            float(aggregate_metrics["fvd"]),
        ]]
        text += "\n" + tabulate(
            aggregate_row,
            headers=["SSIM↑", "CRPS↓", "FVD↓"],
            tablefmt="rounded_outline",
            floatfmt=".4f",
        )
        return text + "\n"

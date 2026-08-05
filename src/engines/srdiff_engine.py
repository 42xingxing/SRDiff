"""PyTorch Lightning orchestration for the two released SRDiff variants."""

from __future__ import annotations

from datetime import datetime
from functools import partial
import os
import re
from typing import Any

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer, callbacks, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from datasets import get_dataset_cls
from src.engines.engine import Engine
from src.misc.progress import IterationRichProgressBar
from src.modules.srdiff_module import SRDiffLightningModule


class ExactDistributedSampler(Sampler[int]):
    """Partition evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas: int, default_rank: int = 0):
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.default_rank = default_rank

    def _rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            if dist.get_world_size() != self.num_replicas:
                raise RuntimeError(
                    "Evaluation sampler world size does not match configured devices"
                )
            rank = dist.get_rank()
        else:
            rank = int(os.environ.get("RANK", self.default_rank))
        if not 0 <= rank < self.num_replicas:
            raise RuntimeError(
                f"Evaluation rank {rank} is outside [0, {self.num_replicas})"
            )
        return rank

    def __iter__(self):
        rank = self._rank()
        return iter(range(rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        rank = self._rank()
        return max(0, (len(self.dataset) - rank + self.num_replicas - 1) // self.num_replicas)


class SRDiffEngine(Engine):
    """Train or evaluate SRDiff on the full SEVIR temporal split."""

    def __init__(self, args: Any, config: OmegaConf):
        super().__init__(args, config)
        seed = int(self.config.experiment.get("seed", 42))
        seed_everything(seed, workers=True)
        self.trainer: Trainer | None = None
        self.module: pl.LightningModule | None = None

    def prepare_data(self) -> None:
        registry = get_dataset_cls(self.config.dataset.name.lower())
        dataset_cls = registry["dataset_cls"]
        split_date = datetime(*self.config.dataset.train_test_split_date)
        first_train_date = datetime(*self.config.dataset.first_train_date)

        self.log(f"Loading dataset: {self.config.dataset.name}")
        self.train_dataset = dataset_cls(
            **self.config.dataset.train_dataset,
            start_date=first_train_date,
            end_date=split_date,
        )
        self.test_dataset = dataset_cls(
            **self.config.dataset.test_dataset,
            start_date=split_date,
        )

        dit_config = self.config.model.dit.model_cfg
        num_frames = int(dit_config.get("num_frames", dit_config.get("split_num", 7)))
        collate = partial(
            registry["collate_fn"],
            target_size=(self.config.dataset.img_height, self.config.dataset.img_width),
            input_keys=tuple(self.config.dataset.data_types),
            num_frames=num_frames,
        )
        num_workers = int(self.config.dataloader.num_workers)
        loader_kwargs = {
            "num_workers": num_workers,
            "collate_fn": collate,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=int(self.config.dataloader.train_batch_size_per_gpu),
            shuffle=True,
            **loader_kwargs,
        )
        # The paper does not define a held-out validation split; validation reuses train.
        self.val_dataloader = DataLoader(
            self.train_dataset,
            batch_size=int(self.config.dataloader.test_batch_size_per_gpu),
            shuffle=False,
            **loader_kwargs,
        )
        test_sampler = None
        if int(self.args.gpus) > 1:
            test_sampler = ExactDistributedSampler(
                self.test_dataset,
                num_replicas=int(self.args.gpus),
                default_rank=int(self.args.local_rank),
            )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=int(self.config.dataloader.test_batch_size_per_gpu),
            shuffle=False,
            sampler=test_sampler,
            **loader_kwargs,
        )

        pool_suffix = self.args.eval_postfix or "_pool1"
        if pool_suffix not in {"_pool1", "_pool2", "_pool4"}:
            raise ValueError(f"Unsupported evaluation suffix: {pool_suffix}")
        metric_cfg = self.config.dataset.metrics
        self.metrics = registry["evaluator_cls"](
            mode=str(metric_cfg.metrics_mode),
            metrics_list=tuple(metric_cfg.metrics_list),
            threshold_list=tuple(metric_cfg.threshold_list),
            preprocess_type=f"ddim{pool_suffix}",
        )
        self.log(
            f"Loaded {len(self.train_dataset)} training and "
            f"{len(self.test_dataset)} test samples"
        )

    def prepare_model(self, mode: str = "train") -> None:
        self.config.update({"mode": mode})
        self.module = SRDiffLightningModule(
            config=self.config,
            metrics=self.metrics,
        )

        logger = TensorBoardLogger(
            self.out_dir_logs,
            name=self.config.logging.tensorboard.name,
        )

        trainer_callbacks = [
            IterationRichProgressBar(),
            callbacks.RichModelSummary(max_depth=2),
        ]
        if mode == "train":
            trainer_callbacks.insert(
                0,
                callbacks.ModelCheckpoint(
                    dirpath=self.out_dir_ckpt,
                    filename="srdiff-{step}-{val_avg_csi:.4f}",
                    monitor="val_avg_csi",
                    save_top_k=3,
                    mode="max",
                    every_n_epochs=int(
                        self.config.pipeline.meta.get("check_val_every_n_epoch", 5)
                    ),
                    save_last=True,
                    save_on_train_epoch_end=False,
                ),
            )

        self.trainer = Trainer(
            max_epochs=int(self.config.pipeline.meta.max_epochs),
            max_steps=int(self.config.pipeline.meta.max_iters),
            devices=int(self.args.gpus),
            num_nodes=int(self.args.num_nodes),
            accelerator="gpu",
            logger=logger,
            callbacks=trainer_callbacks,
            check_val_every_n_epoch=int(
                self.config.pipeline.meta.get("check_val_every_n_epoch", 5)
            ),
            default_root_dir=self.out_dir_run,
            precision=self.config.pipeline.meta.get("precision", "bf16-mixed"),
            enable_checkpointing=mode == "train",
            # Latent scaling is calibrated on the first distributed train batch.
            num_sanity_val_steps=0,
            # Evaluation already uses an exact, non-padding distributed sampler.
            use_distributed_sampler=mode == "train",
        )

    def train(self) -> None:
        self.prepare_data()
        self.prepare_model("train")
        self.trainer.fit(
            model=self.module,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            ckpt_path=self.args.resume_run,
            weights_only=True,
        )

    def evaluate(self) -> None:
        if not self.args.resume_run:
            raise ValueError("--resume-run must point to a Lightning .ckpt file")

        self.prepare_data()
        self.prepare_model("eval")
        step_match = re.search(r"step[=_-](\d+)", self.args.resume_run)
        if step_match:
            self.module.val_step = int(step_match.group(1))
        self.trainer.validate(
            model=self.module,
            dataloaders=self.test_dataloader,
            ckpt_path=self.args.resume_run,
            weights_only=True,
        )

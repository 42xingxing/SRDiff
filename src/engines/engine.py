"""Shared filesystem setup for SRDiff train and evaluation runs."""

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

from omegaconf import OmegaConf
from rich.console import Console


class Engine:
    def __init__(self, args: Any, config: OmegaConf) -> None:
        self.args = args
        self.config = config
        self.config.args = OmegaConf.create(vars(args))
        self.console = Console()
        self._initialize_directories()
        self._snapshot_config()

    def _initialize_directories(self) -> None:
        started_at = datetime.now()
        self.log(f"Start time: {started_at}")
        self.root_dir = Path(self.config.experiment.root_dir)

        checkpoint = Path(self.args.resume_run) if self.args.resume_run else None
        resume_dir = None
        if checkpoint and checkpoint.parent.name == "checkpoint":
            candidate = checkpoint.parent.parent
            if (candidate / "config.yaml").is_file():
                resume_dir = candidate

        if self.args.train and resume_dir is not None:
            self.log(f"Resuming run from {checkpoint}")
            # Keep the selected release config authoritative. Pre-cleanup run
            # snapshots contain obsolete fields and may contain machine-local
            # paths; the checkpoint still restores model/optimizer state.
            self.out_dir_run = resume_dir
        else:
            run_name = self.config.experiment.name
            if self.args.add_datetime_prefix:
                run_name = f"{started_at:%Y%m%d-%H%M%S}-{run_name}"
            self.out_dir_run = self.root_dir / "output" / run_name

            if self.args.local_rank == 0 and self.args.debug and self.out_dir_run.exists():
                output_root = (self.root_dir / "output").resolve()
                output_path = self.out_dir_run.resolve()
                if output_path.parent != output_root:
                    raise ValueError(
                        f"Refusing to remove debug output outside {output_root}: {output_path}"
                    )
                self.log(f"Recreating debug output directory: {output_path}")
                shutil.rmtree(output_path)

        self.out_dir_ckpt = self._make_subdirectory("checkpoint")
        self.out_dir_logs = self._make_subdirectory("logs")
        self.out_dir_eval = self._make_subdirectory("evaluation")
        self.config.paths = OmegaConf.create(
            {
                "root_dir": str(self.root_dir),
                "out_dir_run": str(self.out_dir_run),
                "out_dir_ckpt": str(self.out_dir_ckpt),
                "out_dir_logs": str(self.out_dir_logs),
                "out_dir_eval": str(self.out_dir_eval),
            }
        )

    def _make_subdirectory(self, name: str) -> Path:
        path = self.out_dir_run / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _snapshot_config(self) -> None:
        if self.args.resume_run or self.args.local_rank != 0:
            return
        path = self.out_dir_run / "config.yaml"
        OmegaConf.save(config=self.config, f=path)
        self.log(f"Config saved to {path}")

    def log(self, message: str) -> None:
        self.console.log(message)

    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        raise NotImplementedError

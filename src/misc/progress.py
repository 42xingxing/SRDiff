"""Progress-bar customization used by the SRDiff trainer."""

import time

from pytorch_lightning.callbacks import RichProgressBar


class IterationRichProgressBar(RichProgressBar):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        super().on_train_start(trainer, pl_module)

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        metrics.pop("epoch", None)
        metrics["step"] = trainer.global_step
        if self.start_time is not None:
            elapsed = int(time.time() - self.start_time)
            days, remainder = divmod(elapsed, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            metrics["elapsed"] = (
                f"{days:02d}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
            )
        return metrics

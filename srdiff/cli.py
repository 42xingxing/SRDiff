import argparse
import os
from pathlib import Path
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate SRDiff on SEVIR")
    parser.add_argument(
        "--config-path",
        "--config_path",
        required=True,
        dest="config_path",
        help="Path to an SRDiff YAML configuration",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train the model")
    mode.add_argument("--eval", action="store_true", help="Evaluate a checkpoint")

    parser.add_argument("--gpus", type=int, default=1, help="Number of visible GPUs")
    parser.add_argument(
        "--resume-run",
        "--resume_run",
        dest="resume_run",
        help="Lightning .ckpt file used to resume or evaluate",
    )
    parser.add_argument("--debug", action="store_true", help="Recreate the selected output directory")
    parser.add_argument(
        "--add-datetime-prefix",
        "--add_datetime_prefix",
        action="store_true",
        dest="add_datetime_prefix",
        help="Create a timestamped output directory",
    )
    parser.add_argument("--inference-steps", "--inference_steps", type=int, dest="inference_steps")
    parser.add_argument("--ensemble-times", "--ensemble_times", type=int, dest="ensemble_times")
    parser.add_argument("--eval-postfix", "--eval_postfix", default="", dest="eval_postfix")
    return parser.parse_args()


def main():
    warnings.simplefilter("ignore", category=FutureWarning)
    args = parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.num_nodes = 1

    if args.gpus < 1:
        raise ValueError("--gpus must be positive")
    if args.eval and not args.resume_run:
        raise ValueError("--resume-run is required with --eval")
    if args.resume_run and (
        not Path(args.resume_run).is_file() or Path(args.resume_run).suffix != ".ckpt"
    ):
        raise ValueError("--resume-run must point to an existing Lightning .ckpt file")
    import torch

    from src.engines import get_engine_cls
    from src.misc.utils import recursive_load_config

    config = recursive_load_config(args.config_path)
    inference_config = config.pipeline.rectified_flow
    if args.inference_steps is None:
        args.inference_steps = int(inference_config.num_inference_steps)
    if args.ensemble_times is None:
        args.ensemble_times = int(inference_config.ensemble_times)
    if args.inference_steps < 1:
        raise ValueError("--inference-steps must be positive")
    if args.ensemble_times < 1:
        raise ValueError("--ensemble-times must be positive")
    engine = get_engine_cls(config.engine)(args, config)

    if args.train:
        engine.train()
    else:
        engine.evaluate()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

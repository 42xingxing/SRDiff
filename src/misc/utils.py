"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def recursive_load_config(config_path: str | Path, _stack: tuple[Path, ...] = ()):
    """Load a YAML config and recursively merge its ``base_config`` entries."""
    path = Path(config_path).resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Circular base_config chain: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    config = OmegaConf.load(path)
    base_configs = list(config.pop("base_config", []))
    merged = OmegaConf.create()
    for base in base_configs:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = OmegaConf.merge(
            merged,
            recursive_load_config(base_path, (*_stack, path)),
        )
    return OmegaConf.merge(merged, config)

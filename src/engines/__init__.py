"""Engine registry for the released SRDiff experiment."""

from .srdiff_engine import SRDiffEngine


_ENGINE_REGISTRY = {
    "SRDiffEngine": SRDiffEngine,
}


def get_engine_cls(name: str):
    """Return the single engine exposed by this release."""
    try:
        return _ENGINE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ENGINE_REGISTRY))
        raise ValueError(f"Unknown engine {name!r}. Available: {available}") from exc

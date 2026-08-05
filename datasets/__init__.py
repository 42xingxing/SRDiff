"""Dataset registry for the released full-SEVIR experiment."""

from .sevir import (
    SEVIRDataset,
    SEVIRMetrics,
    collate_fn_sequence,
)


_DATASET_REGISTRY = {
    "sevir_sequence": {
        "dataset_cls": SEVIRDataset,
        "evaluator_cls": SEVIRMetrics,
        "collate_fn": collate_fn_sequence,
    },
}


def get_dataset_cls(name: str):
    try:
        return _DATASET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {name!r}. Available: {available}") from exc

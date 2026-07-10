"""Phase 0 smoke tests: f1-vision packages import and torch resolves."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "f1_vision",
        "f1_vision.proxies",
        "f1_vision.imaging",
        "f1_vision.models",
        "f1_vision.train",
        "f1_vision.interpret",
    ],
)
def test_imports(module: str) -> None:
    importlib.import_module(module)


def test_torch_importable() -> None:
    import torch

    assert torch.tensor([1.0]).item() == 1.0

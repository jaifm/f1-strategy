"""Phase 0 smoke tests: workspace packages import and key deps resolve."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "f1_core",
        "f1_core.ingest",
        "f1_core.features",
        "f1_core.models",
        "f1_core.simulate",
        "f1_core.evaluate",
    ],
)
def test_imports(module: str) -> None:
    importlib.import_module(module)


def test_key_dependencies_importable() -> None:
    import lightgbm  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401

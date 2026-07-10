"""Phase 0 smoke tests: f1-pitwall packages import; CLI stub is wired up."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "f1_pitwall",
        "f1_pitwall.orchestrator",
        "f1_pitwall.dashboard",
        "f1_pitwall.cli",
    ],
)
def test_imports(module: str) -> None:
    importlib.import_module(module)


def test_cli_stub_raises_not_implemented() -> None:
    from f1_pitwall.cli import main

    with pytest.raises(NotImplementedError):
        main([])

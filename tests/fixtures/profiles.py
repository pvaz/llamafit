"""Builders for hardware profile tests: valid documents, and directories of them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MINIMAL: dict[str, Any] = {
    "schema_version": 1,
    "name": "test-machine",
    "provenance": "Invented for a test; nothing here was measured.",
    "os": "linux",
    "os_version": "Ubuntu 24.04",
    "arch": "x86_64",
    "cpu": {"model": "Test CPU", "physical_cores": 8, "logical_cores": 16},
    "memory": {"total": "64GiB"},
}
"""The smallest document that validates: every required field and nothing else."""


def document(**overrides: Any) -> dict[str, Any]:
    """A valid profile document with the given top-level fields replaced."""
    return {**MINIMAL, **overrides}


def with_gpu(**overrides: Any) -> dict[str, Any]:
    """A profile of a machine with one NVIDIA card the bundled table knows."""
    return document(
        gpus=[
            {
                "vendor": "nvidia",
                "name": "NVIDIA GeForce RTX 4090",
                "vram_total": "24GiB",
                "vram_used": "1GiB",
                "backend_hint": "cuda",
            }
        ],
        **overrides,
    )


def write(directory: Path, *documents: dict[str, Any]) -> Path:
    """Write each document to ``directory`` as ``<name>.json``; return the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        (directory / f"{doc['name']}.json").write_text(
            json.dumps(doc, indent=2), encoding="utf-8", newline="\n"
        )
    return directory

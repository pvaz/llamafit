# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What LlamaFit knows about the llama.cpp installation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llamafit.models.host import Probe


class RunningServer(BaseModel):
    """A ``llama-server`` answering on a local port."""

    url: str
    model: str | None = None
    n_ctx: int | None = None
    build: str | None = None


class LocalModel(BaseModel):
    """A GGUF file found on disk."""

    path: str
    bytes: int
    sha256: str | None = None
    catalog_id: str | None = None


class LlamaCpp(BaseModel):
    """Installation status; ``problems`` explains anything that limits use."""

    installed: bool
    path: str | None = None
    build: int | None = None
    commit: str | None = None
    backends: list[str] = Field(default_factory=list)
    running_servers: list[RunningServer] = Field(default_factory=list)
    local_models: list[LocalModel] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)

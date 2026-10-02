# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Request constraints and JSON-safe validation failures for the dashboard API."""

from __future__ import annotations

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


class PlanRequest(BaseModel):
    """The body of ``POST /api/v1/plan``: which model, and how to place it.

    Attributes:
        model_id: The catalog id, spelled ``model`` in the published document.
        quant: Which quantisation, or the best one for this machine when omitted.
        context: The context to size for, or the planner's default.
        ub: A micro-batch chosen by hand, which the whole budget is rebuilt around.
        vision: Whether to keep the vision projector.
        target_tps: A finite, positive generation speed to answer against.
        profile: A hardware profile's name, to plan for a machine that is not this one.
        memory: VRAM to pretend the card has.
        ram: System memory to pretend the machine has.
        cpu_cores: Physical cores to pretend the processor has.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model_id: str = Field(alias="model")
    quant: str | None = None
    context: int | None = Field(default=None, gt=0)
    ub: int | None = Field(default=None, gt=0)
    vision: bool = True
    target_tps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    profile: str | None = None
    memory: str | None = None
    ram: str | None = None
    cpu_cores: int | None = Field(default=None, gt=0)


def validation_failure(_request: Request, exc: Exception) -> JSONResponse:
    """Report invalid fields without echoing rejected values or non-JSON error context."""
    if not isinstance(exc, RequestValidationError):
        raise exc
    detail = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(detail)})

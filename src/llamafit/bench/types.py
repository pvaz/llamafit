# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What a benchmark result is, written down so two of them can never be confused.

This is section 16's data model. Everything else in the package produces one of these,
stores one, compares one against an estimate, or fits constants to a pile of them.

One rule shapes the whole file. **A result records the run that happened, not the run that
was asked for.** The two are not the same thing: llama.cpp clamps a context it cannot
honour, a micro-batch sweep passes ``-ub 512,1024,2048`` on one command line and produces
three rows that each ran at one of them, and a binary on ``PATH`` is not necessarily the
build somebody thinks it is. So a result carries three separate things: :attr:`RunConditions.argv`,
which is exactly what was executed; :attr:`RunConditions.settings`, which is what the tool
itself said about the run it did; and :attr:`RunConditions.flag_string`, built from the
second of those, which is the form a later reader compares against a placement.

That last one matters more than it looks. :func:`llamafit.speed.estimate_speed` matches a
stored benchmark to a placement by reading flags out of a recorded command line, and a flag
the record does not mention cannot disagree -- so a record that mentions ``-ngl`` and not
``--n-cpu-moe`` matches every offload that shares its layer count, including ones it says
nothing about. The defence is not in the matcher, which cannot invent what it was not told.
It is here: a record written by this package names every setting that changes a speed, so
there is nothing left for a placement to differ in silently.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llamafit.models.catalog import Measured


def measured_context(n_prompt: int | None, n_gen: int | None) -> int | None:
    """How much key-value cache a measurement's tokens were actually read against.

    Args:
        n_prompt: Prompt tokens the run evaluated, or ``None`` when the tool did not say.
        n_gen: Tokens it generated, likewise.

    Returns:
        The depth the cache had reached by the end of the run, or ``None`` when neither
        count is known.

    A generated token reads every key and value in front of it, so the context a
    measurement belongs to is the one it filled, not the one the server allocated.
    ``llama-bench``'s ``tg128`` sizes its whole context at ``n_prompt + n_gen`` and fills
    it; a server started at 32,768 tokens and asked a thousand-token question fills about
    a thousand. Section 10.1's ``working_context`` is that figure and nothing else, and
    ``docs/calibration/`` already knows it: the reference machine's table gives
    "generation, short context, ``llama-bench tg128``" at 24.7 tokens per second and
    "generation at 32K tokens of context" at 21.7 on one line each, as two measurements
    rather than one.

    ``None`` rather than zero when nothing was reported, because zero is a context and
    would be estimated against as one.
    """
    if n_prompt is None and n_gen is None:
        return None
    return (n_prompt or 0) + (n_gen or 0)


BENCH_SCHEMA_VERSION = 1
"""The layout of the stored result this build reads and writes.

It lives with the model that reads it, the way the catalog's and the hardware profile's
versions do: a row from a future version is refused whole rather than half-understood,
because half-understanding a benchmark is how a number gets attributed to conditions it
was not taken under.
"""

BenchKind = Literal[
    "llama-bench-tg",
    "llama-bench-pp",
    "server-short",
    "server-1k",
    "server-toolcall",
]
"""What was measured, which is part of what a result *is* and not a detail of it.

The reference machine generates 13.0 tokens per second under ``llama-bench tg128`` and
13.9 on a real thousand-token request with the same flags, and the first request after a
cold start is slower again while the model's lookup table pages in from disk. Three
numbers, one configuration, all of them true. Filing them under one label would make the
spread look like noise, so each is a kind of its own and nothing ever averages across them.
"""

PagingReason = Literal[
    "paging",
    "card-not-full",
    "speed-as-expected",
    "no-vram-reading",
    "no-card-total",
    "no-estimate",
]
"""Why the paging detector said what it said, as a key rather than as a sentence.

A key, because the verdict is stored and a sentence is written in whatever language the
reader who ran the benchmark happened to be using. :func:`llamafit.bench.paging.reason_text`
turns one of these into prose at the moment somebody reads it.
"""

RefusalReason = Literal[
    "no-data",
    "too-few-measurements",
    "not-identifiable",
    "unphysical",
    "discarded-with-the-fit",
    "no-term-in-the-model",
]
"""Why a constant was not fitted. Stored as a key, for the reason above.

``no-term-in-the-model`` is the odd one and the one worth reading twice: the fit was not
declined for want of data but because the estimator has no such term at all, so a number
fitted for it would be stored, printed, and read by nothing.

``discarded-with-the-fit`` is the other one. A least-squares solution is a single object:
its parameters are chosen together, against each other, to explain the same rows. If one of
them lands somewhere impossible -- an efficiency above one, a negative overhead -- the
others are not thereby correct, they are the numbers that were chosen to compensate for an
impossible one. So the whole fit goes, and the parameters that were individually plausible
are refused under this reason rather than kept.
"""


class RunTraffic(BaseModel):
    """What the estimator believed this run would read, kept with the run.

    Calibration is arithmetic on these numbers, and they are recorded here rather than
    recomputed later on purpose. Recomputing would mean rebuilding a placement out of a
    command line months afterwards, against a catalog and a set of GGUF facts that may
    have moved -- which is the same "a record matched a configuration it did not
    describe" failure arriving through a different door.

    Attributes:
        device_bytes: Bytes the graphics card reads per generated token.
        sequential_bytes: Bytes read from system memory in contiguous runs, per token.
        scattered_bytes: Routed expert bytes read from system memory, per token.
        ram_gbps: The raw system-memory bandwidth those two were charged against.
        device_gbps: The raw graphics-card bandwidth, or ``None`` with no card.
        streamed_expert_bytes: Expert bytes a prompt micro-batch streams off the card.
        active_params: Parameters active per token, for the prompt-processing compute term.
        compute_flops: Peak fp16 throughput the compute term was charged against.
        pcie_gbps: The link's rated bandwidth, before any efficiency.
        micro_batch: The micro-batch the prompt figures belong to.
        working_context: Tokens of KV cache the generation figure was sized for.
    """

    model_config = ConfigDict(extra="forbid")

    device_bytes: int = Field(ge=0)
    sequential_bytes: int = Field(ge=0)
    scattered_bytes: int = Field(ge=0)
    ram_gbps: float = Field(gt=0)
    device_gbps: float | None = Field(default=None, gt=0)
    streamed_expert_bytes: int = Field(default=0, ge=0)
    active_params: float = Field(default=0.0, ge=0)
    compute_flops: float = Field(default=0.0, ge=0)
    pcie_gbps: float = Field(default=0.0, ge=0)
    micro_batch: int = Field(default=0, ge=0)
    working_context: int = Field(default=0, ge=0)


class PagingCheck(BaseModel):
    """Whether the driver was paging the card out to system memory during the run.

    Attributes:
        paged: True when both halves of section 16.4's signature held, false when the
            question could be asked and answered no, and ``None`` when it could not be
            asked at all. The third state is not a decoration: a machine with no vendor
            tool, or a card whose total nobody could read, has not been cleared of
            paging, and saying so is different from saying it did not page.
        vram_ratio: Peak VRAM over the card's total, when both are known.
        speed_ratio: Measured generation over the estimate it was compared with.
        reason: Which of the two halves decided it, as a key.
        suggested_context: The largest rung of the plan's context ladder that would fit
            without paging, when the run paged and a ladder was supplied.
    """

    model_config = ConfigDict(extra="forbid")

    paged: bool | None
    vram_ratio: float | None = None
    speed_ratio: float | None = None
    reason: PagingReason
    suggested_context: int | None = None


class RunConditions(BaseModel):
    """Everything that has to be equal before two results describe the same run.

    Attributes:
        host_fingerprint: The machine, hashed from the parts of it that do not change
            between one run and the next.
        host_summary: The same machine in words, so a reader can see what a fingerprint
            stands for without having to reproduce it.
        llama_cpp_build: The build number the binary reported for itself.
        llama_cpp_commit: The commit it reported, when it gave one.
        model_id: The catalog id.
        quant: The quantisation.
        model_file: The file the command line named.
        model_bytes: Its size, when it could be read.
        argv: Exactly what was executed, program name first. Never normalised, never
            reordered, never edited: this is the provenance of everything else here.
        settings: The run as the tool itself reported it, keyed by llama.cpp's own flag
            names without their dashes -- ``ngl``, ``ub``, ``n-cpu-moe``, ``ctk``. Where
            the tool reports nothing, the value comes from ``argv``, and
            :func:`llamafit.bench.fingerprint.conditions_conflicts` is what checks the two
            against each other before anything is stored.
        context: The context the run actually held.
        micro_batch: The micro-batch this row actually ran at, which for a sweep is one
            of several the command line named.
        n_prompt: Prompt tokens the measurement used.
        n_gen: Generated tokens the measurement used.
    """

    model_config = ConfigDict(extra="forbid")

    host_fingerprint: str = Field(min_length=1)
    host_summary: str = ""
    llama_cpp_build: int | None = None
    llama_cpp_commit: str | None = None
    model_id: str = Field(min_length=1)
    quant: str = Field(min_length=1)
    model_file: str = ""
    model_bytes: int | None = Field(default=None, ge=0)
    argv: tuple[str, ...] = ()
    settings: dict[str, str] = Field(default_factory=dict)
    context: int | None = Field(default=None, ge=0)
    micro_batch: int | None = Field(default=None, ge=0)
    n_prompt: int | None = Field(default=None, ge=0)
    n_gen: int | None = Field(default=None, ge=0)

    @property
    def measured_context(self) -> int | None:
        """The key-value cache this run filled, which is what its figures are for.

        Not :attr:`context`, which is what the server was *started* at. A server holding a
        32,768-token cache and answering a thousand-token question reads a thousand tokens
        of it per generated token, and section 10.1 charges the read and not the
        allocation. The two are different numbers and only one of them is comparable with
        an estimate.
        """
        return measured_context(self.n_prompt, self.n_gen)

    @property
    def flag_string(self) -> str:
        """The run's settings as a complete llama.cpp command-line fragment.

        Built from :attr:`settings` and never from :attr:`argv`, because a sweep's command
        line says ``-ub 512,1024,2048`` and the row in front of you ran at exactly one of
        those. A reader parsing ``-ub`` out of the raw line would take the first and file
        a 2048-token measurement under 512.

        The order is section 9.5's, and every setting that changes a speed is named, so a
        matcher comparing this against a placement has nothing left to differ in silently.
        """
        order = ("ngl", "n-cpu-moe", "ot", "c", "ub", "b", "t", "ctk", "ctv", "fa")
        parts: list[str] = []
        for key in order:
            value = self.settings.get(key)
            if value is None or value == "":
                continue
            flag = f"--{key}" if len(key) > 3 else f"-{key}"
            parts.append(f"{flag} {value}")
        for key in sorted(set(self.settings) - set(order)):
            value = self.settings[key]
            if value:
                flag = f"--{key}" if len(key) > 3 else f"-{key}"
                parts.append(f"{flag} {value}")
        return " ".join(parts)


class BenchRun(BaseModel):
    """One measurement, with the conditions it was taken under and what it was compared to.

    Attributes:
        id: A random identifier, so two runs of identical conditions stay two rows.
        schema_version: The layout this row was written in.
        recorded_at: When it was stored, in UTC.
        kind: What was measured.
        conditions: The run that happened.
        conditions_hash: A digest of everything in ``conditions`` that a later reader
            compares. Rows sharing it are repeats of one another and nothing else is.
        gen_tps: Generated tokens per second, when this kind produces one.
        pp_tps: Prompt tokens per second, when this kind produces one.
        ttft_ms: Time to first token, in milliseconds, for a server request.
        peak_vram_bytes: The highest VRAM reading taken while the run was going.
        vram_total_bytes: The card's total at the time, for the paging check.
        traffic: What the estimator believed the run would read.
        estimated_gen_tps: What the estimator predicted before the run, and never after.
        estimated_pp_tps: The same for prompt processing.
        paging: The paging verdict, when one could be formed.
        tool_call_ok: Whether the tool-call request came back well-formed.
        buffer_bytes: The budget lines the server's verbose log reported, keyed by the
            buffer's name, so a memory prediction can be checked against what was
            actually allocated.
        llamafit_version: Which build of LlamaFit wrote the row.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    schema_version: int = BENCH_SCHEMA_VERSION
    recorded_at: datetime
    kind: BenchKind
    conditions: RunConditions
    conditions_hash: str = Field(min_length=1)
    gen_tps: float | None = Field(default=None, ge=0)
    pp_tps: float | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    vram_total_bytes: int | None = Field(default=None, ge=0)
    traffic: RunTraffic | None = None
    estimated_gen_tps: float | None = Field(default=None, ge=0)
    estimated_pp_tps: float | None = Field(default=None, ge=0)
    paging: PagingCheck | None = None
    tool_call_ok: bool | None = None
    buffer_bytes: dict[str, int] = Field(default_factory=dict)
    llamafit_version: str = ""

    @property
    def trustworthy(self) -> bool:
        """Whether this run may be fitted against or shown as a measurement of the model.

        A run the paging detector caught measured the driver moving pages, not the
        configuration somebody asked about. Its number is real and worth keeping -- it is
        the evidence that the configuration pages -- but it is not a measurement of the
        speed that configuration would run at, and letting it into a calibration would
        drag every constant towards a machine that was thrashing.
        """
        return not (self.paging is not None and self.paging.paged)

    def as_measurement(self) -> Measured:
        """This run in the shape :func:`llamafit.speed.estimate_speed` already understands.

        The flags handed over are :attr:`RunConditions.flag_string` and not the raw command
        line, for the reason that property gives. The date travels with the number because
        section 10.3 requires a measured figure to carry one.
        """
        taken: date | None = self.recorded_at.date() if self.recorded_at else None
        return Measured(
            profile=f"{self.kind} {self.conditions.host_fingerprint[:8]}",
            quant=self.conditions.quant,
            gen_tps=self.gen_tps,
            pp_tps=self.pp_tps,
            context=self.conditions.context,
            flags=self.conditions.flag_string,
            llama_cpp_build=self.conditions.llama_cpp_build,
            peak_vram_gb=(
                None if self.peak_vram_bytes is None else round(self.peak_vram_bytes / 1e9, 3)
            ),
            date=taken,
            source=f"llamafit bench {self.id}",
        )


class Refusal(BaseModel):
    """One constant that was not fitted, and what stopped it.

    Attributes:
        parameter: The constant's name, as :mod:`llamafit.constants` spells it.
        reason: Why, as a key.
        measurements: How many distinct configurations were available.
        parameters: How many free parameters the fit would have had.
        value: The number the fit produced, when it produced one that was then refused.
    """

    model_config = ConfigDict(extra="forbid")

    parameter: str
    reason: RefusalReason
    measurements: int = Field(default=0, ge=0)
    parameters: int = Field(default=0, ge=0)
    value: float | None = None


class ComputeBufferPoint(BaseModel):
    """What the compute buffer cost at one micro-batch, fitted from the server's own log.

    Attributes:
        micro_batch: The micro-batch this point belongs to.
        base_bytes: What the buffer costs at no context at all.
        per_1k_context_bytes: What each 1,024 tokens of context adds.
        measurements: How many distinct contexts the two came from.
    """

    model_config = ConfigDict(extra="forbid")

    micro_batch: int = Field(gt=0)
    base_bytes: int = Field(ge=0)
    per_1k_context_bytes: int = Field(ge=0)
    measurements: int = Field(ge=0)


class Calibration(BaseModel):
    """The constants fitted to one machine, and everything that was refused.

    The refusals are not an appendix. A calibration that fitted two of five things and
    said nothing about the other three would read as a complete answer, and the label
    ``calibrated`` would then be carried by an estimate three of whose constants are still
    the project's defaults. :attr:`refusals` is what keeps that visible.

    Attributes:
        host_fingerprint: The machine these belong to.
        fitted_at: When the fit was run, in UTC.
        llamafit_version: Which build fitted them.
        eff_vram: Fraction of the card's bandwidth generation reached here.
        eff_ram_sequential: The same for a contiguous system-memory read.
        eff_ram_scattered: The same for a routed expert read.
        fixed_overhead_s: The per-token intercept, in seconds.
        eff_pp: Fraction of the card's fp16 throughput prompt processing reached.
        pcie_effective_gbps: What an expert set actually streamed at, in GB/s.
        compute_buffer: What the compute buffer cost, per micro-batch.
        generation_runs: Distinct configurations the generation fit used.
        prompt_runs: Distinct configurations the prompt fit used.
        generation_residual: Root-mean-square residual of the generation fit, in seconds
            per token, or ``None`` when the system was exactly determined and therefore
            has no residual to report.
        prompt_residual: The same for the prompt fit, in seconds per micro-batch.
        refusals: Every constant that was not fitted, and why.
    """

    model_config = ConfigDict(extra="forbid")

    host_fingerprint: str = Field(min_length=1)
    fitted_at: datetime
    llamafit_version: str = ""
    eff_vram: float | None = Field(default=None, gt=0)
    eff_ram_sequential: float | None = Field(default=None, gt=0)
    eff_ram_scattered: float | None = Field(default=None, gt=0)
    fixed_overhead_s: float | None = Field(default=None, ge=0)
    eff_pp: float | None = Field(default=None, gt=0)
    pcie_effective_gbps: float | None = Field(default=None, gt=0)
    compute_buffer: tuple[ComputeBufferPoint, ...] = ()
    generation_runs: int = Field(default=0, ge=0)
    prompt_runs: int = Field(default=0, ge=0)
    generation_residual: float | None = Field(default=None, ge=0)
    prompt_residual: float | None = Field(default=None, ge=0)
    refusals: tuple[Refusal, ...] = ()

    @property
    def fitted_anything(self) -> bool:
        """Whether a single constant came out of this, which decides what may be labelled."""
        return any(
            value is not None
            for value in (
                self.eff_vram,
                self.eff_ram_sequential,
                self.eff_ram_scattered,
                self.fixed_overhead_s,
                self.eff_pp,
                self.pcie_effective_gbps,
            )
        ) or bool(self.compute_buffer)


class ComparisonRow(BaseModel):
    """One figure the estimator predicted, beside the one that was measured.

    **A row's conditions belong to the whole row, not to one column of it.** Both figures
    are for :attr:`context` and :attr:`micro_batch`, and the validator below is what keeps
    that true: a row may not carry a ratio without saying what context its two halves were
    taken at. The finding that put it there was a table which printed an estimate made at
    32,768 tokens next to a ``tg128`` measurement taken at 128, called the quotient a
    ratio, and so reported a 6.77 error in a formula that was right -- the two numbers
    answered different questions and the column heading did not say so.

    Attributes:
        metric: Which figure, as a key.
        estimated: What the formula said, for the conditions below.
        measured: What the run produced, under the same conditions.
        ratio: Measured over estimated, when both exist and the estimate is not zero.
        unit: ``tok/s``, ``ms`` or ``bytes``.
        context: Tokens of key-value cache both figures are for. For a speed that is what
            the run filled; for peak VRAM it is what the server allocated, since an
            allocation is what a memory prediction is about.
        micro_batch: The micro-batch the run was made at. ``None`` only on peak VRAM,
            which is an allocation rather than a run. A generation row carries one even
            though section 10.1 has no micro-batch term: the run had one, and a sweep
            produces a generation row per rung of the ladder that nothing else tells
            apart.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    estimated: float | None = None
    measured: float | None = None
    ratio: float | None = None
    unit: str = "tok/s"
    context: int | None = Field(default=None, ge=0)
    micro_batch: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _a_ratio_names_the_context_it_was_taken_at(self) -> ComparisonRow:
        """Refuse a quotient of two figures the row cannot say were comparable."""
        if self.ratio is not None and self.context is None:
            raise ValueError("a comparison row with a ratio must name the context both halves")
        return self


class BenchReport(BaseModel):
    """Everything ``llamafit bench`` produced in one invocation.

    Attributes:
        model_id: The catalog id benchmarked.
        quant: The quantisation benchmarked.
        host_fingerprint: The machine it ran on.
        command: The ``llama-bench`` command line that was run, ready to be repeated.
        server_command: The ``llama-server`` command line, when a server was started.
        runs: Every result, in the order they were taken.
        comparison: The estimate beside the measurement, with the gap.
        paging: The paging verdict for the configuration as a whole.
        calibration: The fit, when ``--calibrate`` asked for one.
        stored: Whether the results were written to the store.
        planned_context: The context the plan sized this configuration for, which is the
            context the rest of LlamaFit quotes a speed at.
        planned_gen_tps: What the estimator says generation will run at *there*. It is
            reported and never compared: no row of :attr:`comparison` reaches that
            context, because filling a 32,768-token cache to measure one token is minutes
            of prompt processing per figure. Carried anyway, so that the number `plan`
            prints does not silently vanish from the command that exists to check it.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    quant: str
    host_fingerprint: str
    command: list[str] = Field(default_factory=list)
    server_command: list[str] = Field(default_factory=list)
    runs: list[BenchRun] = Field(default_factory=list)
    comparison: list[ComparisonRow] = Field(default_factory=list)
    paging: PagingCheck | None = None
    planned_context: int | None = Field(default=None, ge=0)
    planned_gen_tps: float | None = Field(default=None, ge=0)
    calibration: Calibration | None = None
    stored: bool = False

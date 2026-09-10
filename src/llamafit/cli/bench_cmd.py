# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit bench``: measure what the rest of the tool has been calculating.

Section 16.1's command. It plans the model exactly as ``llamafit plan`` would, keeps the
estimate that plan produced, runs ``llama-bench`` and a real server at those flags, and
prints the two side by side.

The table is the point of the command, and the column that matters is the last one. A tool
that measured a model and then showed only the measurement would leave its reader better
informed about that model and no better informed about the next one. The ratio is what says
whether the next estimate can be trusted, so the estimate is shown even -- and especially --
when it was wrong.

Three things this command will not do. It will not benchmark a simulated machine, because a
measurement of a machine nobody is sitting at is not a measurement. It will not benchmark a
file that is not on the disk, because the command line would name a path that does not exist
and whatever ran would be something else. And it will not store a result whose flags
disagree with what the tool says it did.

Every ``help=`` here is deferred for the reason ``app.py`` gives: a decorator runs while the
module is imported, before any language has been chosen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import typer
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from llamafit.bench import (
    BenchInputs,
    BenchOptions,
    BenchReport,
    BenchStore,
    Calibration,
    HttpxBenchClient,
    NvidiaSmiSampler,
    SubprocessServerLauncher,
    calibrate,
    compare,
    llama_bench_argv,
    metric_label,
    open_store,
    reason_text,
    run_benchmark,
    store_report,
    within_tolerance,
)
from llamafit.bench.calibrate import refusal_text
from llamafit.bench.fingerprint import host_fingerprint
from llamafit.cli.app import CliState, app
from llamafit.cli.common import (
    find_model,
    load_catalog_or_warn,
    refuse_substitution,
    scan,
)
from llamafit.cli.plan_cmd import choose_quant
from llamafit.constants import MICRO_BATCH_LADDER
from llamafit.errors import NotInstalledError, ProbeError
from llamafit.hardware.runner import SubprocessRunner
from llamafit.i18n import _, for_display, isolate, lazy_gettext, ngettext
from llamafit.llamacpp.detect import exe_name
from llamafit.models.host import Host
from llamafit.models.plan import Needs
from llamafit.models.report import SystemReport
from llamafit.paths import get_paths
from llamafit.services.plan import plan_report
from llamafit.units import format_bytes, localise_number

_MODEL_ARGUMENT: str | None = typer.Argument(
    None, metavar="MODEL", help=cast(str, lazy_gettext("A catalog model id."))
)
_QUANT_OPTION: str | None = typer.Option(
    None,
    "--quant",
    help=cast(str, lazy_gettext("Benchmark this quantisation instead of the planned one.")),
)
_CONTEXT_OPTION: int | None = typer.Option(
    None, "--context", min=1, help=cast(str, lazy_gettext("Size for this many tokens."))
)
_UB_OPTION: int | None = typer.Option(
    None,
    "--ub",
    min=1,
    help=cast(str, lazy_gettext("Set the micro-batch by hand; the budget is rebuilt around it.")),
)

_HOST_FLAG = re.compile(r"(?:^|\s)--host\s+(\S+)")
_PORT_FLAG = re.compile(r"(?:^|\s)--port\s+(\d+)")


def base_url_of(command: list[str]) -> str:
    """Where the server this plan renders will answer.

    Args:
        command: The ``llama-server`` command line the plan produced.

    Returns:
        The address, read out of the command line itself rather than assumed. The plan is
        what decides the port, and a benchmark that talked to a different one would be
        timing somebody else's server.
    """
    line = " ".join(command)
    host = _HOST_FLAG.search(line)
    port = _PORT_FLAG.search(line)
    return f"http://{host.group(1) if host else '127.0.0.1'}:{port.group(1) if port else '8080'}"


@app.command(
    "bench",
    help=cast(
        str,
        lazy_gettext(
            "Measure a model on this machine and show the measurement beside the estimate."
            "\n\nRuns llama-bench and a real server at the planned flags, stores the result"
            " with the conditions it was taken under, and flags a configuration the driver"
            " was paging. With --calibrate, fits the estimator's constants to this machine"
            " from every stored result and says by name which ones the measurements do not"
            " determine."
        ),
    ),
)
def bench_command(
    ctx: typer.Context,
    model_id: str | None = _MODEL_ARGUMENT,
    quant: str | None = _QUANT_OPTION,
    context: int | None = _CONTEXT_OPTION,
    ub: int | None = _UB_OPTION,
    sweep: bool = typer.Option(
        False,
        "--sweep",
        help=cast(
            str,
            lazy_gettext(
                "Measure prompt processing at every micro-batch of the ladder, not only the "
                "planned one. Section 10.2 has two free parameters and one micro-batch is "
                "one equation, so without this the prompt half of a calibration cannot be "
                "fitted at all."
            ),
        ),
    ),
    no_server: bool = typer.Option(
        False,
        "--no-server",
        help=cast(
            str,
            lazy_gettext(
                "Run llama-bench only. Faster, and it gives up the paging check, the buffer "
                "sizes and the tool-call test, all of which need a real server."
            ),
        ),
    ),
    no_store: bool = typer.Option(
        False,
        "--no-store",
        help=cast(str, lazy_gettext("Print the result without writing it to the database.")),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=cast(str, lazy_gettext("Print the commands that would run, and run nothing.")),
    ),
    calibrate_after: bool = typer.Option(
        False,
        "--calibrate",
        help=cast(
            str,
            lazy_gettext("Fit the estimator's constants to this machine from every result."),
        ),
    ),
    show: bool = typer.Option(
        False, "--show", help=cast(str, lazy_gettext("List what is already stored and stop."))
    ),
) -> None:
    """Measure a model on this machine and show the measurement beside the estimate."""
    state: CliState = ctx.obj
    refuse_substitution(state, command="bench")
    system = scan()
    fingerprint = host_fingerprint(system.host)

    if show:
        _show_stored(state, fingerprint, model_id)
        return
    if model_id is None:
        if not calibrate_after:
            raise ProbeError(
                _("`llamafit bench` needs a model to measure"),
                hint=_("Try `llamafit bench <model>`, `--calibrate` on its own, or `--show`."),
            )
        _calibrate_only(state, fingerprint)
        return

    catalog = load_catalog_or_warn(state)
    model = find_model(catalog, model_id)
    chosen = choose_quant(model, system.host, quant)
    plan = plan_report(
        model,
        chosen,
        system.host,
        needs=Needs(use_case=model.use_cases[0], requested_context=context),
        micro_batch=ub,
        local_files=[local.path for local in system.llamacpp.local_models],
    )
    options = BenchOptions(
        with_server=not no_server,
        micro_batches=tuple(
            size for size in MICRO_BATCH_LADDER if sweep and size != plan.placement.micro_batch
        ),
    )
    inputs = BenchInputs(
        plan=plan,
        host=system.host,
        facts=chosen.gguf_facts,
        active_params=model.params.active_b * 1e9,
        bench_executable=_bench_executable(system),
        server_argv=plan.command,
        base_url=base_url_of(plan.command),
        llama_cpp_build=system.llamacpp.build,
        llama_cpp_commit=system.llamacpp.commit,
    )
    bench_argv = llama_bench_argv(
        plan.placement,
        executable=inputs.bench_executable,
        model_path=plan.model_path,
        options=options,
    )
    if dry_run:
        _print_commands(state, bench_argv, plan.command if options.with_server else [])
        return

    _refuse_impossible_runs(system.host.simulated, plan.model_present, plan.model_path)
    log_path = get_paths().log_dir / "bench-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runner = SubprocessRunner()
    result = run_benchmark(
        inputs,
        runner=runner,
        launcher=SubprocessServerLauncher(log_path=log_path),
        http=HttpxBenchClient(),
        sampler=NvidiaSmiSampler(runner=runner, index=_gpu_index(system.host)),
        options=options,
    )
    result = result.model_copy(
        update={
            "comparison": compare(
                result.runs, predicted_vram_bytes=plan.placement.budget.vram_required
            )
        }
    )
    with open_store() as store:
        if not no_store:
            result = store_report(store, result)
        if calibrate_after:
            result = result.model_copy(update={"calibration": _fit_and_save(store, fingerprint)})
    _render(state, result)


def _bench_executable(system: SystemReport) -> str:
    """The ``llama-bench`` binary next to the installation's ``llama-server``.

    Raises:
        NotInstalledError: If llama.cpp was not found, or its directory has no
            ``llama-bench``. Some release archives ship the server without it, and a
            benchmark that silently skipped half of section 16.1 would be a worse answer
            than a clear refusal.
    """
    if not system.llamacpp.installed or not system.llamacpp.path:
        raise NotInstalledError(
            _("llama.cpp was not found, so there is nothing to benchmark with"),
            hint=_("Run `llamafit doctor` to see where it was looked for."),
        )
    binary = Path(system.llamacpp.path) / exe_name("llama-bench", system.host.os)
    if not binary.is_file():
        raise NotInstalledError(
            _("%(path)s has llama-server but no llama-bench") % {"path": system.llamacpp.path},
            hint=_("Some release archives leave it out; a full build has it beside the server."),
        )
    return str(binary)


def _gpu_index(host: Host) -> int:
    """Which card the VRAM sampler asks about: the one everything else is sized for."""
    gpu = host.primary_gpu
    return 0 if gpu is None else gpu.index


def _refuse_impossible_runs(simulated: bool, present: bool, path: str) -> None:
    """Stop before running anything that could not produce an honest measurement."""
    if simulated:
        raise ProbeError(
            _("this host is simulated, and a simulated machine cannot be measured"),
            hint=_("Run `llamafit bench` on the machine itself."),
        )
    if not present:
        raise ProbeError(
            _("the model file is not on this machine: %(path)s") % {"path": path},
            hint=_("Download it first; `llamafit plan` shows the path the command line uses."),
        )


def _print_commands(state: CliState, bench: list[str], server: list[str]) -> None:
    """Print the two command lines and nothing else, for ``--dry-run``."""
    console = state.console
    console.print(Text(_("Would run:"), style="bold"))
    console.print(Text(" ".join(bench)))
    if server:
        console.print(Text(" ".join(server)))


def _fit_and_save(store: BenchStore, fingerprint: str) -> Calibration:
    """Fit this machine's constants from everything stored, and keep the result."""
    fitted = calibrate(store.runs(host_fingerprint=fingerprint), host_fingerprint=fingerprint)
    store.save_calibration(fitted)
    return fitted


def _calibrate_only(state: CliState, fingerprint: str) -> None:
    """``bench --calibrate`` with no model: fit from what is already stored."""
    with open_store() as store:
        fitted = _fit_and_save(store, fingerprint)
    if state.json_output:
        typer.echo(fitted.model_dump_json(indent=2))
        return
    state.console.print(render_calibration(fitted))


def _show_stored(state: CliState, fingerprint: str, model_id: str | None) -> None:
    """``bench --show``: what this machine has already measured."""
    with open_store() as store:
        runs = store.runs(host_fingerprint=fingerprint, model_id=model_id)
    if state.json_output:
        typer.echo(json.dumps([run.model_dump(mode="json") for run in runs], indent=2))
        return
    console = state.console
    if not runs:
        console.print(_("Nothing has been measured on this machine yet."))
        return
    table = Table(title=for_display(_("Stored measurements")))
    table.add_column(for_display(_("model")), overflow="fold")
    table.add_column(for_display(_("kind")))
    table.add_column(for_display(_("gen tok/s")), justify="right")
    table.add_column(for_display(_("prompt tok/s")), justify="right")
    table.add_column(for_display(_("taken")))
    table.add_column(for_display(_("flags")), overflow="fold")
    for run in runs:
        table.add_row(
            _cell(f"{run.conditions.model_id} {run.conditions.quant}"),
            _cell(run.kind),
            _number(run.gen_tps),
            _number(run.pp_tps),
            _cell(run.recorded_at.date().isoformat()),
            _cell(run.conditions.flag_string),
        )
    console.print(table)


def _render(state: CliState, result: BenchReport) -> None:
    """Print the whole benchmark, or its JSON."""
    if state.json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    state.console.print(render_bench(result))


def render_bench(result: BenchReport) -> RenderableType:
    """The comparison table, the paging verdict and the calibration, in that order.

    Args:
        result: What the benchmark produced.

    Returns:
        Everything to print. The comparison comes first because it is the answer; the
        paging verdict next because it is the one thing that can invalidate the answer; the
        calibration last because it is about the next model rather than this one.
    """
    blocks: list[RenderableType] = [render_comparison(result)]
    if result.paging is not None:
        blocks.append(render_paging(result))
    if result.calibration is not None:
        blocks.append(render_calibration(result.calibration))
    if not result.stored:
        blocks.append(
            Text(
                _("Nothing was written to the database, so no estimate will be relabelled."),
                style="yellow",
            )
        )
    return Group(*blocks)


def render_comparison(result: BenchReport) -> RenderableType:
    """The estimate beside the measurement, with the ratio between them."""
    table = Table(title=for_display(_("Estimated against measured")))
    table.add_column(for_display(_("figure")), overflow="fold")
    table.add_column(for_display(_("estimated")), justify="right")
    table.add_column(for_display(_("measured")), justify="right")
    table.add_column(for_display(_("ratio")), justify="right")
    for row in result.comparison:
        in_bytes = row.unit == "bytes"
        table.add_row(
            _cell(metric_label(row.metric)),
            _size(row.estimated) if in_bytes else _number(row.estimated),
            _size(row.measured) if in_bytes else _number(row.measured),
            _ratio(row.ratio),
        )
    blocks: list[RenderableType] = [table]
    outside = [row for row in result.comparison if within_tolerance(row.ratio) is False]
    if outside:
        blocks.append(
            Text(
                ngettext(
                    "%(count)d figure is outside the 0.8 to 1.25 band an uncalibrated formula"
                    " is expected to land in.",
                    "%(count)d figures are outside the 0.8 to 1.25 band an uncalibrated"
                    " formula is expected to land in.",
                    len(outside),
                )
                % {"count": len(outside)},
                style="yellow",
            )
        )
    return Group(*blocks)


def render_paging(result: BenchReport) -> RenderableType:
    """The paging verdict, in the three states the detector actually has."""
    check = result.paging
    if check is None:
        return Text("")
    if check.paged:
        style, headline = "red", _("This configuration was paging.")
    elif check.paged is False:
        style, headline = "green", _("This configuration was not paging.")
    else:
        style, headline = "yellow", _("Paging could not be checked.")
    lines: list[RenderableType] = [Text(headline, style=style), Text(reason_text(check.reason))]
    if check.vram_ratio is not None:
        speed = (
            localise_number(f"{check.speed_ratio * 100:.0f}")
            if check.speed_ratio is not None
            else None
        )
        lines.append(
            Text(
                _(
                    "Peak VRAM was %(percent)s percent of the card, and generation was"
                    " %(speed)s percent of the estimate."
                )
                % {
                    "percent": localise_number(f"{check.vram_ratio * 100:.0f}"),
                    "speed": speed if speed is not None else _("not compared"),
                },
                style="dim",
            )
        )
    if check.suggested_context:
        lines.append(
            Text(
                _("The largest context on the ladder that would not page is %(tokens)s.")
                % {"tokens": localise_number(f"{check.suggested_context:,}")}
            )
        )
    return Group(*lines)


def render_calibration(calibration: Calibration) -> RenderableType:
    """What was fitted to this machine, and what was refused, with the reason for each."""
    table = Table(title=for_display(_("Calibration")))
    table.add_column(for_display(_("constant")), overflow="fold")
    table.add_column(for_display(_("value")), justify="right")
    table.add_column(for_display(_("from")), overflow="fold")
    overhead_ms = (
        calibration.fixed_overhead_s * 1000 if calibration.fixed_overhead_s is not None else None
    )
    prompt_names = ("eff_pp", "pcie_effective_gbps")
    for name, value in (
        ("eff_vram", calibration.eff_vram),
        ("eff_ram_sequential", calibration.eff_ram_sequential),
        ("eff_ram_scattered", calibration.eff_ram_scattered),
        ("fixed_overhead_ms", overhead_ms),
        ("eff_pp", calibration.eff_pp),
        ("pcie_effective_gbps", calibration.pcie_effective_gbps),
    ):
        if value is None:
            continue
        runs = calibration.prompt_runs if name in prompt_names else calibration.generation_runs
        table.add_row(_cell(name), _number(value), _cell(_configurations(runs)))
    for point in calibration.compute_buffer:
        table.add_row(
            _cell(f"compute_buffer[{point.micro_batch}]"),
            _cell(
                _("%(base)s + %(slope)s per 1K")
                % {
                    "base": format_bytes(point.base_bytes),
                    "slope": format_bytes(point.per_1k_context_bytes),
                }
            ),
            _cell(
                ngettext(
                    "%(count)d measured context", "%(count)d measured contexts", point.measurements
                )
                % {"count": point.measurements}
            ),
        )
    blocks: list[RenderableType] = [table]
    if not calibration.fitted_anything:
        blocks.append(
            Text(
                _(
                    "Nothing was fitted. Every line below says what its constant was"
                    " missing; a calibration made from less than that would look precise"
                    " and be arbitrary."
                ),
                style="yellow",
            )
        )
    for refusal in calibration.refusals:
        blocks.append(
            Text(
                _("%(name)s was not fitted: %(reason)s")
                % {"name": isolate(refusal.parameter), "reason": refusal_text(refusal)},
                style="dim",
            )
        )
    return Group(*blocks)


def _configurations(count: int) -> str:
    """How many distinct configurations a fitted constant came from."""
    return ngettext(
        "%(count)d measured configuration", "%(count)d measured configurations", count
    ) % {"count": count}


def _cell(value: str) -> Text:
    """One cell, prepared for the reader's writing direction and never parsed as markup."""
    return Text(for_display(value))


def _number(value: float | None) -> str | None:
    """A figure with two decimals in the reader's punctuation, or ``None``."""
    return None if value is None else localise_number(f"{value:.2f}")


def _size(value: float | None) -> str | None:
    """A byte count in the reader's punctuation, or ``None``."""
    return None if value is None else localise_number(format_bytes(int(value)))


def _ratio(value: float | None) -> Text:
    """The gap between an estimate and a measurement, marked when it is outside the band."""
    if value is None:
        return Text("")
    inside = within_tolerance(value)
    return Text(localise_number(f"{value:.2f}"), style="" if inside else "yellow")


__all__ = [
    "base_url_of",
    "bench_command",
    "render_bench",
    "render_calibration",
    "render_comparison",
    "render_paging",
]

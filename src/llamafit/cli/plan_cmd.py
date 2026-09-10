# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit plan``: one model, one quantisation, and the command line that runs it.

This is the end of the pipeline and the place where a mistake costs the most. Everything
before it can be re-read and argued with; the last line of this output is copied into a
terminal by somebody who has already decided to trust it.

So the plan shows its working. The budget is printed line by line with the source of every
line, the context ladder is printed with what each rung costs the card and what would
happen there, the speed is printed with where a token's time goes, and the runs the
catalog records are printed beside the estimate rather than being allowed to become it.
The last thing on the page is the command, on a line of its own, unwrapped.
"""

from __future__ import annotations

from typing import cast

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.common import checked_max_context, find_model, load_catalog_or_warn, machine
from llamafit.cli.render_board import render_plan
from llamafit.errors import CatalogError
from llamafit.i18n import _, lazy_gettext
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.host import Host
from llamafit.models.plan import Needs
from llamafit.services.plan import plan_report
from llamafit.services.recommend import best_quant, quant_entries

_MODEL_ARGUMENT: str = typer.Argument(
    ..., metavar="MODEL", help=cast(str, lazy_gettext("A catalog model id."))
)
_QUANT_OPTION: str | None = typer.Option(
    None,
    "--quant",
    help=cast(
        str,
        lazy_gettext("Plan this quantisation instead of the one that scores best on this machine."),
    ),
)
_CONTEXT_OPTION: int | None = typer.Option(
    None,
    "--context",
    min=1,
    help=cast(
        str,
        lazy_gettext(
            "Size for this many tokens. When it does not fit, the plan sizes down and shows "
            "what the context you asked for would have cost."
        ),
    ),
)
_UB_OPTION: int | None = typer.Option(
    None,
    "--ub",
    min=1,
    help=cast(
        str,
        lazy_gettext(
            "Set the micro-batch by hand. The whole budget is rebuilt around it, since it "
            "moves the compute buffer and with it the verdict."
        ),
    ),
)
_TARGET_TPS_OPTION: float | None = typer.Option(
    None,
    "--target-tps",
    min=0.0,
    help=cast(
        str,
        lazy_gettext(
            "A generation speed to answer against: whether this reaches it, and what context would."
        ),
    ),
)


def choose_quant(model: CatalogModel, host: Host, name: str | None) -> Quant:
    """The quantisation to plan: the one asked for, or the one that scores best here.

    Args:
        model: The catalog entry.
        host: The scanned machine.
        name: What ``--quant`` named, case-insensitively, or ``None``.

    Returns:
        The chosen quantisation.

    Raises:
        CatalogError: If ``--quant`` names one this model does not publish.

    With no ``--quant``, the choice is made the same way ``recommend`` would make it, for
    the job the entry says the model was built for: plan each quantisation, score each, and
    take the best. Anything else would let ``plan qwen3.8-flash-next`` and
    ``recommend --use-case coding`` disagree about the same model on the same machine,
    which is the kind of difference nobody would think to check.
    """
    quants = quant_entries(model)
    if not quants:
        raise CatalogError(
            _("%(model)s publishes no quantisations") % {"model": model.id},
            hint=_("Run `llamafit catalog refresh --model %(model)s`.") % {"model": model.id},
        )
    if name is not None:
        wanted = name.strip().casefold()
        for quant in quants:
            if quant.name.casefold() == wanted:
                return quant
        raise CatalogError(
            _("%(model)s has no %(quant)s quantisation") % {"model": model.id, "quant": repr(name)},
            hint=_("It publishes: %(names)s") % {"names": ", ".join(q.name for q in quants)},
        )
    if len(quants) == 1:
        return quants[0]
    return best_quant(model, host)


@app.command(
    "plan",
    help=cast(
        str,
        lazy_gettext(
            "Place one model on this machine and print the command line that runs it.\n\n"
            "Shows the memory budget component by component, the context ladder a launch "
            "script chooses from, where a token's time goes, and any run the catalog "
            "records for comparison."
        ),
    ),
)
def plan_command(
    ctx: typer.Context,
    model_id: str = _MODEL_ARGUMENT,
    quant: str | None = _QUANT_OPTION,
    context: int | None = _CONTEXT_OPTION,
    ub: int | None = _UB_OPTION,
    target_tps: float | None = _TARGET_TPS_OPTION,
    no_vision: bool = typer.Option(
        False,
        "--no-vision",
        help=cast(
            str,
            lazy_gettext("Leave the vision projector out, freeing its memory for context."),
        ),
    ),
) -> None:
    """Place one model on this machine and print the command line that runs it."""
    state: CliState = ctx.obj
    catalog = load_catalog_or_warn(state)
    model = find_model(catalog, model_id)
    report = machine(state)
    chosen = choose_quant(model, report.host, quant)
    report_out = plan_report(
        model,
        chosen,
        report.host,
        needs=Needs(
            use_case=model.use_cases[0],
            requested_context=context,
            max_context=checked_max_context(state),
        ),
        vision=not no_vision,
        micro_batch=ub,
        target_tps=target_tps,
        local_files=[local.path for local in report.llamacpp.local_models],
    )
    if state.json_output:
        typer.echo(report_out.model_dump_json(indent=2, by_alias=True))
        return
    state.console.print(render_plan(report_out))

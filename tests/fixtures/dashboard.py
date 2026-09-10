"""Dashboards for the terminal interface, built from recorded machines and known catalogs.

Every test of the dashboard hands it a scan and a catalog rather than letting it take one,
so nothing here depends on the graphics card of whatever runs the suite, and the awkward
cases -- an empty catalog, a machine with no card, a scan that failed -- are arguments
rather than accidents.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from llamafit.catalog.loader import Problem as CatalogProblem
from llamafit.errors import LlamaFitError, ProbeError
from llamafit.models.catalog import Catalog
from llamafit.models.host import Host
from llamafit.models.llamacpp import LocalModel
from llamafit.models.report import SystemReport
from llamafit.tui.state import Dashboard, Request
from tests.fixtures.board import catalog, report


def empty_catalog() -> Catalog:
    """A catalog with nothing in it, which a broken installation can genuinely produce."""
    return Catalog(models=[])


def some_models(*ids: str) -> Catalog:
    """The bundled catalog cut down to the entries named."""
    whole = catalog()
    return Catalog(models=[model for model in whole.models if model.id in ids])


def ready(
    *,
    host: Host | None = None,
    models: Catalog | None = None,
    local_models: Sequence[LocalModel] = (),
    catalog_problems: Sequence[CatalogProblem] = (),
    request: Request | None = None,
) -> Dashboard:
    """A dashboard that has already scanned and ranked, so no worker runs at start-up."""
    board = Dashboard(
        scanner=lambda: report(host=host, local_models=list(local_models)),
        loader=lambda: (models if models is not None else catalog(), list(catalog_problems)),
        request=request,
    )
    board.refresh()
    return board


def unscanned(
    *,
    scanner: Callable[[], SystemReport] | None = None,
    models: Catalog | None = None,
) -> Dashboard:
    """A dashboard that has not scanned yet, so mounting it starts the worker."""
    return Dashboard(
        scanner=scanner or report,
        loader=lambda: (models if models is not None else catalog(), []),
    )


def refuses_to_scan(message: str = "nvidia-smi could not be run") -> Callable[[], SystemReport]:
    """A scan that fails the way a real one does, with a message and a hint."""

    def scan() -> SystemReport:
        raise ProbeError(message, hint="Install or repair the NVIDIA driver.")

    return scan


def refuses_to_load(exc: LlamaFitError) -> Callable[[], tuple[Catalog, list[CatalogProblem]]]:
    """A catalog loader that raises, the way a wheel missing its packaged data does."""

    def load() -> tuple[Catalog, list[CatalogProblem]]:
        raise exc

    return load

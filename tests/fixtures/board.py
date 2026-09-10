"""A scanned machine and a catalog for the board and plan services to be tested against.

The catalog here is the real bundled one, loaded once, because the whole point of these
tests is the join: five real entries with real GGUF facts, sized by the real budget against
the recorded reference machine. A made-up catalog would test the plumbing and nothing else.
"""

from __future__ import annotations

from llamafit.catalog import load_catalog
from llamafit.models.catalog import Catalog, CatalogModel, Quant
from llamafit.models.host import Host
from llamafit.models.llamacpp import LlamaCpp, LocalModel
from llamafit.models.report import SystemReport
from tests.fixtures.budget_hosts import first_quant, reference_host

GIB = 1024**3


def catalog() -> Catalog:
    """The bundled catalog, which must load without a problem for any of this to mean much."""
    loaded, problems = load_catalog()
    assert problems == [], problems
    return loaded


def model_and_quant(model_id: str) -> tuple[CatalogModel, Quant]:
    """One bundled entry and the quantisation it publishes."""
    model = catalog().by_id[model_id]
    return model, first_quant(model)


def report(
    *,
    host: Host | None = None,
    local_models: list[LocalModel] | None = None,
) -> SystemReport:
    """A system report a command can be handed instead of scanning the machine under test.

    llama.cpp is reported as installed with no running server, because none of these
    commands cares whether it is: they size models, they do not launch them. What they do
    care about is ``local_models``, which is what turns a planned path into a real one.
    """
    return SystemReport(
        host=host or reference_host(),
        llamacpp=LlamaCpp(
            installed=True, path="/opt/llama.cpp", build=10867, local_models=local_models or []
        ),
        version="0.0.0-test",
    )

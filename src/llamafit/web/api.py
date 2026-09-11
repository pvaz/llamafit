# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The JSON API, and the mount that serves the page which reads it.

Every endpoint here does the same three things: turn query parameters into the arguments
a service function already takes, call that service, and hand back
``model.model_dump_json(by_alias=True)`` -- the identical call ``--json`` makes. Nothing
on this page computes a byte, a token or a score. That is not a style preference: section
13.3 promises that ``--json`` and the API never disagree, and the only way to keep that
promise is for one of them to have no arithmetic to get wrong.

``by_alias`` is load-bearing rather than decorative. A budget line's size is ``bytes_`` in
Python and ``bytes`` in the document, because ``bytes`` is a builtin; an API that
serialised without the aliases would publish a second, subtly different spelling of every
model in the project.

Three things this file refuses to do, all for the same reason -- section 1.3 puts remote
hosts, multi-user access and authentication outside this project, so the server has no
way to tell one caller from another and must not hold anything that assumes it can:

* **No CORS headers.** A page on another origin may send a request and may not read the
  reply. There is nothing to gain from relaxing that on a server with no login.
* **The Host header is checked.** Without it, any website in the world could point a
  hostname of its own at ``127.0.0.1`` and read this API through the browser of anyone
  who visits -- DNS rebinding, which is the one way a loopback bind can leak to the
  internet. Starlette's ``TrustedHostMiddleware`` is what closes it.
* **A profile is named, never pathed.** ``--profile`` on the command line happily takes a
  path, because the person typing it already owns the filesystem. Over HTTP that would be
  a way to ask the server to read a file, so the query parameter takes a profile's name
  and refuses anything shaped like a path.

An unknown query parameter is an error, not something to ignore. A caller who writes
``use-case`` for ``use_case`` would otherwise get a confident answer to a question they
did not ask, which is the exact failure this project's "no silent defaults" rule exists
to prevent.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Imported for its side effect and its ordering, not for a name. ``llamafit.cli.common``
# reads ``CliState`` out of ``llamafit.cli.app``, whose own last act is to import every
# command module, and each of those reads ``llamafit.cli.common`` back. Reached in that
# order the cycle resolves; reached the other way round -- which is what happens when a
# test imports this module before anything has touched the command line -- ``common`` is
# still half-built when a command asks it for a name. Loading the Typer application first
# is what makes the two entrances to the package equivalent.
import llamafit.cli.app  # noqa: F401
from llamafit import __version__
from llamafit.catalog.loader import Problem, load_catalog
from llamafit.cli.common import (
    check_capabilities,
    check_context_ceiling,
    check_size,
    check_use_case,
    find_model,
)
from llamafit.cli.plan_cmd import choose_quant
from llamafit.cli.render_board import SORT_KEYS, parse_sort, sorted_rows
from llamafit.data import packaged_dir, packaged_text
from llamafit.errors import (
    CatalogError,
    ConfigError,
    LlamaFitError,
    NotInstalledError,
    PackagedDataError,
    ProbeError,
)
from llamafit.hwprofile import load_profiles, resolve_host
from llamafit.i18n import _, ngettext
from llamafit.models.catalog import Catalog, CatalogModel
from llamafit.models.host import Host
from llamafit.models.plan import Needs
from llamafit.models.report import SystemReport
from llamafit.scoring import PREFERENCES
from llamafit.services.catalog import describe
from llamafit.services.doctor import diagnose
from llamafit.services.plan import PlanReport, plan_report
from llamafit.services.recommend import BoardRow, build_board
from llamafit.services.scan import scan_system
from llamafit.web.strings import ui_payload

API_PREFIX = "/api/v1"
"""Where the JSON lives. The version is in the path so a later shape can live beside it."""

DEFAULT_LIMIT = 10
"""Rows the board returns when the caller does not say; the same default as ``recommend``."""

LOOPBACK_HOSTNAMES: tuple[str, ...] = ("127.0.0.1", "localhost", "::1", "[::1]")
"""The names a browser may address this server by, before ``--host`` adds another.

Anything else is refused by :class:`TrustedHostMiddleware`. The attack this stops is DNS
rebinding: a site the user visits resolves its own hostname to ``127.0.0.1`` and then
reads this API from their browser, which believes it is talking to that site's origin and
so lets the site's JavaScript see every answer. Checking the ``Host`` header is what makes
"bound to loopback" mean what a reader thinks it means.
"""

BOARD_PARAMETERS: frozenset[str] = frozenset(
    {
        "use_case",
        "require",
        "prefer",
        "license",
        "min_context",
        "max_context",
        "max_download",
        "all_quants",
        "limit",
        "sort",
        "vision",
        "profile",
        "memory",
        "ram",
        "cpu_cores",
    }
)
"""Every query parameter the board endpoints accept; anything else is a 400."""


class Dashboard:
    """What the endpoints share: one scan, one catalog, and a lock over both.

    The scan is cached because scanning runs vendor tools and measures memory bandwidth,
    which takes seconds; ``POST /api/v1/scan`` is how a reader asks for a fresh one.
    Handlers are ordinary ``def`` functions, so Starlette runs each in a worker thread and
    two can arrive at once; the lock is what stops two first requests scanning twice.
    """

    def __init__(
        self,
        scan: Callable[[], SystemReport] = scan_system,
        catalog_loader: Callable[[], tuple[Catalog, list[Problem]]] = load_catalog,
    ) -> None:
        """Hold the two ways in to the machine and to the catalog.

        Args:
            scan: How to scan the machine, injected so tests do not probe hardware.
            catalog_loader: How to load the catalog, injected for the same reason.
        """
        self.scan = scan
        self.catalog_loader = catalog_loader
        self._lock = threading.Lock()
        self._report: SystemReport | None = None
        self._catalog: Catalog | None = None
        self._problems: list[Problem] = []

    def report(self, *, refresh: bool = False) -> SystemReport:
        """The scan every endpoint sizes against.

        Args:
            refresh: Scan again rather than reuse the cached result.

        Returns:
            The machine and its llama.cpp installation.
        """
        with self._lock:
            if refresh or self._report is None:
                self._report = self.scan()
            return self._report

    def catalog(self) -> Catalog:
        """The catalog, loaded once.

        Returns:
            Every model LlamaFit knows, bundled and custom.
        """
        with self._lock:
            if self._catalog is None:
                self._catalog, self._problems = self.catalog_loader()
            return self._catalog

    def problems(self) -> list[Problem]:
        """What was wrong with the catalog files, for the page to say so once.

        Returns:
            Every problem the loader found, empty when every file loaded. A user who
            mistypes a field in their own ``custom_models.yaml`` would otherwise simply
            not see their model, with nothing on the page to say why; the command line
            warns about exactly this on stderr, and the page has no stderr.
        """
        self.catalog()
        return list(self._problems)


class PlanRequest(BaseModel):
    """The body of ``POST /api/v1/plan``: which model, and how to place it.

    Attributes:
        model_id: The catalog id, spelled ``model`` in the document because that is what
            ``docs/web.md`` published.
        quant: Which quantisation, or the best one for this machine when omitted.
        context: The context to size for, or the planner's default.
        ub: A micro-batch chosen by hand, which the whole budget is rebuilt around.
        vision: Whether to keep the vision projector.
        target_tps: A generation speed to answer against.
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
    target_tps: float | None = Field(default=None, gt=0)
    profile: str | None = None
    memory: str | None = None
    ram: str | None = None
    cpu_cores: int | None = Field(default=None, gt=0)


def _json(model: BaseModel) -> Response:
    """Serialise a pydantic model exactly as ``--json`` does.

    Args:
        model: What the service returned.

    Returns:
        The document, aliases and all, as an ``application/json`` response.
    """
    return Response(content=model.model_dump_json(by_alias=True), media_type="application/json")


def _error_body(exc: LlamaFitError) -> dict[str, dict[str, str | None]]:
    """The one error shape the whole API uses, from ``docs/web.md``."""
    return {
        "error": {
            "message": str(exc.message),
            "hint": exc.hint,
            "command": exc.command,
        }
    }


def _status_for(exc: LlamaFitError) -> int:
    """Which status code a failure deserves.

    Args:
        exc: The failure a service raised.

    Returns:
        503 when the machine is the problem -- a probe that could not run, llama.cpp
        missing, an installation missing its own data -- and 400 when the request is.
        404 is never decided here: an unknown model or profile is raised as a
        :class:`NotFoundError` at the point where the name was looked up, because only
        that call site knows the difference between "no such id" and "that id is fine and
        something else went wrong".
    """
    if isinstance(exc, NotFoundError):
        return 404
    if isinstance(exc, NotInstalledError | ProbeError | PackagedDataError):
        return 503
    return 400


class NotFoundError(CatalogError):
    """A model, profile or file the caller named does not exist.

    It is a :class:`~llamafit.errors.CatalogError` so that everything which already
    catches LlamaFit's own errors keeps catching it, and a subclass so that the API can
    answer 404 rather than 400 without inspecting message text.
    """


def _model_or_404(catalog: Catalog, model_id: str) -> CatalogModel:
    """The model with that id, as a 404 when there is none.

    Args:
        catalog: Every model LlamaFit knows.
        model_id: What the caller asked for.

    Returns:
        The catalog entry.

    Raises:
        NotFoundError: No model has that id. The message is the command line's own,
            closest-ids hint included, so the two interfaces refuse a typo alike.
    """
    try:
        return find_model(catalog, model_id)
    except CatalogError as exc:
        raise NotFoundError(exc.message, hint=exc.hint) from exc


def _checked_profile(name: str | None) -> str | None:
    """A profile's name, refusing anything shaped like a path.

    Args:
        name: What the caller asked for.

    Returns:
        The name, or ``None`` when none was given.

    Raises:
        ConfigError: The value looks like a path. ``--profile`` on the command line does
            take a path, because whoever types it already owns the filesystem; a request
            arriving over a socket does not, so over HTTP a profile is a name and nothing
            else. Names come from ``GET /api/v1/profiles``.
    """
    if name is None:
        return None
    if any(mark in name for mark in ("/", "\\", "..")) or name.endswith(".json"):
        raise ConfigError(
            _("profile must be a name, not a path: %(value)s") % {"value": repr(name)},
            hint=_("Ask GET /api/v1/profiles for the names this machine has."),
        )
    return name


def _host_for(
    dashboard: Dashboard,
    *,
    profile: str | None,
    memory: str | None,
    ram: str | None,
    cpu_cores: int | None,
) -> Host:
    """The machine a request should be scored against.

    Args:
        dashboard: Where the cached scan lives.
        profile: A hardware profile's name, for a machine that is not this one.
        memory: VRAM to pretend the card has, as a size such as ``16G``.
        ram: System memory to pretend the machine has.
        cpu_cores: Physical cores to pretend the processor has.

    Returns:
        The scan, a profile, or either with pools substituted -- marked as simulated
        whenever it is not simply the scan, which is what puts the badge on the page.

    Raises:
        NotFoundError: The profile name matches nothing.
        ConfigError: An override cannot apply to this machine, or is not a size.
    """
    try:
        return resolve_host(
            scan_host=lambda: dashboard.report().host,
            profile=_checked_profile(profile),
            gpu_memory=None if memory is None else check_size(memory, option="memory"),
            ram=None if ram is None else check_size(ram, option="ram"),
            cpu_cores=cpu_cores,
        )
    except ConfigError as exc:
        if profile and "hardware profile" in str(exc.message):
            raise NotFoundError(exc.message, hint=exc.hint) from exc
        raise


def _sorted_rows(rows: Sequence[BoardRow], sort: str) -> list[BoardRow]:
    """Reorder rows the board already ranked, without renumbering them.

    Args:
        rows: The ranked rows.
        sort: One of :data:`SORT_KEYS`, with ``:asc`` or ``:desc`` after it to turn the
            key's own direction round.

    Returns:
        The same rows in the asked-for order. ``rank`` is left alone on purpose: it is the
        position section 11's composite score gave the row, and a caller who sorts by
        speed is choosing what to look at first, not overruling the ranking. A row with
        nothing to sort on -- no placement, no speed -- goes last rather than first,
        whichever direction the key runs.

    The order is :func:`llamafit.cli.render_board.sorted_rows`, the same function
    ``recommend --sort`` and the terminal dashboard's ``s`` key call, so the three
    interfaces cannot put a different row first for the same word.
    """
    key, descending = parse_sort(sort)
    return sorted_rows(rows, key, descending)


@dataclass(frozen=True)
class BoardQuery:
    """One request for a board, already checked and converted.

    Attributes:
        needs: What was asked for, as the services' own model.
        licenses: SPDX identifiers the request accepts; empty accepts any.
        prefer: ``balanced``, ``quality`` or ``speed``.
        all_quants: Every quantisation rather than the best one per model.
        limit: How many ranked rows to keep.
        sort: Which order to return them in.
        vision: Whether to try to keep a vision projector.
        host: The machine to score against.
    """

    needs: Needs
    licenses: tuple[str, ...]
    prefer: str
    all_quants: bool
    limit: int
    sort: str
    vision: bool
    host: Host


def _no_unknown_parameters(request: Request) -> None:
    """Refuse a query parameter no endpoint here understands.

    Args:
        request: The incoming request, for its query string.

    Raises:
        ConfigError: A parameter was given that is not in :data:`BOARD_PARAMETERS`. A
            misspelled name would otherwise be dropped in silence and the caller would
            get a confident answer to a question they did not ask.
    """
    unknown = sorted(set(request.query_params) - BOARD_PARAMETERS)
    if unknown:
        raise ConfigError(
            _("unknown query parameter: %(names)s") % {"names": ", ".join(unknown)},
            hint=_("This endpoint takes: %(names)s")
            % {"names": ", ".join(sorted(BOARD_PARAMETERS))},
        )


def _check_prefer(value: str) -> str:
    """Return the preference asked for, or refuse it and list the three that exist."""
    if value not in PREFERENCES:
        raise ConfigError(
            _("invalid prefer %(value)s") % {"value": repr(value)},
            hint=_("Valid values: %(values)s") % {"values": ", ".join(PREFERENCES)},
        )
    return value


def _check_sort(value: str) -> str:
    """Return the sort asked for, or refuse it and list the keys that exist."""
    try:
        parse_sort(value)
    except ValueError as exc:
        raise ConfigError(
            _("invalid sort %(value)s") % {"value": repr(value)},
            hint=_("Valid values: %(values)s") % {"values": ", ".join(SORT_KEYS)},
        ) from exc
    return value


def create_app(dashboard: Dashboard | None = None, *, extra_hosts: Sequence[str] = ()) -> FastAPI:
    """Build the application: the JSON API, then the page mounted underneath it.

    Args:
        dashboard: The shared scan and catalog; a fresh one when omitted. Tests pass their
            own so nothing probes the machine running them.
        extra_hosts: Host header values to accept beyond the loopback names, which is how
            ``--host`` on a deliberately non-loopback bind stays reachable.

    Returns:
        The application, ready for ``uvicorn`` or for FastAPI's test client.

    The static mount is added last and matches everything left, so ``/`` is the dashboard
    and every path under ``/api/v1`` is still the API: Starlette tries routes in the order
    they were added.
    """
    state = dashboard or Dashboard()
    app = FastAPI(
        title="LlamaFit",
        version=__version__,
        summary="Which open-weight models run on this machine, and how.",
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        redoc_url=None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[*LOOPBACK_HOSTNAMES, *extra_hosts])
    app.state.dashboard = state

    def failure(_request: Request, exc: Exception) -> JSONResponse:
        """Render a LlamaFit error as the one error shape the API publishes."""
        known = exc if isinstance(exc, LlamaFitError) else LlamaFitError(str(exc))
        return JSONResponse(status_code=_status_for(known), content=_error_body(known))

    app.add_exception_handler(LlamaFitError, failure)
    app.include_router(router)
    app.mount(
        "/",
        StaticFiles(
            directory=packaged_dir(
                "llamafit.web.static", what="its web dashboard", contains="index.html"
            ),
            html=True,
        ),
        name="dashboard",
    )
    return app


def dashboard_of(request: Request) -> Dashboard:
    """The scan and the catalog this application shares.

    Args:
        request: The incoming request, which carries the application it arrived at.

    Returns:
        The dashboard :func:`create_app` put on the application.

    It is read off the application rather than closed over by each endpoint because every
    annotation in this module is a string -- ``from __future__ import annotations`` -- and
    FastAPI resolves those against module globals. A dependency defined inside a factory
    is not a module global, so an endpoint written that way is handed a query parameter
    named ``query`` instead of the board it asked for, and answers 422 to every request.
    """
    shared: Dashboard = request.app.state.dashboard
    return shared


Shared = Annotated[Dashboard, Depends(dashboard_of)]
"""The shared scan and catalog, as an endpoint asks for it."""


def board_query(
    request: Request,
    state: Shared,
    use_case: str = "general",
    require: Annotated[list[str] | None, Query()] = None,
    prefer: str = "balanced",
    license_: Annotated[list[str] | None, Query(alias="license")] = None,
    min_context: Annotated[int, Query(ge=0)] = 0,
    max_context: Annotated[int | None, Query(ge=1)] = None,
    max_download: str | None = None,
    all_quants: bool = False,
    limit: Annotated[int, Query(ge=1)] = DEFAULT_LIMIT,
    sort: str = "score",
    vision: bool = True,
    profile: str | None = None,
    memory: str | None = None,
    ram: str | None = None,
    cpu_cores: Annotated[int | None, Query(ge=1)] = None,
) -> BoardQuery:
    """Turn one query string into the arguments ``build_board`` already takes.

    Args:
        request: The incoming request, so a misspelled parameter can be refused by name.
        state: The shared scan, which a simulation override is applied on top of.
        use_case: What the model is for, which sets the weights and the speed target.
        require: A capability the model must have; repeatable.
        prefer: ``balanced``, ``quality`` or ``speed``.
        license_: An SPDX identifier the request will accept; repeatable, and spelled
            ``license`` in the query string.
        min_context: Exclude candidates that cannot hold this many tokens.
        max_context: Plan, report and score no context longer than this.
        max_download: Exclude anything larger to fetch, as a size such as ``40G``.
        all_quants: Every quantisation rather than the best one per model.
        limit: How many ranked rows to keep.
        sort: Which order to return them in.
        vision: Whether to try to keep a vision projector.
        profile: A profile's name, to score against a machine that is not this one.
        memory: VRAM to pretend the card has.
        ram: System memory to pretend the machine has.
        cpu_cores: Physical cores to pretend the processor has.

    Returns:
        The request, checked and converted.

    Every value is checked by the command line's own checker, so a bad ``use_case`` or a
    bad ``require`` is refused here in exactly the words ``llamafit recommend`` refuses it
    in, and there is no second opinion anywhere about what a capability is.
    """
    _no_unknown_parameters(request)
    return BoardQuery(
        needs=Needs(
            use_case=check_use_case(use_case),
            capabilities=check_capabilities(require or [], option="require"),
            min_context=min_context,
            max_context=check_context_ceiling(
                max_context,
                min_context=min_context,
                ceiling_option="max_context",
                floor_option="min_context",
            ),
            max_download_bytes=(
                None if max_download is None else check_size(max_download, option="max_download")
            ),
        ),
        licenses=tuple(license_ or []),
        prefer=_check_prefer(prefer),
        all_quants=all_quants,
        limit=limit,
        sort=_check_sort(sort),
        vision=vision,
        host=_host_for(state, profile=profile, memory=memory, ram=ram, cpu_cores=cpu_cores),
    )


Asked = Annotated[BoardQuery, Depends(board_query)]
"""One checked board request, as an endpoint asks for it."""


def _board(state: Dashboard, query: BoardQuery, *, all_quants: bool) -> Response:
    """Rank the catalog for one request and serialise the board the service returned."""
    report = state.report()
    board = build_board(
        state.catalog(),
        query.host,
        query.needs,
        all_quants=all_quants,
        limit=query.limit,
        local_models=report.llamacpp.local_models,
        licenses=query.licenses,
        prefer=query.prefer,
        vision=query.vision,
    )
    if query.sort == "score":
        return _json(board)
    return _json(board.model_copy(update={"rows": _sorted_rows(board.rows, query.sort)}))


router = APIRouter()
"""Every endpoint the dashboard serves; :func:`create_app` mounts the page under it."""


@router.get("/health")
def health() -> dict[str, str]:
    """Say the server is up and which version it is."""
    return {"status": "ok", "version": __version__}


@router.get(f"{API_PREFIX}/ui")
def ui() -> JSONResponse:
    """Every word the page draws, in the language this process speaks."""
    return JSONResponse(content=ui_payload())


@router.get(f"{API_PREFIX}/system")
def system(state: Shared) -> Response:
    """The machine and its llama.cpp installation, from the cached scan."""
    return _json(state.report())


@router.post(f"{API_PREFIX}/scan")
def rescan(state: Shared) -> Response:
    """Scan the machine again and return what it found."""
    return _json(state.report(refresh=True))


@router.get(f"{API_PREFIX}/doctor")
def doctor(state: Shared) -> Response:
    """Probe by probe: what worked, what did not, and what would unlock more."""
    return _json(diagnose(state.report()))


@router.get(f"{API_PREFIX}/models")
def models(state: Shared, query: Asked) -> Response:
    """The catalog with per-host scoring: every quantisation of every model, ranked."""
    return _board(state, query, all_quants=True)


@router.get(f"{API_PREFIX}/models/top")
def top(state: Shared, query: Asked) -> Response:
    """The board: the best quantisation per model, ranked."""
    return _board(state, query, all_quants=query.all_quants)


@router.get(f"{API_PREFIX}/models/{{model_id}}")
def model_detail(state: Shared, model_id: str) -> Response:
    """One model: its entry, every quantisation it publishes, and the facts read."""
    report = state.report()
    model = _model_or_404(state.catalog(), model_id)
    return _json(describe(model, report.llamacpp.local_models))


@router.post(f"{API_PREFIX}/plan")
def plan(state: Shared, body: PlanRequest) -> Response:
    """Place one model on this machine and return the command line that runs it."""
    report = state.report()
    model = _model_or_404(state.catalog(), body.model_id)
    host = _host_for(
        state,
        profile=body.profile,
        memory=body.memory,
        ram=body.ram,
        cpu_cores=body.cpu_cores,
    )
    chosen = choose_quant(model, host, body.quant)
    built: PlanReport = plan_report(
        model,
        chosen,
        host,
        needs=Needs(use_case=model.use_cases[0], requested_context=body.context),
        vision=body.vision,
        micro_batch=body.ub,
        target_tps=body.target_tps,
        local_files=[local.path for local in report.llamacpp.local_models],
    )
    return _json(built)


@router.get(f"{API_PREFIX}/profiles")
def profiles() -> JSONResponse:
    """The hardware profiles this machine has, bundled and the user's own.

    The shape is the one ``llamafit hardware list --json`` prints, field for field, so a
    script that reads one reads the other.
    """
    loaded, _problems = load_profiles()
    return JSONResponse(
        content=[
            {
                "name": profile.name,
                "bundled": profile.bundled,
                "path": str(profile.path),
                "profile": profile.profile.model_dump(mode="json"),
            }
            for profile in loaded
        ]
    )


@router.get(f"{API_PREFIX}/catalog/schema")
def catalog_schema() -> JSONResponse:
    """The JSON schema of a catalog entry, for somebody else's tooling."""
    text = packaged_text("llamafit.data.schema", "catalog.schema.json", what="its catalog schema")
    return JSONResponse(content=json.loads(text))


@router.get(f"{API_PREFIX}/catalog/problems")
def catalog_problems(state: Shared) -> JSONResponse:
    """What was wrong with the catalog files, so the page can say so once.

    The sentence is built here, with the command line's own ``ngettext`` call, rather than
    assembled in the browser from a count and a noun: the languages LlamaFit speaks do not
    agree on how many plural forms a number has, and JavaScript has no catalog to ask.
    """
    problems = state.problems()
    return JSONResponse(
        content={
            "count": len(problems),
            "message": ngettext(
                "%(count)d catalog problem found; run `llamafit catalog validate` for details.",
                "%(count)d catalog problems found; run `llamafit catalog validate` for details.",
                len(problems),
            )
            % {"count": len(problems)},
            "problems": [
                {
                    "file": problem.file,
                    "model_id": problem.model_id,
                    "location": problem.location,
                    "message": problem.message,
                }
                for problem in problems
            ],
        }
    )


def endpoints(app: FastAPI) -> Iterator[str]:
    """Every API path the application serves, for a test that asks what shipped.

    Args:
        app: The application.

    Yields:
        Each route's path, including the ones an included router holds. FastAPI keeps a
        router included with ``include_router`` as a single opaque route that carries the
        real ones on ``original_router``, so a walk that read only the top level would
        report an application with no endpoints at all -- and a test written on that
        would pass for the wrong reason.
    """

    def walk(routes: Iterable[object]) -> Iterator[str]:
        for route in routes:
            included = getattr(route, "original_router", None)
            nested = getattr(included, "routes", None) or getattr(route, "routes", None)
            if nested:
                yield from walk(nested)
            path = getattr(route, "path", None)
            if isinstance(path, str) and path:
                yield path

    yield from walk(app.routes)


__all__ = [
    "API_PREFIX",
    "BOARD_PARAMETERS",
    "LOOPBACK_HOSTNAMES",
    "SORT_KEYS",
    "Dashboard",
    "NotFoundError",
    "PlanRequest",
    "create_app",
    "endpoints",
]

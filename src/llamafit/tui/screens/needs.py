# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The Needs form: section 12.1's request, as questions rather than as flags.

``llamafit recommend --use-case coding --require tools --min-context 32768 --license
Apache-2.0`` is a good command line and a bad first encounter. Every one of those values
is a thing the program already knows the whole list of, and a person who does not know the
list cannot type a member of it.

So each field here offers what exists. The six use cases, the eight capabilities and the
licences the catalog actually carries are all read from the catalog's own types and
entries, never typed out again, which is also why a capability added tomorrow appears on
this form without anybody remembering to add it.

Three fields still take free text -- the minimum context in tokens, a download ceiling
like ``40G``, and the slowest generation worth having -- because none has a list to
offer. Each is checked before it is applied, and a value that is not a size or a number
says so on the form rather than raising past it.
"""

from __future__ import annotations

from typing import ClassVar, get_args

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Checkbox, Input, Label, Select, SelectionList, Static

from llamafit.i18n import _
from llamafit.models.catalog import Capability, UseCase
from llamafit.models.plan import Needs
from llamafit.scoring import PREFERENCES
from llamafit.tui.keys import NEEDS_KEYS, bindable
from llamafit.tui.labels import capability_label, prefer_label, use_case_label
from llamafit.tui.state import Dashboard, Request
from llamafit.units import format_grouped, parse_size

USE_CASES: tuple[str, ...] = get_args(UseCase)
"""Every use case, taken from the catalog's own type rather than listed again."""

CAPABILITIES: tuple[str, ...] = get_args(Capability)
"""Every capability, taken from the catalog's own type rather than listed again."""


class NeedsPane(VerticalScroll):
    """What is being asked for, as a form the board re-ranks against."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key.name, key.action, show=False) for key in bindable(NEEDS_KEYS)
    ]

    DEFAULT_CSS = """
    NeedsPane { height: 1fr; padding: 0 1; }
    NeedsPane > Label { margin-top: 1; text-style: bold; }
    NeedsPane > #needs-capabilities { height: 6; }
    NeedsPane > #needs-licenses { height: 5; }
    NeedsPane > #needs-error { color: $error; height: auto; }
    NeedsPane > #needs-buttons { height: auto; margin-top: 1; }
    """

    class Applied(Message):
        """The request changed, so whatever draws the board should draw it again."""

    def __init__(self, dashboard: Dashboard, *, id: str | None = None) -> None:  # noqa: A002
        super().__init__(id=id)
        self.dashboard = dashboard

    def compose(self) -> ComposeResult:
        """One field per part of section 12.1's request, each offering what exists."""
        yield Label(_("What do you want a model for?"), markup=False)
        yield Select[str](
            [(use_case_label(name), name) for name in USE_CASES],
            value="general",
            allow_blank=False,
            id="needs-use-case",
        )
        yield Label(_("It must be able to:"), markup=False)
        yield SelectionList[str](
            *[(capability_label(name), name) for name in CAPABILITIES],
            id="needs-capabilities",
        )
        yield Label(_("At least this much context, in tokens:"), markup=False)
        yield Input(value="0", id="needs-min-context")
        yield Label(_("Willing to download at most, for example 40G (blank for no limit):"))
        yield Input(value="", id="needs-max-download")
        yield Label(
            _(
                "Slowest generation worth having, in tokens per second (blank for the speed "
                "a person reads at; 0 when nobody is waiting on the tokens):"
            ),
            markup=False,
        )
        yield Input(value="", id="needs-min-tps")
        yield Label(_("Licences you will accept (tick none to accept any):"), markup=False)
        yield SelectionList[str](id="needs-licenses")
        yield Label(_("Lean the score:"), markup=False)
        yield Select[str](
            [(prefer_label(name), name) for name in PREFERENCES],
            value="balanced",
            allow_blank=False,
            id="needs-prefer",
        )
        yield Checkbox(
            _("Keep the vision projector when a model has one"), value=True, id="needs-vision"
        )
        yield Static("", id="needs-error", markup=False)
        with Horizontal(id="needs-buttons"):
            yield Button(_("Apply"), variant="primary", id="needs-apply")
            yield Button(_("Reset"), id="needs-reset")

    # --- drawing -----------------------------------------------------------------------

    def refresh_needs(self) -> None:
        """Offer the licences the loaded catalog actually carries, and show the request.

        The list cannot be built when the form is composed, because the catalog is loaded
        on a thread and may not have arrived. Offering the licences of models nobody has
        would be worse than offering none: a filter that can only ever exclude everything.
        """
        catalog = self.dashboard.catalog
        licenses = sorted({model.license.spdx for model in catalog.models}) if catalog else []
        chooser = self.query_one("#needs-licenses", SelectionList)
        chosen = set(self.dashboard.request.licenses)
        chooser.clear_options()
        chooser.add_options([(spdx, spdx, spdx in chosen) for spdx in licenses])

    # --- what the reader does ----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Apply or reset, whichever button was pressed."""
        event.stop()
        if event.button.id == "needs-apply":
            self.action_apply()
        else:
            self.action_reset()

    def action_apply(self) -> None:
        """Read the form, re-rank against it, and say so when a field cannot be read."""
        error = self.query_one("#needs-error", Static)
        try:
            request = self._request()
        except ValueError as exc:
            error.update(str(exc))
            return
        error.update("")
        self.dashboard.ask(request)
        self.post_message(self.Applied())

    def action_reset(self) -> None:
        """Put every field back to what a request asks for when it asks for nothing."""
        self.query_one("#needs-use-case", Select).value = "general"
        self.query_one("#needs-capabilities", SelectionList).deselect_all()
        self.query_one("#needs-min-context", Input).value = "0"
        self.query_one("#needs-max-download", Input).value = ""
        self.query_one("#needs-min-tps", Input).value = ""
        self.query_one("#needs-licenses", SelectionList).deselect_all()
        self.query_one("#needs-prefer", Select).value = "balanced"
        self.query_one("#needs-vision", Checkbox).value = True
        self.query_one("#needs-error", Static).update("")
        self.dashboard.ask(Request())
        self.post_message(self.Applied())

    # --- reading the form --------------------------------------------------------------

    def _request(self) -> Request:
        """The form as a :class:`~llamafit.tui.state.Request`.

        Raises:
            ValueError: A typed field holds something that is not what it asks for. The
                message is what the form shows; it names the field, because "not a size"
                with two size fields on screen tells a reader half of what they need.
        """
        capabilities = self.query_one("#needs-capabilities", SelectionList).selected
        licenses = self.query_one("#needs-licenses", SelectionList).selected
        return Request(
            needs=Needs(
                use_case=str(self.query_one("#needs-use-case", Select).value),
                capabilities=tuple(str(name) for name in capabilities),
                min_context=self._tokens(),
                min_tps=self._min_tps(),
                max_download_bytes=self._download_ceiling(),
            ),
            prefer=str(self.query_one("#needs-prefer", Select).value),
            licenses=tuple(str(spdx) for spdx in licenses),
            vision=bool(self.query_one("#needs-vision", Checkbox).value),
            all_quants=self.dashboard.request.all_quants,
        )

    def _tokens(self) -> int:
        """The minimum context, as a whole number of tokens that is not negative."""
        typed = self.query_one("#needs-min-context", Input).value.strip()
        if not typed:
            return 0
        try:
            tokens = int(typed.replace(" ", ""))
        except ValueError as exc:
            raise ValueError(
                _("The minimum context is a number of tokens, for example %(example)s.")
                % {"example": format_grouped(32768)}
            ) from exc
        if tokens < 0:
            raise ValueError(_("The minimum context cannot be below zero."))
        return tokens

    def _min_tps(self) -> float | None:
        """The speed floor in tokens per second, or ``None`` for the reading floor.

        Zero is kept apart from blank, as ``--min-tps`` keeps it: blank means the figure
        the specification derived from reading rates, zero means nobody is waiting.
        """
        typed = self.query_one("#needs-min-tps", Input).value.strip()
        if not typed:
            return None
        try:
            tps = float(typed.replace(",", "."))
        except ValueError as exc:
            raise ValueError(
                _("The minimum speed is a number of tokens per second, for example 12.")
            ) from exc
        if tps < 0:
            raise ValueError(_("The minimum speed cannot be below zero."))
        return tps

    def _download_ceiling(self) -> int | None:
        """The download ceiling in bytes, or ``None`` when the field is empty."""
        typed = self.query_one("#needs-max-download", Input).value.strip()
        if not typed:
            return None
        try:
            return parse_size(typed)
        except ValueError as exc:
            raise ValueError(
                _("Sizes look like 8G, 7.5GiB or 512M; a bare number is bytes.")
            ) from exc

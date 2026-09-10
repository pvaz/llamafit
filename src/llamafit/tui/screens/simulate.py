# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Simulate: answer for a machine that is not this one, and never stop saying so.

Two substitutions, and section 4.4 already decided what each means. A profile replaces the
machine outright -- somebody else's fleet, or the reference machine this project's numbers
were measured on. An override moves one pool of the machine that is here: what would a
24 GB card run, what would half the memory cost me.

Nothing here works out what an override implies. ``resolve_host`` does that, the arithmetic
is written down once in ``hwprofile/simulate.py``, and the substituted host carries a
:class:`~llamafit.models.host.Simulation` that every screen downstream reads. That is why
the badge in the header is not something this form remembers to switch on: it is on the
host, so it reaches the board, the plan and ``--json`` alike, and a screen cannot forget it.

A substitution that cannot be made is refused here rather than half-applied. A card cannot
be resized on a machine that has none, and a profile can name a file nobody wrote; either
way the form says so and the machine stays what it was.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Label, Select, Static

from llamafit.errors import LlamaFitError
from llamafit.hwprofile.loader import LoadedProfile, load_profiles
from llamafit.i18n import _, pgettext
from llamafit.tui.keys import SIMULATE_KEYS, bindable
from llamafit.tui.state import Dashboard, Substitution
from llamafit.units import parse_size

THIS_MACHINE = ""
"""The profile value meaning "do not replace the machine", which is what a scan is."""


def available_profiles() -> tuple[list[LoadedProfile], LlamaFitError | None]:
    """Every bundled and user profile, or the error that stopped them being read.

    A dashboard on an installation missing its packaged data still has a Simulate screen;
    it just has nothing to offer on it, and says which.
    """
    try:
        profiles, _problems = load_profiles()
    except LlamaFitError as exc:
        return [], exc
    return profiles, None


class SimulatePane(VerticalScroll):
    """Override a pool, or stand in a whole profile, and re-rank against the result."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key.name, key.action, show=False) for key in bindable(SIMULATE_KEYS)
    ]

    DEFAULT_CSS = """
    SimulatePane { height: 1fr; padding: 0 1; }
    SimulatePane > Label { margin-top: 1; text-style: bold; }
    SimulatePane > #simulate-error { color: $error; height: auto; }
    SimulatePane > #simulate-buttons { height: auto; margin-top: 1; }
    """

    class Applied(Message):
        """The machine being answered about changed, so every screen should be redrawn."""

    def __init__(self, dashboard: Dashboard, *, id: str | None = None) -> None:  # noqa: A002
        super().__init__(id=id)
        self.dashboard = dashboard

    def compose(self) -> ComposeResult:
        """The profile chooser, the three overrides, and the two buttons."""
        yield Static(
            _(
                "Everything on every screen answers about the machine chosen here. While it "
                "is not this one, the band at the top says SIMULATED."
            ),
            id="simulate-intro",
            markup=False,
        )
        yield Label(_("Answer for this machine instead:"), markup=False)
        yield Select[str](
            [(pgettext("hardware profile", "this machine"), THIS_MACHINE)],
            value=THIS_MACHINE,
            allow_blank=False,
            id="simulate-profile",
        )
        yield Label(_("Pretend the graphics card has this much memory (blank to leave it):"))
        yield Input(placeholder="24G", id="simulate-vram")
        yield Label(_("Pretend the machine has this much system memory (blank to leave it):"))
        yield Input(placeholder="64G", id="simulate-ram")
        yield Label(_("Pretend the processor has this many cores (blank to leave it):"))
        yield Input(placeholder="8", id="simulate-cores")
        yield Static("", id="simulate-error", markup=False)
        with Horizontal(id="simulate-buttons"):
            yield Button(_("Apply"), variant="primary", id="simulate-apply")
            yield Button(_("Back to this machine"), id="simulate-reset")

    # --- drawing -----------------------------------------------------------------------

    def refresh_profiles(self) -> None:
        """Offer the profiles that are actually installed, or say why there are none."""
        profiles, problem = available_profiles()
        chooser = self.query_one("#simulate-profile", Select)
        current = chooser.value
        options = [(pgettext("hardware profile", "this machine"), THIS_MACHINE)]
        options += [(profile.name, profile.name) for profile in profiles]
        chooser.set_options(options)
        chooser.value = current if current in {value for _label, value in options} else THIS_MACHINE
        if problem is not None:
            self.query_one("#simulate-error", Static).update(problem.render())

    # --- what the reader does ----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Apply or go back, whichever button was pressed."""
        event.stop()
        if event.button.id == "simulate-apply":
            self.action_apply()
        else:
            self.action_reset()

    def action_apply(self) -> None:
        """Read the form and answer for that machine, or say why it could not be built."""
        error = self.query_one("#simulate-error", Static)
        try:
            substitution = self._substitution()
        except ValueError as exc:
            error.update(str(exc))
            return
        refused = self.dashboard.substitute(substitution)
        if refused is not None:
            error.update(refused.render())
            return
        error.update("")
        self.post_message(self.Applied())

    def action_reset(self) -> None:
        """Go back to the machine LlamaFit is running on, and clear every field."""
        self.query_one("#simulate-profile", Select).value = THIS_MACHINE
        for field in ("#simulate-vram", "#simulate-ram", "#simulate-cores"):
            self.query_one(field, Input).value = ""
        self.query_one("#simulate-error", Static).update("")
        self.dashboard.substitute(Substitution())
        self.post_message(self.Applied())

    # --- reading the form --------------------------------------------------------------

    def _substitution(self) -> Substitution:
        """The form as a :class:`~llamafit.tui.state.Substitution`.

        Raises:
            ValueError: A field holds something that is not a size or not a count.
        """
        profile = str(self.query_one("#simulate-profile", Select).value)
        return Substitution(
            profile=profile or None,
            gpu_memory=self._bytes_field("#simulate-vram"),
            ram=self._bytes_field("#simulate-ram"),
            cpu_cores=self._cores(),
        )

    def _bytes_field(self, field: str) -> int | None:
        """One size field in bytes, or ``None`` when it is empty."""
        typed = self.query_one(field, Input).value.strip()
        if not typed:
            return None
        try:
            return parse_size(typed)
        except ValueError as exc:
            raise ValueError(
                _("Sizes look like 8G, 7.5GiB or 512M; a bare number is bytes.")
            ) from exc

    def _cores(self) -> int | None:
        """The core count, or ``None`` when the field is empty."""
        typed = self.query_one("#simulate-cores", Input).value.strip()
        if not typed:
            return None
        try:
            return int(typed)
        except ValueError as exc:
            raise ValueError(_("The core count is a whole number, for example 8.")) from exc

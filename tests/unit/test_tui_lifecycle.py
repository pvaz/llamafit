"""Delayed navigation events must not redraw widgets that shutdown has removed."""

from __future__ import annotations

from typing import Any, cast

import pytest
from textual.widgets import TabbedContent

from llamafit.tui.app import LlamaFitApp
from tests.fixtures.dashboard import ready


@pytest.mark.asyncio
async def test_a_pending_tab_activation_during_shutdown_does_not_redraw_removed_widgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = LlamaFitApp(ready())
    delivered = False
    async with app.run_test(size=(120, 44)):
        tabs = app.query_one("#tabs", TabbedContent)
        event = TabbedContent.TabActivated(tabs, cast(Any, tabs.get_tab("tab-board")))
        close_all = app._close_all

        async def close_with_pending_tab_activation() -> None:
            nonlocal delivered
            # Force the CI ordering: shutdown has started, the tab widget is gone,
            # and an already queued activation reaches the app's message handler.
            assert not app.is_running
            try:
                await tabs.remove()
                await app._dispatch_message(event)
                delivered = True
            finally:
                await close_all()

        monkeypatch.setattr(app, "_close_all", close_with_pending_tab_activation)

    assert delivered

"""Test-wide defaults.

Two things a test would otherwise inherit from the machine it runs on, and neither is
ever what the test meant. Both are pinned here, for the same reason: an assertion about
what LlamaFit printed should fail because the program printed the wrong thing, not
because the developer's computer differs from the build server's.

**The language.** Without this the suite reads the machine's own user interface language,
so a developer on a Portuguese Windows sees every command render in Portuguese and every
assertion about English output fail, while the same commit passes in CI.

**Whether output is coloured.** This one cost a release rehearsal. Seventeen tests across
six files passed on a developer's machine and failed on all six CI jobs, and the reason
was that GitHub Actions sets ``GITHUB_ACTIONS`` and ``CI``, which Rich reads as "this is a
terminal that wants colour". Every assertion that compares printed text then meets
``\\x1b[1m`` where it expected a word: ``"--host" in output`` is false, and a list of
drawn table cells compares as escape sequences against plain strings. Nothing was wrong
with the software. ``NO_COLOR`` is the convention Rich, Click and this program all honour,
and the three variables that argue with it are removed rather than left to win by
accident.

A test that is *about* colour sets its own environment and overrides both of these; the
point of a default is that it is what a test gets when it has not said.
"""

from collections.abc import Iterator

import pytest

from llamafit.i18n import LANGUAGE_ENV_VAR, translator

_COLOUR_VARIABLES = ("FORCE_COLOR", "GITHUB_ACTIONS", "CI")
"""What turns colour on behind a test's back. Removed so ``NO_COLOR`` is uncontested."""


@pytest.fixture(autouse=True)
def _english_unless_a_test_says_otherwise(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(LANGUAGE_ENV_VAR, "en")
    translator.reset()
    yield
    translator.reset()


@pytest.fixture(autouse=True)
def _plain_text_unless_a_test_says_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """No colour, so an assertion about text meets text."""
    for name in _COLOUR_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

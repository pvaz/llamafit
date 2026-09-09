"""Test-wide defaults.

Every test runs in English unless it says otherwise. Without this the suite would read
the machine's own user interface language, so a developer on a Portuguese Windows would
see every command render in Portuguese and every assertion about English output fail,
while the same commit passed in CI. The language a test wants is a thing a test states.
"""

from collections.abc import Iterator

import pytest

from llamafit.i18n import LANGUAGE_ENV_VAR, translator


@pytest.fixture(autouse=True)
def _english_unless_a_test_says_otherwise(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(LANGUAGE_ENV_VAR, "en")
    translator.reset()
    yield
    translator.reset()

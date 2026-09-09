"""The package exposes a PEP 440 version."""

import re

import llamafit


def test_version_is_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([a-z]+\d+)?", llamafit.__version__)

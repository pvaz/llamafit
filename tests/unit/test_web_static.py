"""The three files the browser gets, and the promises they have to keep on their own.

``tests/unit/test_licensing.py`` walks ``*.py`` and would never see these, so the notice
the AGPL asks every file to carry is checked here instead, in each language's own comment
syntax. So is the harder promise: that nothing on the page is fetched from the internet
when it is opened. A dashboard that pulled a font or a script from a stranger's server
would stop working with the network unplugged and would tell that server, every time,
that somebody had opened it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llamafit.web import strings

STATIC = Path(strings.__file__).parent / "static"
ASSETS = ("index.html", "styles.css", "app.js")

NOTICE = (
    "LlamaFit. Copyright (C) 2026 Paulo Vaz.",
    "SPDX-License-Identifier: AGPL-3.0-or-later",
    "This file is part of LlamaFit; see LICENSE for the full terms",
)
"""The three lines every Python module carries, in a comment the browser's parser accepts.

An HTML comment for the page, a ``/* */`` block for the stylesheet and ``//`` lines for the
script: the same three sentences, because a file somebody copies out of this project must
say what it is wherever they paste it, and none of these is a ``.py`` file the licensing
test would have caught.
"""


@pytest.mark.parametrize("name", ASSETS)
def test_every_asset_carries_the_licence_notice(name: str) -> None:
    head = (STATIC / name).read_text(encoding="utf-8")[:1200]
    for line in NOTICE:
        assert line in head, f"{name} does not carry: {line}"


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
"""An XML namespace name, which browsers never dereference; it is an identifier, not a URL."""


@pytest.mark.parametrize("name", ASSETS)
def test_no_asset_reaches_the_internet_to_draw_the_page(name: str) -> None:
    """Only the source link may name another host, and a link the reader clicks is not a fetch."""
    text = (STATIC / name).read_text(encoding="utf-8")
    for url in re.findall(r"https?://[^\s\"'<>)]+", text):
        allowed = url.startswith(strings.SOURCE_URL) or url == SVG_NAMESPACE
        assert allowed, f"{name} reaches {url}"


def _code(name: str) -> str:
    """One asset with its comments taken out.

    Every rule below is about what the file *does*, and each of these files explains in a
    comment why it does not do the thing the rule forbids. Reading the prose as code would
    fail the test on the sentence promising to pass it.
    """
    text = (STATIC / name).read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)


def test_the_stylesheet_imports_no_font_and_no_stylesheet() -> None:
    css = _code("styles.css")
    assert "@import" not in css
    assert "url(" not in css, "an embedded url() is a request the page would make"


def test_the_page_loads_only_its_own_two_files() -> None:
    """One script, one stylesheet, and an icon drawn inline rather than fetched."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.findall(r'<script[^>]*src="([^"]+)"', html) == ["app.js"]
    linked = {}
    for tag in re.findall(r"<link\b[^>]*>", html, flags=re.DOTALL):
        relation = re.search(r'rel="([a-z]+)"', tag)
        target = re.search(r'href="([^"]*)"', tag)
        assert relation and target, tag
        linked[relation.group(1)] = target.group(1)
    assert set(linked) == {"icon", "stylesheet"}
    assert linked["stylesheet"] == "styles.css"
    assert linked["icon"].startswith("data:image/svg+xml,"), "the icon must not be a request"


def test_the_page_carries_no_words_of_its_own() -> None:
    """Every visible string is a key the server fills in, or the page is English-only."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    body = html[html.index("<body>") :]
    without_comments = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", without_comments)
    words = [word for word in text.split() if word.isalpha() and len(word) > 2]
    assert words == ["LlamaFit"], f"untranslatable text on the page: {words}"


def test_the_script_builds_the_page_from_text_rather_than_markup() -> None:
    """A model id or a file path off somebody's disk must never become markup."""
    js = _code("app.js")
    assert "innerHTML" not in js
    assert "insertAdjacentHTML" not in js
    assert "eval(" not in js


def test_the_footer_offers_the_source_as_section_13_asks() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'data-t="app.source"' in html
    assert strings.SOURCE_URL in html


def test_the_page_is_small_enough_that_somebody_will_read_it() -> None:
    """A contributor who cannot change the dashboard has a framework, not a page.

    The ceiling was 1,500 while the board drew seven columns and explained itself in four
    paragraphs above the list. It draws thirteen now, keeps every one of them for the
    candidates it could not rank, folds the prose into a dialog and holds the whole page
    to one screen -- all of which is page code and all of which was asked for. 1,800 is
    the new ceiling and the same rule: a page a contributor cannot read whole is a
    framework, and a number that moves whenever it is reached is not a ceiling. Cut
    before raising it again.
    """
    lines = sum(len((STATIC / name).read_text(encoding="utf-8").splitlines()) for name in ASSETS)
    assert lines < 1800, f"the dashboard has grown to {lines} lines; split it or cut it"

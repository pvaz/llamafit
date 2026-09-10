"""Check that every catalog entry's cited sources are still reachable.

Marked ``network``: this needs internet access and is skipped in CI, the same as
``hardware``. Run it deliberately after adding or editing a catalog entry, for example
with ``pytest -m network tests/unit/test_catalog_sources.py -v``, so a link that rots is
found on purpose rather than by accident.

Two things this test learned when the catalog grew past forty models.

**A rate limit is not a dead link.** Most sources are Hugging Face model cards, and asking
that host two hundred questions in twenty seconds earns an HTTP 429. Reporting that as
"source does not resolve" would be the tool telling a curator to fix an entry that is
perfectly correct, which is the one kind of error this project refuses to make. A throttled
URL is reported as unchecked and the test says how many it could not reach.

**Sources repeat.** A model's four benchmarks usually cite one model card between them, and
several models in a family cite the same technical report. Checking the set rather than the
list turns two hundred requests into a few dozen, which is both faster and most of why the
throttling stops happening at all.

A model card on Hugging Face is asked about through the API rather than by fetching the
page. It is the same question -- does this repository exist -- put to the endpoint built to
answer it, and it answers when the rendered page is busy turning us away. Nothing else is
special-cased: every other host is fetched as written.
"""

from __future__ import annotations

import time

import httpx
import pytest

from llamafit.catalog.loader import load_catalog

pytestmark = pytest.mark.network

_HEADERS = {"User-Agent": "llamafit-catalog-check/0.1 (+https://github.com/pvaz/llamafit)"}

_WITHHELD = frozenset({401, 403, 429})
"""The statuses that mean "the host will not answer" rather than "this is not here".

A dead link is a 404, a 410, or a name that does not resolve. These three are a server
that is present and declining: a login wall, a bot filter, a rate limit. Bot filters are
not a curiosity here -- one licence in the catalog is served behind Cloudflare, which
returns 200 to a browser and to curl and 403 to this checker whatever headers it sends,
because the block is on the TLS handshake rather than on anything a request can change.
Calling that a broken link would send a curator to fix an entry that is perfectly correct.
"""

_UNCHECKED = "withheld"
"""What :func:`_check` returns when the host declined to answer both times."""

_RETRY_PAUSE = 5.0
"""Seconds to wait before the single retry a throttled host is given."""

_POLITE_PAUSE = 0.2
"""Seconds between requests, so a catalog of any size stays a reasonable visitor."""

_HF_PAGE = "https://huggingface.co/"
_HF_API = "https://huggingface.co/api/models/"


def _askable(url: str) -> str:
    """Return the URL to actually request for ``url``.

    A Hugging Face model card is asked about through the API, which answers the same
    question -- does this repository exist -- and keeps answering while the rendered page
    is rate-limiting us. A URL that carries a path inside the repository (a file, a
    revision, a discussion) is left alone, because the API would answer about the
    repository and not about the thing that was cited.

    Args:
        url: The source exactly as the catalog entry writes it.

    Returns:
        The URL to fetch, which is ``url`` itself for everything but a bare model card.
    """
    if not url.startswith(_HF_PAGE):
        return url
    path = url[len(_HF_PAGE) :].rstrip("/")
    owner, _, name = path.partition("/")
    if not owner or not name or "/" in name or not path or path.startswith("datasets/"):
        return url
    return _HF_API + path


def _check(url: str, client: httpx.Client) -> str | None:
    """What went wrong fetching ``url``, or ``None`` when nothing did.

    Args:
        url: The source to fetch.
        client: The shared client, so connections and headers are reused.

    Returns:
        ``None`` when the URL resolved, ``_UNCHECKED`` when the host declined to answer
        both times, and otherwise the status or the transport error, so a curator is told
        what happened rather than only that something did. "Does not resolve" without a
        reason sends somebody to re-check a link by hand.
    """
    asked = _askable(url)
    trouble = "no attempt was made"
    withheld = False
    for attempt in range(2):
        try:
            response = client.get(asked, follow_redirects=True, timeout=15.0)
        except httpx.HTTPError as error:
            trouble = f"{type(error).__name__}: {error}"
        else:
            if response.status_code not in _WITHHELD:
                return None if response.is_success else f"HTTP {response.status_code}"
            withheld = True
            trouble = f"HTTP {response.status_code}"
        if attempt == 0:
            time.sleep(_RETRY_PAUSE)
    return _UNCHECKED if withheld else trouble


def test_every_licence_and_benchmark_source_resolves() -> None:
    """Every bundled entry's ``license.url`` and every ``benchmark.source`` must resolve."""
    catalog, problems = load_catalog(custom_path=None)
    assert problems == []

    # Where each URL came from, so a failure names the entry a curator has to open rather
    # than only the link. One URL can serve several entries, and then it names them all.
    cited: dict[str, list[str]] = {}
    for model in catalog.models:
        cited.setdefault(model.license.url, []).append(f"{model.id}: license.url")
        for benchmark in model.quality.benchmarks:
            cited.setdefault(benchmark.source, []).append(f"{model.id}/{benchmark.name}")

    failures: list[str] = []
    unchecked = 0
    with httpx.Client(headers=_HEADERS) as client:
        for url, where in sorted(cited.items()):
            trouble = _check(url, client)
            if trouble == _UNCHECKED:
                unchecked += 1
            elif trouble is not None:
                failures.extend(f"{cite}: {url} -> {trouble}" for cite in where)
            time.sleep(_POLITE_PAUSE)

    assert not failures, "\n".join(failures)
    if unchecked:
        pytest.skip(f"{unchecked} of {len(cited)} sources withheld an answer, not checked")

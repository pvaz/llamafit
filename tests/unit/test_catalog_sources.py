"""Check that every catalog entry's cited sources are still reachable.

Marked ``network``: this needs internet access and is skipped in CI, the same as
``hardware``. Run it deliberately after adding or editing a catalog entry, for example
with ``pytest -m network tests/unit/test_catalog_sources.py -v``, so a link that rots is
found on purpose rather than by accident.
"""

from __future__ import annotations

import httpx
import pytest

from llamafit.catalog.loader import load_catalog

pytestmark = pytest.mark.network

_HEADERS = {"User-Agent": "llamafit-catalog-check/0.1 (+https://github.com/pvaz/llamafit)"}


def _resolves(url: str, client: httpx.Client) -> bool:
    """Whether ``url`` returns a successful status, following redirects."""
    try:
        response = client.get(url, follow_redirects=True, timeout=15.0)
    except httpx.HTTPError:
        return False
    return response.is_success


def test_every_licence_and_benchmark_source_resolves() -> None:
    """Every bundled entry's ``license.url`` and every ``benchmark.source`` must resolve."""
    catalog, problems = load_catalog(custom_path=None)
    assert problems == []

    failures: list[str] = []
    with httpx.Client(headers=_HEADERS) as client:
        for model in catalog.models:
            if not _resolves(model.license.url, client):
                failures.append(f"{model.id}: license.url does not resolve: {model.license.url}")
            for benchmark in model.quality.benchmarks:
                if not _resolves(benchmark.source, client):
                    failures.append(
                        f"{model.id}/{benchmark.name}: source does not resolve: {benchmark.source}"
                    )

    assert not failures, "\n".join(failures)

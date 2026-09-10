"""The real Hugging Face, over the real network.

Marked ``network``: skipped in the ordinary suite and in CI, run with
``pytest -m network tests/unit/test_download_real_hub.py -v``.

It is here because the one thing no mock can check is the thing this project has already
been bitten by: Hugging Face answers a weights URL with a redirect to its content network,
and whether that redirect is followed — and whether the range survives it — is a property
of the real pair of servers. A fully green mocked suite proved nothing about it once
before. Two range requests and a few hundred bytes of traffic are what this costs.
"""

from __future__ import annotations

import pytest

from llamafit.catalog import load_catalog
from llamafit.download.engine import DownloadOptions, probe_size
from llamafit.download.plan import build_plan
from llamafit.download.transport import HttpRangeReader

pytestmark = pytest.mark.network

MODEL_ID = "qwen3-coder-next"


def test_the_real_hub_honours_a_range_through_its_redirect() -> None:
    catalog, problems = load_catalog()
    assert problems == []
    plan = build_plan(catalog.by_id[MODEL_ID])
    request = plan.files[0]

    with HttpRangeReader() as reader:
        size = probe_size(reader, request, options=DownloadOptions(max_attempts=2))
        assert size > 0
        with reader.open(request.url, 0, 255) as response:
            # 206, not 200: a range answered with the whole body is the failure this
            # package restarts a file over, and it must not be happening on the real hub.
            assert response.status == 206
            first = b"".join(response.body)

    assert len(first) == 256
    assert first[:4] == b"GGUF"


def test_the_size_the_hub_reports_is_the_size_the_catalog_records() -> None:
    # A single-file quant is the one case where the catalog's own total is that file's
    # size, so this is a real check that the catalog has been refreshed against what the
    # repository currently publishes.
    catalog, problems = load_catalog()
    assert problems == []
    plan = build_plan(catalog.by_id[MODEL_ID])
    request = plan.files[0]
    assert request.size is not None, "expected a single-file quant for this check"

    with HttpRangeReader() as reader:
        assert probe_size(reader, request, options=DownloadOptions(max_attempts=2)) == request.size

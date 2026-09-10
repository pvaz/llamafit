# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The ``--json`` documents the install commands print, and nothing else.

A script reading these is the other reader the ``install`` group has, and it wants the
opposite of what a person wants. Sizes stay integers of bytes rather than becoming
``4.1GiB``; a tag, a backend, a path and a repository stay the identifiers they are rather
than being wrapped in the bidirectional marks a terminal needs; and no sentence appears at
all, because a sentence is the thing that changes when somebody picks another language and
a document a pipeline parses must not.

That is why these live apart from :mod:`llamafit.cli.render_install` rather than beside the
tables they describe the same facts as. The two shapes have to be able to move
independently — a column can be added to a table without breaking a script, and a field can
be added to a document without a person having to read it — and one module holding both is
one edit away from a localised figure reaching a parser.

Every function is a plain mapping builder: it reads what a service already produced and
does not compute, format or decide anything.
"""

from __future__ import annotations

from typing import Any

from llamafit.download.install import InstallOutcome
from llamafit.download.plan import DiskCheck, DownloadPlan
from llamafit.llamacpp.install import InstallPlan, InstallResult, PathChange


def release_plan_payload(plan: InstallPlan) -> dict[str, Any]:
    """The llama.cpp plan as JSON: identifiers stay identifiers, sizes stay numbers."""
    return {
        "tag": plan.release.tag,
        "build": plan.release.build,
        "commit": plan.release.commit,
        "release_url": plan.release.url,
        "backend": plan.backend,
        "os": plan.os_name,
        "arch": plan.arch,
        "asset": plan.asset.name,
        "asset_bytes": plan.asset.size,
        "asset_sha256": plan.asset.sha256,
        "companions": [
            {"name": extra.name, "bytes": extra.size, "sha256": extra.sha256}
            for extra in plan.companions
        ],
        "verified": plan.verified,
        "target": str(plan.target.path),
        "bin_dir": str(plan.bin_dir),
        "ownership": plan.target.ownership,
        "replaces_tag": plan.replaces.tag if plan.replaces else None,
        "archive": str(plan.archive_path),
        "resume_bytes": plan.resume_bytes,
        "download_bytes": plan.download_bytes,
        "space": [
            {
                "path": str(check.path),
                "purpose": check.purpose,
                "needed_bytes": check.needed,
                "free_bytes": check.free,
                "enough": check.enough,
            }
            for check in plan.space
        ],
    }


def release_result_payload(result: InstallResult, change: PathChange | None) -> dict[str, Any]:
    """The finished llama.cpp install as JSON."""
    return {
        "installed": True,
        "root": str(result.root),
        "bin_dir": str(result.bin_dir),
        "tag": result.marker.tag,
        "build": result.marker.build,
        "commit": result.marker.commit,
        "backend": result.marker.backend,
        "backends": list(result.backends),
        "replaced_tag": result.replaced.tag if result.replaced else None,
        "warnings": list(result.warnings),
        "path_change": None
        if change is None
        else {"kind": change.kind, "where": change.where, "applied": change.needed},
    }


def download_plan_payload(plan: DownloadPlan, check: DiskCheck) -> dict[str, object]:
    """The ``--json`` document for ``--dry-run``."""
    return {
        "model": plan.model_id,
        "quant": plan.quant,
        "repo": plan.repo,
        "directory": str(plan.directory),
        "total_bytes": plan.total_bytes,
        "needed_bytes": check.needed,
        "free_bytes": check.free,
        "headroom_bytes": check.headroom,
        "enough_space": check.ok,
        "unverifiable": list(plan.unverifiable),
        "files": [
            {
                "name": file.name,
                "repo_path": file.repo_path,
                "url": file.url,
                "target": str(file.target),
                "bytes": file.expected_size,
                "sha256": file.sha256,
                "role": file.role,
                "already_present": file.already_present(),
            }
            for file in plan.files
        ],
    }


def download_outcome_payload(outcome: InstallOutcome) -> dict[str, object]:
    """The ``--json`` document for a finished install."""
    return {
        "model": outcome.plan.model_id,
        "quant": outcome.plan.quant,
        "repo": outcome.plan.repo,
        "directory": str(outcome.plan.directory),
        "manifest": str(outcome.manifest),
        "total_bytes": outcome.plan.total_bytes,
        "fetched_bytes": outcome.fetched_bytes,
        "resumed": outcome.resumed,
        "files": [
            {
                "name": file.name,
                "path": str(file.path),
                "bytes": file.size,
                "fetched_bytes": file.fetched,
                "already_present": file.already_present,
                "verified": file.verified,
                "sha256": file.sha256,
            }
            for file in outcome.files
        ],
    }


__all__ = [
    "download_outcome_payload",
    "download_plan_payload",
    "release_plan_payload",
    "release_result_payload",
]

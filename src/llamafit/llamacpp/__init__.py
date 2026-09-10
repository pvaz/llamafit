# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""llama.cpp integration: installation detection and running-server discovery."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from llamafit.hardware import current_os
from llamafit.hardware.runner import Runner, SubprocessRunner
from llamafit.llamacpp.detect import detect_install
from llamafit.llamacpp.server import HttpClient, HttpxClient, candidate_ports, discover_with_probes
from llamafit.logging import get_logger
from llamafit.models.host import OsName
from llamafit.models.llamacpp import LlamaCpp

_log = get_logger("llamacpp")


def detect_llamacpp(
    runner: Runner | None = None,
    *,
    os_name: OsName | None = None,
    env: Mapping[str, str] | None = None,
    http: HttpClient | None = None,
    well_known: Iterable[Path] | None = None,
) -> LlamaCpp:
    """Detect the installation and any running servers; never raises.

    Args:
        runner: Executes install-detection commands; a ``SubprocessRunner`` by default.
        os_name: Selects the executable suffix and the platform's well-known directories;
            defaults to the current OS.
        env: Environment mapping read for ``LLAMA_CPP_PATH``, ``PATH`` and
            ``LLAMA_SERVER_PORT``; defaults to ``os.environ``.
        http: HTTP client used for running-server discovery; a real ``HttpxClient`` by
            default.
        well_known: Directories to treat as the well-known llama.cpp install locations,
            passed straight through to ``detect_install``. ``None`` (the default) means
            the platform's well-known directories, matching a real scan; callers such as
            tests can pass a list (e.g. ``[]``) to override them so results do not depend
            on what happens to be installed on the machine running them.
    """
    runner = runner or SubprocessRunner()
    os_name = os_name or current_os()
    env = os.environ if env is None else env
    http = http or HttpxClient()
    llamacpp, probes = detect_install(runner, os_name, env=env, well_known=well_known)
    servers, server_probes = discover_with_probes(http, candidate_ports(env))
    llamacpp.running_servers = servers
    llamacpp.probes = [*probes, *server_probes]
    if llamacpp.installed:
        _log.debug(
            "llama.cpp found: build %s at %s, %d servers running",
            llamacpp.build,
            llamacpp.path,
            len(servers),
        )
    else:
        _log.debug("llama.cpp not found, %d servers running", len(servers))
    return llamacpp


__all__ = ["detect_llamacpp"]

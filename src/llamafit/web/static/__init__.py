# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The dashboard's files, read out of the installed package rather than off the source tree.

They are packaged data like the catalog and the translations, so they are reached through
:func:`llamafit.data.packaged_dir`: a wheel built without them installs perfectly and would
otherwise fail with a bare ``FileNotFoundError`` from inside Starlette. This subpackage
exists so ``importlib.resources`` has an anchor to resolve.
"""

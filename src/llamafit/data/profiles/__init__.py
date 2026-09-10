# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Bundled hardware profiles, read through ``importlib.resources``.

One file per machine, named for the profile inside it. Only machines this project has
actually measured ship here: a profile is data, and an invented memory bandwidth would be
data that reads exactly like a measured one to everything downstream. A user's own
profiles live in the data directory instead, which ``llamafit hardware path`` prints.
"""

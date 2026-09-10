# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The words on the Needs form: what a job is called, what a model can do, what to lean on.

A command line can put an identifier in front of a reader because they had to type one to
get there: ``--use-case coding`` is a value somebody chose. A form offers the choices
instead, and a list of enum values is not a set of choices anybody made -- it is the
catalog's vocabulary showing through. So each one is written out here as its own literal
call, which is also the only shape the extractor can find.

The capability words carry the ``model capability`` context
:func:`llamafit.cli.render.render_catalog_list` already uses, so a translator meets each of
those eight exactly once for the whole program. The use cases and the preferences are new,
because the command line never had to name them as anything but flag values.

A value the catalog grows before this table does comes back as its identifier: ugly, and
truthful, and visible enough that somebody fixes it.
"""

from __future__ import annotations

from llamafit.i18n import pgettext


def use_case_label(use_case: str) -> str:
    """What a model is wanted for, as words rather than as the value ``--use-case`` takes."""
    labels = {
        "general": pgettext("use case", "general use"),
        "coding": pgettext("use case", "writing code"),
        "reasoning": pgettext("use case", "reasoning"),
        "chat": pgettext("use case", "chat"),
        "multimodal": pgettext("use case", "images and text"),
        "embedding": pgettext("use case", "embeddings"),
    }
    return labels.get(use_case, use_case)


def capability_label(capability: str) -> str:
    """One of a model's capabilities, in the reader's language."""
    labels = {
        "coding": pgettext("model capability", "coding"),
        "thinking": pgettext("model capability", "thinking"),
        "vision": pgettext("model capability", "vision"),
        "tools": pgettext("model capability", "tools"),
        "multilingual": pgettext("model capability", "multilingual"),
        "long-context": pgettext("model capability", "long-context"),
        "embeddings": pgettext("model capability", "embeddings"),
        "audio": pgettext("model capability", "audio"),
    }
    return labels.get(capability, capability)


def prefer_label(prefer: str) -> str:
    """Which way ``--prefer`` leans the weights, said as a lean rather than as a value."""
    labels = {
        "balanced": pgettext("preference", "balanced"),
        "quality": pgettext("preference", "lean towards quality"),
        "speed": pgettext("preference", "lean towards speed"),
    }
    return labels.get(prefer, prefer)

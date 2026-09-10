# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The five screens of section 13.2, and the key map behind ``?``.

Each is a container widget rather than a Textual ``Screen``, because they sit inside one
tab bar and a reader moves between them with Tab rather than by pushing and popping. The
one real ``Screen`` here is the key map, which is modal because it is the only thing in the
dashboard that covers what a reader was looking at.

They are not re-exported. Importing this package pulls in Textual, and
:mod:`llamafit.tui` deliberately does not.
"""

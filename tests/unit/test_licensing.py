"""The licence is stated in three places; none of them may drift from the others.

``pyproject.toml`` carries it twice, as a PEP 639 expression and as a trove classifier, and
every module under ``src/llamafit`` carries the short notice the AGPL asks each file to have.
A file that loses its notice is a file somebody will later copy out of the project and believe
is unencumbered, which is the whole reason the notice is there.
"""

from importlib.metadata import metadata
from pathlib import Path

SPDX = "AGPL-3.0-or-later"
CLASSIFIER = "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)"
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "llamafit"


def test_the_distribution_metadata_declares_the_agpl() -> None:
    meta = metadata("llamafit")
    # hatchling >= 1.27 writes License-Expression; older backends fall back to License.
    assert (meta.get("License-Expression") or meta.get("License")) == SPDX
    assert CLASSIFIER in (meta.get_all("Classifier") or [])


def test_every_source_module_carries_the_notice() -> None:
    modules = sorted(SOURCE_ROOT.rglob("*.py"))
    assert modules, f"no modules found under {SOURCE_ROOT}"
    missing = [
        str(path.relative_to(SOURCE_ROOT))
        for path in modules
        if f"# SPDX-License-Identifier: {SPDX}" not in path.read_text(encoding="utf-8")
    ]
    assert missing == []

"""Write the published JSON schemas from the pydantic models.

Usage:
    python scripts/gen_schema.py

Two documents, one per kind of file a person writes by hand: a catalog entry and a
hardware profile. Neither is what LlamaFit validates against -- the pydantic models are --
so both are published for other people's tooling: an editor that offers completion, a CI
job that checks a file before it is opened as a pull request. See ``docs/catalog.md`` and
``docs/hardware-profiles.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from llamafit.models.catalog import CatalogModel
from llamafit.models.hwprofile import HardwareProfile

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/llamafit/data/schema"

SCHEMAS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("catalog.schema.json", CatalogModel),
    ("hwprofile.schema.json", HardwareProfile),
)


def main() -> int:
    """Regenerate every schema file and report where each went."""
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS:
        destination = SCHEMA_DIR / name
        schema = model.model_json_schema(by_alias=True)
        # newline="\n" so the file is byte-identical on every platform. Without it Python
        # writes the host's line ending, and the staleness check in CI compares a file a
        # Windows contributor regenerated against one Linux wrote: same content, every line
        # different, and a failure that says nothing about what is actually stale.
        destination.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Write the catalog JSON schema from the pydantic models.

Usage:
    python scripts/gen_schema.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llamafit.models.catalog import CatalogModel

DEST = Path(__file__).resolve().parents[1] / "src/llamafit/data/schema/catalog.schema.json"


def main() -> int:
    """Regenerate the schema file and report where it went."""
    DEST.parent.mkdir(parents=True, exist_ok=True)
    schema = CatalogModel.model_json_schema(by_alias=True)
    # newline="\n" so the file is byte-identical on every platform. Without it Python
    # writes the host's line ending, and the staleness check in CI compares a file a
    # Windows contributor regenerated against one Linux wrote: same content, every line
    # different, and a failure that says nothing about what is actually stale.
    DEST.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

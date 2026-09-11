"""The catalog has to read as one list, not as five curators' lists side by side.

Everything here was a real divergence found by reading the catalog top to bottom: a file
that changed units halfway down, a `family` value that meant a generation in one file and a
vendor in another, a capability one entry dropped while its four siblings kept it, a date
whose meaning nobody wrote down. None of it broke a command -- `catalog validate` passed
throughout -- which is exactly why it needs tests of its own. The conventions these enforce
are written down in `docs/catalog.md`; the failure messages point back at it.
"""

import re
from pathlib import Path

import yaml

from llamafit.catalog.loader import bundled_catalog_dir, load_catalog

# MT-Bench and its multimodal sibling are a judge's rating from 1 to 10, not a percentage.
# Every other benchmark in the catalog is a percentage on 0 to 100.
_RATING_BENCHMARKS = ("MTBench", "MT-Bench")

# phi-4 and phi-4-mini-instruct list `general` without `chat`, and phi4.yaml's header says
# why at length: Microsoft's cards name reasoning and logic among the primary use cases and
# name no conversational one. A deliberate omission with its reason in the file is the thing
# this test is trying to protect, not the thing it is trying to catch.
_GENERAL_WITHOUT_CHAT = frozenset({"phi-4", "phi-4-mini-instruct"})

# The Phi line trades world knowledge for reasoning and only the mini model is trained for
# more than English; phi4.yaml says so. Every other family agrees with itself.
_MULTILINGUAL_EXCEPTIONS = frozenset(
    {"phi-4", "phi-4-reasoning-plus", "phi-4-reasoning-vision-15b"}
)


def _catalog_files() -> list[Path]:
    """Every curated family file in the bundled catalog."""
    files = sorted(bundled_catalog_dir().glob("*.yaml"))
    assert files, "the bundled catalog has no family files"
    return files


def _stem(value: str) -> str:
    """A family or id reduced to letters and digits, so `gemma3` matches `gemma-3-27b-it`."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def test_no_benchmark_score_is_recorded_as_a_fraction() -> None:
    """Scores are on the scale they were published on, and that scale is 0 to 100.

    Mistral publishes Arena Hard, MATH, AIME and GPQA as fractions; five entries recorded
    them that way while the sixth entry in the same file, and every other family, used
    percentages. A reader scanning one file cannot be asked to notice that `0.860` and
    `86.0` are the same score.
    """
    catalog, _ = load_catalog(custom_path=None)
    for model in catalog.models:
        for benchmark in model.quality.benchmarks:
            assert benchmark.score > 1.0, (
                f"{model.id}/{benchmark.name} scores {benchmark.score}, which is a fraction; "
                "docs/catalog.md puts benchmark scores on 0 to 100"
            )


def test_the_only_scores_below_ten_are_the_ones_measured_on_a_ten_point_scale() -> None:
    """A percentage under 10 is nearly always a fraction somebody forgot to convert.

    The catalog holds exactly one honest example -- phi-4's SimpleQA 3.0, which the Phi-4
    card publishes as a percentage and phi4.yaml explains -- and the rating benchmarks,
    which are not percentages at all. Anything else below 10 is worth a second look.
    """
    catalog, _ = load_catalog(custom_path=None)
    suspicious = [
        f"{model.id}/{benchmark.name}={benchmark.score}"
        for model in catalog.models
        for benchmark in model.quality.benchmarks
        if benchmark.score < 10.0
        and not benchmark.name.startswith("SimpleQA")
        and not any(rating in benchmark.name for rating in _RATING_BENCHMARKS)
    ]
    assert suspicious == [], f"{suspicious} look like unconverted fractions"


def test_every_id_begins_with_its_family() -> None:
    """`family` is the generation, and an entry's id starts with the generation it is in.

    This is what caught `family: qwen3` on a Qwen3.8 model and on Qwen3-Coder-Next, and
    `family: deepseek` on a DeepSeek-R1 distillation. See docs/catalog.md, "family".
    """
    catalog, _ = load_catalog(custom_path=None)
    for model in catalog.models:
        assert _stem(model.id).startswith(_stem(model.family)), (
            f"{model.id} is filed under family {model.family!r}, which is not its own "
            "generation; docs/catalog.md says which of the two to change"
        )


def test_a_family_never_spans_two_vendors() -> None:
    """A generation belongs to the vendor that trained it, whoever's architecture it uses."""
    catalog, _ = load_catalog(custom_path=None)
    vendors: dict[str, set[str]] = {}
    for model in catalog.models:
        vendors.setdefault(model.family, set()).add(model.vendor)
    spread = {family: sorted(names) for family, names in vendors.items() if len(names) > 1}
    assert spread == {}, f"{spread} put one family under two vendors"


def test_a_model_good_for_general_use_is_also_good_for_a_chat() -> None:
    """Nothing that answers a general request should hide from an ordinary one.

    `qwen3-0.6b` listed `[general, coding]` and so never appeared for
    `llamafit list --use-case chat`, repeating a mistake the catalog's own comments say was
    made and fixed on another entry. Leaving `chat` off is allowed; leaving it off silently
    is not, which is why the exceptions are named here and explained in their file.
    """
    catalog, _ = load_catalog(custom_path=None)
    missing = [
        model.id
        for model in catalog.models
        if "general" in model.use_cases
        and "chat" not in model.use_cases
        and model.id not in _GENERAL_WITHOUT_CHAT
    ]
    assert missing == [], f"{missing} answer a general request but not a chat one"


def test_multilingual_is_not_dropped_from_one_sibling_of_a_family() -> None:
    """One entry of five lacked `multilingual` with nothing in the file saying why.

    Capabilities like `coding` and `vision` really do come and go with size inside a family,
    so they are not checked here. Languages are a property of the training data, which a
    vendor varies across a generation far more rarely -- and never without saying so.
    """
    catalog, _ = load_catalog(custom_path=None)
    families: dict[str, list[tuple[str, bool]]] = {}
    for model in catalog.models:
        families.setdefault(model.family, []).append(
            (model.id, "multilingual" in model.capabilities)
        )
    for family, members in sorted(families.items()):
        if not any(multilingual for _, multilingual in members):
            continue
        missing = [
            model_id
            for model_id, multilingual in members
            if not multilingual and model_id not in _MULTILINGUAL_EXCEPTIONS
        ]
        assert missing == [], (
            f"{missing} lack multilingual while their {family} siblings claim it; add it, or "
            "add the comment that says why not"
        )


def test_every_family_file_says_where_its_release_dates_come_from() -> None:
    """Three kinds of date were in use and thirteen files did not say which they used.

    An announcement date, a model-card date and a Hugging Face repository's creation date
    are days to months apart and look identical in the field. docs/catalog.md fixes the
    meaning; each file has to say which evidence it had.
    """
    for path in _catalog_files():
        header = [
            line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")
        ]
        assert any(re.search(r"release[ _]date", line, re.IGNORECASE) for line in header), (
            f"{path.name} does not say where its release dates come from"
        )


def test_a_hybrid_with_no_kv_types_says_that_nobody_measured_them() -> None:
    """An empty `kv_types_allowed` means unmeasured, and the file has to say so.

    Nothing distinguishes "nobody measured this" from "the curator did not think to ask"
    except a comment. The Qwen files wrote one; five other hybrid families had not.
    """
    for path in _catalog_files():
        text = path.read_text(encoding="utf-8")
        entries = yaml.safe_load(text)
        unexplained = [
            entry["id"]
            for entry in entries
            if entry["architecture"]["class"].endswith("hybrid")
            and not (entry.get("llama_cpp") or {}).get("kv_types_allowed")
        ]
        if unexplained:
            assert "kv_types_allowed" in text, (
                f"{path.name} leaves kv_types_allowed empty on {unexplained} without saying "
                "that it is unmeasured"
            )

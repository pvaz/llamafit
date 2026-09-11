"""The sentences ``--explain`` ends with, held one shape at a time.

``test_cli_board.py`` proves the counterfactuals reach the screen on a real board; this
file pins the shapes that board cannot reach on one machine — the alternative that fits
entirely on the card, the one that fits no better, the one nobody could size, and the
machine roomy enough that nothing about the context is worth saying.

The rule every assertion here serves is the one the feature turns on. A sentence saying a
smaller quantisation would fit when it would not is worse than no sentence, so the shape
with nothing behind it must render nothing at all, and the shapes with something behind
them must say which of the four things changed rather than all of them at once.
"""

from __future__ import annotations

from rich.console import Console

from llamafit.cli.render import render_counterfactuals, render_quant_budgets
from llamafit.i18n.translator import set_language
from llamafit.models.plan import Budget, BudgetLine, ContextTier, Placement
from llamafit.services.catalog import QuantDetail
from llamafit.services.plan import ContextReach, Counterfactuals, QuantSwap

GIB = 1024**3


def drawn(renderable: object) -> str:
    """Whatever a renderer produced, as one line of plain text."""
    console = Console(width=200, no_color=True, record=True)
    console.print(renderable)
    return " ".join(console.export_text().split())


def english() -> None:
    """Assertions here are on English text, so the process has to be speaking it."""
    set_language("en", env={}, directory=None)


def budget(*, vram_required: int, vram_available: int, verdict: str = "tight") -> Budget:
    """A budget with only the fields these sentences read."""
    return Budget(
        lines=(
            BudgetLine(component="dense-weights", pool="vram", bytes=vram_required, exact=True),
        ),
        vram_required=vram_required,
        ram_required=GIB,
        vram_available=vram_available,
        ram_available=32 * GIB,
        vram_utilisation=vram_required / vram_available,
        ram_utilisation=0.1,
        verdict=verdict,  # type: ignore[arg-type]
    )


def placement(
    *, mode: str = "moe-offload", verdict: str = "tight", context: int = 32768
) -> Placement:
    """A placement with only the fields these sentences read."""
    return Placement(
        mode=mode,  # type: ignore[arg-type]
        context=context,
        micro_batch=2048,
        batch=4096,
        kv_type="f16",
        gpu_layers=99,
        threads=16,
        budget=budget(vram_required=6 * GIB, vram_available=7 * GIB, verdict=verdict),
        max_context_fit=40960,
        tiers=(ContextTier(tokens=context, vram_required=6 * GIB, fits=True, verdict="tight"),),
    )


def test_a_rung_nothing_could_absorb_is_not_sold_as_a_card_problem() -> None:
    english()
    found = Counterfactuals(
        context=ContextReach(
            tokens=65536,
            vram_required=9 * GIB,
            vram_available=7 * GIB,
            vram_to_free=0,
            verdict="does-not-fit",
        )
    )
    text = drawn(render_counterfactuals(found, placement()))
    assert "64K tokens is past what this machine holds" in text
    assert "free" in text  # the sentence says freeing the card is not the change
    assert "0 B" not in text, "a figure of nothing must never be offered as an amount to free"


def test_a_machine_that_holds_every_rung_says_so() -> None:
    english()
    text = drawn(render_counterfactuals(Counterfactuals(every_context_fits=True), placement()))
    assert "Every context this model offers already fits here" in text


def test_an_alternative_that_reaches_the_card_says_that_first() -> None:
    """Section 12.3's own example: "the Q3 quant would fit entirely in VRAM"."""
    english()
    swap = QuantSwap(
        quant="Q3_K_M",
        mode="gpu",
        verdict="fits",
        max_context_fit=40960,
        gen_tps=18.0,
        entirely_on_card=True,
        better_verdict=True,
    )
    text = drawn(render_counterfactuals(Counterfactuals(quant=swap), placement()))
    assert "Q3_K_M, the next quantisation down, was planned here too" in text
    assert "18.0 tokens per second" in text
    assert "It fits entirely on the graphics card" in text
    assert "It fits better" not in text, "one sentence, the one a reader would act on first"


def test_an_alternative_that_only_fits_better_says_that() -> None:
    english()
    swap = QuantSwap(
        quant="Q3_K_M",
        mode="moe-offload",
        verdict="fits",
        max_context_fit=40960,
        better_verdict=True,
    )
    text = drawn(render_counterfactuals(Counterfactuals(quant=swap), placement(verdict="tight")))
    assert "It fits better: fits rather than tight." in text
    assert "tokens per second" not in text, "no speed was estimated, so none is claimed"


def test_an_alternative_that_changes_nothing_says_it_changes_nothing() -> None:
    """Stopping a reader chasing a download is as useful as sending them after one."""
    english()
    swap = QuantSwap(quant="Q3_K_M", mode="moe-offload", verdict="tight", max_context_fit=32768)
    text = drawn(render_counterfactuals(Counterfactuals(quant=swap), placement()))
    assert "It is no better here" in text


def test_an_alternative_that_fits_nowhere_is_not_offered_as_a_way_out() -> None:
    english()
    swap = QuantSwap(quant="Q3_K_M", mode="unsupported", verdict="does-not-fit")
    text = drawn(render_counterfactuals(Counterfactuals(quant=swap), placement()))
    assert "does not fit this machine either" in text
    assert "It is no better" not in text


def test_an_alternative_nobody_could_size_carries_the_reason() -> None:
    english()
    swap = QuantSwap(quant="Q3_K_M", unplaceable_because="nobody has read this file")
    text = drawn(render_counterfactuals(Counterfactuals(quant=swap), placement()))
    assert "could not be sized: nobody has read this file" in text


def test_nothing_to_say_is_said_by_printing_nothing() -> None:
    assert render_counterfactuals(Counterfactuals(), placement()) is None


def test_a_model_with_no_quant_anybody_could_size_still_answers() -> None:
    english()
    detail = QuantDetail(
        name="Q4_K_M",
        bytes=None,
        bpw=None,
        facts=None,
        files=[],
        unplaceable_because="nobody has read this file",
    )
    text = drawn(render_quant_budgets([detail], context=32768))
    assert "Q4_K_M: nobody has read this file" in text
    assert "Sized for" not in text, "there is no table to caption"


def test_no_quants_at_all_draws_no_table() -> None:
    assert render_quant_budgets([], context=32768) is None

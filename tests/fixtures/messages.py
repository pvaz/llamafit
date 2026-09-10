"""The worked set of messages the translation machinery is exercised against.

Every message here is also wrapped at a real call site in `llamafit.services.doctor`,
`llamafit.cli.render`, `llamafit.cli.app`, `llamafit.cli.system_cmd` and
`llamafit.catalog.loader`, which is what makes it safe for the translation tests to
assert on them word for word: a test that froze a real call site would turn improving
one of those sentences into a broken build.

`scripts.gen_messages.SOURCE_ROOTS` no longer names this file, so nothing here can add a
message to the template on its own. Adding one that no call site in `src/` makes would
show up at once, as a catalog entry the template does not have.
"""

from llamafit.i18n import (
    LazyString,
    _,
    lazy_gettext,
    lazy_ngettext,
    lazy_pgettext,
    ngettext,
    pgettext,
)

# Built while this module is imported, which is before any test has chosen a language:
# exactly the shape of llamafit.services.doctor.PROBE_HINTS and of a Typer help= string.
# Wrapped with the eager _() these would be English for the life of the process, silently.
PROBE_HINTS: dict[str, LazyString] = {
    "nvidia-smi": lazy_gettext("If a GPU is present, check that its driver tools are installed."),
    "memory-modules": lazy_gettext("the vendor tool that reports memory was not available"),
}
GPU_SUMMARY: LazyString = lazy_gettext("No GPU detected")
MODULE_COUNT: LazyString = lazy_ngettext("%(count)d module", "%(count)d modules", 3)
GPU_ROW_EMPTY: LazyString = lazy_pgettext("GPU", "none detected")


def llamacpp_installed() -> str:
    return _("llama.cpp installed")


def llamacpp_missing() -> str:
    return _("llama.cpp not found")


def no_gpu_detected() -> str:
    return _("No GPU detected")


def cpu_only() -> str:
    return _("models will run on the CPU only")


def check_driver_tools() -> str:
    return _("If a GPU is present, check that its driver tools are installed.")


def bandwidth_assumed() -> str:
    return _("RAM bandwidth assumed")


def install_numpy() -> str:
    return _(
        "Install numpy (`pip install llamafit[fast]`) so LlamaFit can measure the RAM bandwidth."
    )


def vram_tool_missing() -> str:
    return _("the vendor tool that reports memory was not available")


def free_disk_space() -> str:
    return _("Most useful models need 5 to 120 GB; free space or change the downloads directory.")


# "unknown" and "none detected" are each the value of two different rows, and Portuguese
# does not agree with itself about them: the memory bandwidth is feminine and a quant's
# bits per weight masculine, the GPU is feminine and a backend masculine. One msgid could
# only ever be right in one of each pair, so each row names the context it reads in.
def bandwidth_unknown() -> str:
    return pgettext("memory bandwidth", "unknown")


def bits_per_weight_unknown() -> str:
    return pgettext("bits per weight", "unknown")


def no_gpu_row() -> str:
    return pgettext("GPU", "none detected")


def no_backends_row() -> str:
    return pgettext("backends", "none detected")


def no_server() -> str:
    return pgettext("probe result", "no server")


def run_verbose() -> str:
    return _("Run again with --verbose for the full traceback.")


def skip_measurement() -> str:
    # Deliberately left untranslated in pt_PT.po, to prove an empty msgstr falls back.
    return _("Skip the RAM bandwidth measurement.")


def memory_modules(count: int) -> str:
    return ngettext("%(count)d module", "%(count)d modules", count) % {"count": count}


def catalog_problems(count: int) -> str:
    return ngettext(
        "the catalog has %(count)d problem",
        "the catalog has %(count)d problems",
        count,
    ) % {"count": count}


def a_message_split_over_lines() -> str:
    return _(
        "LlamaFit could not read this machine's memory totals, so it cannot work out what will fit."
    )

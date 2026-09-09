"""The worked set of messages the translation machinery is exercised against.

These are real messages, copied from `llamafit.services.doctor`, `llamafit.cli.render`,
`llamafit.cli.app` and `llamafit.catalog.loader`, wrapped here rather than at their call
sites: four branches are editing those files at the moment, and the sweep that wraps them
is a separate piece of work. Wrapping them here is what lets the extractor, the template
and the Portuguese catalog be tested end to end before that sweep lands.

When the sweep does land, these functions go away and `scripts.gen_messages.SOURCE_ROOTS`
stops naming this file.
"""

from llamafit.i18n import _, ngettext


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
    return _("Install numpy (`pip install llamafit[fast]`) so LlamaFit can measure it.")


def vram_tool_missing() -> str:
    return _("the vendor tool that reports memory was not available")


def free_disk_space() -> str:
    return _("Most useful models need 5 to 120 GB; free space or change the downloads directory.")


def unknown() -> str:
    return _("unknown")


def none_detected() -> str:
    return _("none detected")


def no_server() -> str:
    return _("no server")


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

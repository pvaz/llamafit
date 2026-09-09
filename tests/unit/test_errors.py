from llamafit.errors import LlamaFitError, NotInstalledError, ProbeError


def test_render_includes_hint_and_command() -> None:
    err = ProbeError("nvidia-smi failed", hint="Install the NVIDIA driver", command="nvidia-smi -L")
    text = err.render()
    assert "nvidia-smi failed" in text
    assert "Hint: Install the NVIDIA driver" in text
    assert "Command: nvidia-smi -L" in text


def test_render_without_extras_is_just_the_message() -> None:
    assert LlamaFitError("boom").render() == "boom"


def test_subclasses_are_llamafit_errors() -> None:
    assert issubclass(NotInstalledError, LlamaFitError)

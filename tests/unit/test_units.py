import pytest

from llamafit.units import format_bytes, gib, parse_size


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8G", 8_000_000_000),
        ("8GB", 8_000_000_000),
        ("8GiB", 8 * 1024**3),
        ("7.5GiB", int(7.5 * 1024**3)),
        ("512M", 512_000_000),
        ("512MiB", 512 * 1024**2),
        ("1T", 1_000_000_000_000),
        ("1024", 1024),
        (" 16 gb ", 16_000_000_000),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "abc", "8X", "-1G", "1.2.3G"])
def test_parse_size_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_size(bad)


def test_format_bytes_binary_and_decimal() -> None:
    assert format_bytes(8 * 1024**3) == "8.0 GiB"
    assert format_bytes(8 * 1024**3, binary=False) == "8.6 GB"
    assert format_bytes(1536 * 1024**2) == "1.5 GiB"
    assert format_bytes(900 * 1024) == "900.0 KiB"
    assert format_bytes(12) == "12 B"
    assert format_bytes(None) == "unknown"


def test_gib() -> None:
    assert gib(1024**3) == 1.0
    assert gib(None) is None

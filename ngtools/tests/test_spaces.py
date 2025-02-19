"""Tests for the `spaces` module."""
import neuroglancer as ng

from ngtools import spaces as S


def test_get_neuronames() -> None:
    """List of all possible neurological spaces."""
    ref1 = {"r", "l", "a", "p", "i", "s"}
    assert set(S._get_neuronames(1)) == ref1

    ref2 = {
        "ra", "rp", "la", "lp", "ar", "al", "pr", "pl",
        "ri", "rs", "li", "ls", "ir", "il", "sr", "sl",
        "ai", "as", "pi", "ps", "ia", "ip", "sa", "sp",
    }
    assert set(S._get_neuronames(2)) == ref2

    ref3 = {
        "rai", "ras", "rpi", "rps", "lai", "las", "lpi", "lps",
        "ria", "rip", "rsa", "rsp", "lia", "lip", "lsa", "lsp",
        "ari", "ars", "ali", "als", "pri", "prs", "pli", "pls",
        "air", "ail", "asr", "asl", "pir", "pil", "psr", "psl",
        "ira", "irp", "ila", "ilp", "sra", "srp", "sla", "slp",
        "iar", "ial", "ipr", "ipl", "sar", "sal", "spr", "spl",
    }
    assert set(S._get_neuronames(3)) == ref3


def test_get_defaultnames() -> None:
    """List of all possible default spaces."""
    ref1 = {"x", "y", "z"}
    assert set(S._get_defaultnames(1)) == ref1

    ref2 = {"xy", "yx", "xz", "zx", "yz", "zy"}
    assert set(S._get_defaultnames(2)) == ref2

    ref3 = {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}
    assert set(S._get_defaultnames(3)) == ref3


def test_space_is_compatible() -> None:
    """Compatibility between different oriented spaces."""
    assert S.space_is_compatible("xyz", "ras")
    assert S.space_is_compatible("xyz", "sar")
    assert S.space_is_compatible("xyz", "lpi")
    assert S.space_is_compatible("ras", "lpi")

    assert S.space_is_compatible("xy", "ra")
    assert S.space_is_compatible("xy", "ar")
    assert S.space_is_compatible("yz", "pi")
    assert S.space_is_compatible("as", "pi")

    assert not S.space_is_compatible("xyz", "rac")
    assert not S.space_is_compatible("xyz", "ra")
    assert not S.space_is_compatible("ra", "pi")


def test_convert_space() -> None:
    """Test simple conversion."""
    # NOTE: we use `5 * 1E-4` and not `5E-4` because the conversion
    # code uses the former and both values differ in double precision
    # arithmetic.

    inp = ng.CoordinateSpace({
        "x": [5, "um"],
        "y": [5, "mm"],
        "z": [5, "cm"],
        "t": [4, "ms"],
        "c": [1, ""],
    })

    ref = ng.CoordinateSpace({
        "x": [5 * 1E-4, "cm"],
        "y": [5 * 1E-3, "m"],
        "z": [5 * 1E-5, "km"],
        "t": [4 * 1E-5, "hs"],
        "c": [1, ""],
    })

    out = S.convert_space(inp, {"x": "cm", "y": "m", "z": "km", "t": "hs"})
    assert out.to_json() == ref.to_json(), \
           f"Convert to arbitrary units:\nout={out}\nref={ref}"

    ref = ng.CoordinateSpace({
        "x": [5 * 1E-6, "m"],
        "y": [5 * 1E-3, "m"],
        "z": [5 * 1E-2, "m"],
        "t": [4 * 1E-3, "s"],
        "c": [1, ""],
    })

    out = S.normalize_space(inp)
    assert out.to_json() == ref.to_json(), \
           f"Convert to SI:\nout={out}\nref={ref}"

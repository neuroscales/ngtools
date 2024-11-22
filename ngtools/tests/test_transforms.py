"""Tests for the `transforms` module."""
import neuroglancer as ng

from ngtools import transforms as T


def test_convert_transform() -> None:
    """Test transform conversion."""
    # NOTE: we use `5 * 1E-4` and not `5E-4` because the conversion
    # code uses the former and both values differ in double precision
    # arithmetic.

    inp = ng.CoordinateSpaceTransform(
        input_dimensions=ng.CoordinateSpace({
            "i": [5, "um"],
            "j": [5, "mm"],
            "k": [5, "cm"],
            "t": [4, "ms"],
            "c": [1, ""],
        }),
        output_dimensions=ng.CoordinateSpace({
            "x": [1, "um"],
            "y": [2, "mm"],
            "z": [4, "cm"],
            "u": [6, "ms"],
            "c": [1, ""],
        }),
        matrix=[
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1],
        ]
    )

    ref = ng.CoordinateSpaceTransform(
        input_dimensions=ng.CoordinateSpace({
            "i": [5 * 10**-3,   "mm"],
            "j": [5,            "mm"],
            "k": [5 * 10,       "mm"],
            "t": [4 * 10**-3,   "s"],
            "c": [1,            ""],
        }),
        output_dimensions=ng.CoordinateSpace({
            "x": [1, "um"],
            "y": [2, "mm"],
            "z": [4, "cm"],
            "u": [6, "ms"],
            "c": [1, ""],
        }),
        matrix=[
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1],
        ]
    )
    out = T.convert_transform(inp, ["mm", "s"], None)
    assert out == ref, f"Convert to SI units:\nout={out}\nref={ref}"

    ref = ng.CoordinateSpaceTransform(
        input_dimensions=ng.CoordinateSpace({
            "i": [1, "m"],
            "j": [1, "m"],
            "k": [1, "m"],
            "t": [1, "s"],
            "c": [1, ""],
        }),
        output_dimensions=ng.CoordinateSpace({
            "x": [1, "m"],
            "y": [1, "m"],
            "z": [1, "m"],
            "u": [1, "s"],
            "c": [1, ""],
        }),
        matrix=[
            [5 * 1E-6, 0,        0,        0,        0, 1 * 1E-6],
            [0,        5 * 1E-3, 0,        0,        0, 2 * 1E-3],
            [0,        0,        5 * 1E-2, 0,        0, 4 * 1E-2],
            [0,        0,        0,        4 * 1E-3, 0, 6 * 1E-3],
            [0,        0,        0,        0,        1, 1],
        ]
    )
    out = T.normalize_transform(inp, unit_scale=True)
    assert out == ref, \
           f"Normalize transform (unit scale):\nout={out}\nref={ref}"


def test_inverse_compose() -> None:
    """Invert a transform and compose to recover identity."""
    inp = ng.CoordinateSpaceTransform(
        input_dimensions=ng.CoordinateSpace({
            "i": [5, "um"],
            "j": [5, "mm"],
            "k": [5, "cm"],
            "t": [4, "ms"],
            "c": [1, ""],
        }),
        output_dimensions=ng.CoordinateSpace({
            "x": [1, "um"],
            "y": [2, "mm"],
            "z": [4, "cm"],
            "u": [6, "ms"],
            "c": [1, ""],
        }),
        matrix=[
            [+0, -1, +0, +0, +0, +1],
            [+0, +0, +1, +0, +0, +1],
            [+1, +0, +0, +0, +0, +1],
            [+0, +0, +0, +1, +0, +1],
            [+0, +0, +0, +0, +1, +1],
        ]
    )

    inv = T.inverse(inp)

    ref_left = ng.CoordinateSpaceTransform(
        input_dimensions=inp.input_dimensions,
        output_dimensions=inp.input_dimensions,
        matrix=[
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    out_left = T.compose(inv, inp)
    assert out_left == ref_left, \
           "Left-compose inverse:\nout={out_left}\nref={ref_left}"

    ref_right = ng.CoordinateSpaceTransform(
        input_dimensions=inp.output_dimensions,
        output_dimensions=inp.output_dimensions,
        matrix=[
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    out_right = T.compose(inp, inv)
    assert out_right == ref_right, \
           "Right-compose inverse:\nout={out_right}\nref={ref_right}"

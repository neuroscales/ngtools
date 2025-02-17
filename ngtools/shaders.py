"""
Functions to generate NeuroGlancer shaders.

NeuroGlancer shaders must be snippets of GLSL code. This module contains
functions that dynamically generate such code snippets.

GLSL = OpenGL Shading Language.
"""
# stdlib
import math
from os import PathLike
from textwrap import dedent
from typing import IO

# internals
from ngtools import cmdata
from ngtools.local.termcolors import bformat
from ngtools.opener import open


def _flatten(x: list[list]) -> tuple:
    """Transform a list of list into a flat tuple."""
    return tuple(z for y in x for z in y)


def pretty_colormap_list(linewidth: int = 79, colorbar: bool = True) -> str:
    """List all existing colormaps."""
    names = list(filter(lambda x: not x[0] == '_', dir(cmdata)))
    names = ['greyscale', 'orientation'] + names
    names = list(sorted(names))
    longest_name = max(map(len, names))

    if colorbar:
        colors = []
        for name in names:
            if name == "greyscale":
                name = "blackwhite"
            if hasattr(cmdata, name):
                if not isinstance(getattr(cmdata, name), (list, tuple, dict)):
                    colorstr = " " * longest_name
                else:
                    colormap = pycolormaps.make_colormap(name)
                    colorstr = ""
                    for i in range(longest_name):
                        i = i / (longest_name - 1)
                        color = [255 * c for c in colormap(i)]
                        colorstr += bformat.bg.rgb256(*color)(" ")
            else:
                colorstr = " " * longest_name
            colors += [colorstr]

    nbcol = max(1, linewidth // (longest_name + 2))
    nbrow = int(math.ceil(len(names) / nbcol))
    cell = " {:<" + str(longest_name) + "s}"
    row = cell * nbcol
    lastrow = cell * (len(names) % nbcol)
    rep = 2 if colorbar else 1
    if lastrow:
        rows = "\n".join([row] * (nbrow-1) * rep + [lastrow] * rep)
    else:
        rows = "\n".join([row] * nbrow * rep)

    if colorbar:
        cells = []
        for i in range(nbrow):
            cells += names[nbcol*i:nbcol*(i+1)]
            cells += colors[nbcol*i:nbcol*(i+1)]
    else:
        cells = names
    return rows.format(*cells)


def load_fs_lut(
    path: str | PathLike | IO
) -> dict[int, tuple[str, list[float]]]:
    """
    Load a FreeSurfer lookup table.

    Returns a dictionary that maps integer labels to tuples that contains
    1. the structure name, as a string
    2. the structure RGBA color, as a 4-tuple of values between 0 and 1.
    """
    if not hasattr(path, 'readlines'):
        with open(path, 'rt') as f:
            return load_fs_lut(f)
    lut = {}
    for line in path.readlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        label, name, r, g, b, a, *_ = line.split()
        label, r, g, b, a = int(label), int(r), int(g), int(b), int(a)
        lut[label] = (name, (r/255, g/255, b/255, (a or 255)/255))
    return lut


class pycolormaps:
    """
    Namespace for dynamic colormaps.

    Functions in this class return python functions.
    """

    _DEFAULT_LENGTH = object()

    @staticmethod
    def make_colormap(
        name: str,
        data: object | None = None,
    ) -> str:
        """Generate GLSL code for a colormap."""
        data = data or getattr(cmdata, name)
        if isinstance(data, list):
            return pycolormaps.make_listed(name, data)
        elif isinstance(data, tuple):
            return pycolormaps.make_segmented(name, data)
        elif isinstance(data, dict):
            return pycolormaps.make_weighted(name, data)
        return None

    @staticmethod
    def make_listed(
        name: str,
        data: object | None = None,
    ) -> str:
        """Generate GLSL code for a listed colormap."""
        data = data or getattr(cmdata, name)
        n = len(data)
        data = _flatten(data)

        def eval_color(x: float) -> list[float]:
            y = x * (n - 1)
            i = int(math.floor(y))
            j = min(i + 1, n-1)
            w = y - i
            return [
                (1.0-w) * data[3*i+0] + w * data[3*j+0],
                (1.0-w) * data[3*i+1] + w * data[3*j+1],
                (1.0-w) * data[3*i+2] + w * data[3*j+2],
            ]

        return eval_color

    @staticmethod
    def make_weighted(
        name: str,
        data: object | None = None,
    ) -> str:
        """Generate GLSL code for a listed colormap."""
        data = data or getattr(cmdata, name)
        keys = list(data.keys())

        def eval_color(x: float) -> list[float]:
            wj, dj = keys[0], next(iter(data.values()))
            for wi, di in data.items():
                if x <= wi:
                    break
                wj, dj = wi, di
            wlow = x - wi
            wupp = wj - x
            wsum = wlow + wupp
            if wsum == 0:
                wlow = wupp = 0.5
            else:
                wlow = wlow / wsum
                wupp = wupp / wsum
            return [
                di[0] * wupp + dj[0] * wlow,
                di[1] * wupp + dj[1] * wlow,
                di[2] * wupp + dj[2] * wlow,
            ]

        return eval_color

    @staticmethod
    def make_segmented(
        name: str,
        data: object | None = None,
    ) -> str:
        """Generate GLSL code for a segmented colormap."""
        r, g, b = data or getattr(cmdata, name)
        r, g, b = map(_flatten, (r, g, b))

        def segment(x: float, data: list[float]) -> float:
            j = 0
            for i in range(len(data)//3):
                if x <= data[3*i]:
                    break
                j = i
            wlow = x - data[3*i]
            wupp = data[3*j] - x
            wsum = wlow + wupp
            if wsum == 0:
                wlow = wupp = 0.5
            else:
                wlow = wlow / wsum
                wupp = wupp / wsum
            x = data[3*i+2] * wupp + data[3*j+1] * wlow
            return min(max(x, 0), 1)

        def eval_color(x: float) -> list[float]:
            return [segment(x, r), segment(x, g), segment(x, b)]

        return eval_color


class colormaps:
    """
    Namespace for dynamic colormaps.

    Functions in this class return shader code.
    """

    _DEFAULT_LENGTH = object()

    orientation = dedent(
        """
        vec3 colormapOrient(vec3 orient) {
            vec3 result;
            result.r = abs(orient[0]);
            result.g = abs(orient[1]);
            result.b = abs(orient[2]);
            return clamp(result, 0.0, 1.0);
        }
        """).lstrip()

    @staticmethod
    def make_colormap(
        name: str,
        data: object | None = None,
        n: int = _DEFAULT_LENGTH,
    ) -> str:
        """Generate GLSL code for a colormap."""
        data = data or getattr(cmdata, name)
        if isinstance(data, list):
            return colormaps.make_listed(name, data, n)
        elif isinstance(data, tuple):
            return colormaps.make_segmented(name, data, n)
        elif isinstance(data, dict):
            return None
        return None

    @staticmethod
    def make_listed(
        name: str,
        data: object | None = None,
        n: int = _DEFAULT_LENGTH,
    ) -> str:
        """Generate GLSL code for a listed colormap."""
        data = data or getattr(cmdata, name)
        if n is colormaps._DEFAULT_LENGTH:
            n = 128
        if n:
            step = max(1, len(data) // n)
            data = data[::step]
        data = _flatten(data)
        n = len(data)
        return dedent(
            """
            vec4 %s(float x, bool alpha) {
                float cmap[%d] = float[]%s;
                float y = x * float(%d/3 - 1);
                int   i = int(floor(y));
                int   j = i + 1;
                float w = y - float(i);
                vec4  result;
                result.r = (1.0-w) * cmap[3*i+0] + w * cmap[3*j+0];
                result.g = (1.0-w) * cmap[3*i+1] + w * cmap[3*j+1];
                result.b = (1.0-w) * cmap[3*i+2] + w * cmap[3*j+2];
                result.a = (alpha ? x : 1.0);
                return clamp(result, 0.0, 1.0);
            }
            """ % (name, n, str(data), n)
        ).strip()

    @staticmethod
    def make_segmented(
        name: str,
        data: object | None = None,
        n: int = _DEFAULT_LENGTH,
    ) -> str:
        """Generate GLSL code for a segmented colormap."""
        def subsample(data: list[float], n: int) -> list[float]:
            if n is None:
                return cmap
            step = max(1, len(r) // n)
            if len(data) % step != 1:
                last = data[-1:]
            else:
                last = []
            data = data[::step] + last
            return data

        if n is colormaps._DEFAULT_LENGTH:
            n = 32

        r, g, b = data or getattr(cmdata, name)
        r, g, b = map(lambda x: subsample(x, n), (r, g, b))
        r, g, b = map(_flatten, (r, g, b))
        nr, ng, nb = len(r), len(g), len(b)
        segment = dedent(
            """
            float segment%s(float x)
            {
                float cmap[%d] = float[]%s;
                int i, j;
                for (int c = 0; c < cmap.length()/3; ++c) {
                    if (x > cmap[3*c]) {
                        i = c;
                        break;
                    }
                }
                j = (i+1 < cmap.length() ? i+1 : i);
                float wlow = x - cmap[3*i];
                float wupp = cmap[3*j] - x;
                float wsum = wlow + wupp;
                wlow = wlow / wsum;
                wupp = wupp / wsum;
                x = cmap[3*i+2] * wupp + cmap[3*j+1] * wlow;
                return clamp(x, 0.0, 1.0);
            }
            """
        ).strip()
        segment = [segment % (name, length, str(cmap)) for name, length, cmap
                   in (('r', nr, r), ('g', ng, g), ('b', nb, b))]
        segment = '\n'.join(segment)
        return segment + dedent(
            """
            vec4 %s(float x, bool alpha) {
                vec4  result;
                result.r = segmentr(x);
                result.g = segmentg(x);
                result.b = segmentb(x);
                result.a = (alpha ? x : 1.0);
                return clamp(result, 0.0, 1.0);
            }
            """ % name
        ).strip()


for name in filter(lambda x: not x[0] == '_', dir(cmdata)):
    cmap = colormaps.make_colormap(name)
    if cmap:
        setattr(colormaps, name, cmap)


class shaders:
    """Namespace for full-fledged shaders."""

    # Classic greyscale map, with optional intensity-based opacity.
    greyscale = dedent(
        """
        #uicontrol invlerp normalized
        #uicontrol bool alpha_depth checkbox(default=false)
        void main() {
            float x = normalized();
            if(alpha_depth) {
                vec4 result;
                result.r = x;
                result.g = x;
                result.b = x;
                result.a = x;
                emitRGBA(result);
            } else {
                emitGrayscale(x);
            }
        }
        """).lstrip()

    class skeleton:
        """Shaders for skeletons."""

        # Shader for streamlines that have an `orientation` attribute
        orientation = colormaps.orientation + '\n' + dedent(
            """
            #uicontrol bool orient_color checkbox(default=true)
            void main() {
                if (orient_color)
                    emitRGB(colormapOrient(orientation));
                else
                    emitDefault();
            }
            """).lstrip()

    # Classic RGB map, with brightness + contrast controls
    orientation = colormaps.orientation + dedent(
        """
        #uicontrol invlerp normalized
        #uicontrol bool alpha_depth checkbox(default=false)
        #uicontrol bool alpha_depth checkbox(default=false)
        void main() {
            vec3 orient = vec3(
                toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue(2))
            );
            float alpha = sqrt(
                orient.r * orient.r +
                orient.g * orient.g +
                orient.b * orient.b
            );
            vec3 rgb = colormapOrient(orient);

            if (alpha_depth) {
                vec4 rgba = vec4(
                    rgb.r,
                    rgb.g,
                    rgb.b,
                    normalized(),
                );
                emitRGBA(rgba);
            } else {
                emitRGB(rgb);
            }
        }
        """
    )

    # Classic RGB map, with brightness + contrast controls
    rgb = dedent(
        """
        #uicontrol float brightness slider(min=-1, max=1)
        #uicontrol float contrast slider(min=-3, max=3, step=0.01)
        void main() {
            vec color = vec3(
                toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue(2))
            );
            emitRGB((color + brightness) * exp(contrast));
        }
        """
    )

    @staticmethod
    def colormap(cmap: str) -> str:
        """
        Generate a shader based on a known colormap.

        Parameters
        ----------
        cmap : str
            Name of a colormap. Must be a valid attribute from the
            `colormaps` namespace.

        """
        return getattr(colormaps, cmap) + '\n' + dedent(
            """
            #uicontrol invlerp normalized
            #uicontrol bool alpha_depth checkbox(default=false)
            void main() {
                emitRGBA(%s(normalized(), alpha_depth));
            }
            """ % cmap).lstrip()

    @staticmethod
    def lut(lut: dict | str | PathLike | IO) -> str:
        """
        Generate a shader based on a lookup table.

        Parameters
        ----------
        lut : str | dict[int, tuple[str, tuple[float,float,float,float]]]
            Path to a FreeSurfer lookup table, or pre-loaded LUT.
        """
        if not isinstance(lut, dict):
            lut = load_fs_lut(lut)
        labels, colors = list(lut.keys()), list(lut.values())
        colors = [color[1] for color in colors]
        shader = dedent(
            """
            void main() {
            vec4 color;
            int label = int(getDataValue(0).value);
            if (label == int(%d))
                color = vec4(%f, %f, %f, %f);
            """ % (labels.pop(0), *colors.pop(0))
        )
        for label, color in zip(labels, colors):
            shader += dedent(
                """
                else if (label == int(%d))
                    color = vec4(%f, %f, %f, %f);
                """ % (label, *color)
            )
        shader += dedent(
            """
            else
                color = vec4(0.0, 0.0, 0.0, 0.0);
            emitRGBA(color);
            }
            """
        )
        return shader

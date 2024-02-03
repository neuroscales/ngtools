from textwrap import dedent


class colormaps:

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

    blackred = dedent(
        """
        vec3 blackred(float greyscale) {
            vec3 result;
            result.r = greyscale;
            result.g = 0.0;
            result.b = 0.0;
            return clamp(result, 0.0, 1.0);
        }
        """).lstrip()

    blackgreen = dedent(
        """
        vec3 blackgreen(float greyscale) {
            vec3 result;
            result.r = 0.0;
            result.g = greyscale;
            result.b = 0.0;
            return clamp(result, 0.0, 1.0);
        }
        """).lstrip()

    blackblue = dedent(
        """
        vec3 blackblue(float greyscale) {
            vec3 result;
            result.r = 0.0;
            result.g = 0.0;
            result.b = greyscale;
            return clamp(result, 0.0, 1.0);
        }
        """).lstrip()

    whitered = dedent(
        """
        vec3 whitered(float greyscale) {
            vec3 result;
            result.r = 1.0;
            result.g = 1.0 - greyscale;
            result.b = 1.0 - greyscale;
            return clamp(result, 0.0, 1.0);
        }
        """).lstrip()

    whitegreen = dedent(
        """
        vec3 whitegreen(float greyscale) {
            vec3 result;
            result.r = 1.0 - greyscale;
            result.g = 1.0;
            result.b = 1.0 - greyscale;
            return clamp(result, 0.0, 1.0);
        }
        """).lstrip()

    whiteblue = dedent(
        """
        vec3 whiteblue(float greyscale) {
            vec3 result;
            result.r = 1.0 - greyscale;
            result.g = 1.0 - greyscale;
            result.b = 1.0;
            return clamp(result, 0.0, 1.0);
        }
        """).lstrip()


class shaders:
    greyscale = dedent(
        """
        #uicontrol invlerp normalized
        void main() {
            emitGrayscale(normalized());
        }
        """).lstrip()

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

    @staticmethod
    def colormap(cmap):
        return getattr(colormaps, cmap) + '\n' + dedent(
            """
            #uicontrol invlerp normalized
            void main() {
                emitRGB(%s(normalized()));
            }
            """ % cmap).lstrip()

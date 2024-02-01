class colormaps:

    orientation = """
    vec3 colormapOrient(vec3 orient) {
        vec3 result;
        result.r = abs(orient[0]);
        result.g = abs(orient[1]);
        result.b = abs(orient[2]);
        return clamp(result, 0.0, 1.0);
    }
    """


class shaders:
    greyscale = """
    #uicontrol invlerp normalized
    void main() {
        emitGrayscale(normalized());
    }
    """

    orientation = """
    #uicontrol bool orient_color checkbox(default=true)
    void main() {
        if (orient_color)
            emitRGB(colormapOrient(orientation));
        else
            emitDefault();
    }
    """

# Filtering Tracts

If we load a tract file as a precomputed annotation we can apply filters through the use of local annotation files. We can create a json file to represent a how different filters interact using boolean operators. An example of a json file that filters for all tracts that intersect with 1 of two spheres but don't intersect with a disk between them can be found below.

Tip: instead of changing where the center is, edit the transformation matrix. This will make it easier to move and scale the filter in the web browser

supported operations: "and", "or", "not", "xor"

supported filters:

```json
{
    "type": "ellipsoid",
    "center": [ 0, 0, 0 ],
    "radii": [ 1, 1, 1 ],
    "id": "id",
    "name": "name",
    "transform": [
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 1.0, 0.0]
    ]
}


{
    "type": "axis_aligned_bounding_box",
    "pointA": [ -1, -1, -1 ],
    "pointB": [ 1, 1, 1 ],
    "id": "id",
    "name": "name",
    "transform": [
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 1.0, 0.0]
    ]
}
```

```json
{
    "operation": "and",
    "layer": [
        {
            "operation": "or",
            "layer": [
                {
                    "type": "ellipsoid",
                    "center": [0.0, 0.0, 0.0],
                    "radii": [1.0, 1.0, 1.0],
                    "id": "100001",
                    "name": "addition_1",
                    "transform": [
                        [ 2.0, 0.0, 0.0, 70.0 ],
                        [ 0.0, 2.0, 0.0, 70.0 ],
                        [ 0.0, 0.0, 2.0, 30.0 ]
                    ]
                },
                {
                    "type": "ellipsoid",
                    "center": [ 0.0, 0.0, 0.0 ],
                    "radii": [ 1.0, 1.0, 1.0 ],
                    "id": "100001",
                    "name": "addition_2",
                    "transform": [
                        [ 2.0, 0.0, 0.0, 70.0 ],
                        [ 0.0, 2.0, 0.0, 70.0 ],
                        [ 0.0, 0.0, 2.0, 40.0 ]
                    ]
                }
            ]
        },
        {
            "operation": "not",
            "layer": {
                "type": "ellipsoid",
                "center": [ 0, 0, 0 ],
                "radii": [ 1, 1, 1 ],
                "id": "100001",
                "name": "subtraction",
                "transform": [
                    [ 100.0, 0.0, 0.0, 70.0 ],
                    [ 0.0, 100.0, 0.0, 70.0 ],
                    [ 0.0, 0.0, 0.001, 35.0]
                ]
            }
        }
    ]
}
```

In order to have a filter file applied to a tract annotation layer run the following command:

```shell
filter "name of tract layer" "path to json file"
```

Additionally we can edit the filter's transformation matrix in the browser than export to a new json file with the following command

```shell
export "name of tract layer" "path to json file"
```
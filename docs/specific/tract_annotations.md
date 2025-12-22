# Tractograms

## Tracts in precomputed annotation format

In order to load a precomputed annotation folder as tracts run

    load "tracts://(path to precomputed annotation folder)"

!!! tip
    If a file named `transform.lta` is present in the annotation directory,
    this transformation will be applied automatically, assuming that another
    transformation was not specified using the `--transform` option.

## Other types of precomputed annotations

To load a precomputed annotation without assuming that it represents a
tractogram, run

    load "precomputed://(path to the precomputed annotation folder)"

## Load classic tractograms into precomputed annotations

In order to load a `TRK` or `TCK` file into a virtual precomputed annotation
file, run

    load "(path to .trk file)"

If the `TRK` or `TCK` file does not end in `.trk` or `.tck`, a format hint
can be provided as a "protocol":

    load "trk://(path to file without extension)"

## Load classic tractograms into precomputed skeletons

Tractograms can also be loaded into virtual precomputed skeletons:

    load "tractsv1://(path to file)"

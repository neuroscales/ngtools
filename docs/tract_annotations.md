# Tract Annotation

## precomputed as tract annotations
In order to load a precomputed annotation folder as tracts run the follwoing

    load "tracts://(path to precomputed annotation folder)"

If we have a transform.lta file in this directory that transformation will automatically be applied assuming we did not specify a different transformation

## precomputed annotations
If we wish to load a precomputed annotation not as tracts run the following

    load "precomputed://(path to the precomputed annotation folder)"

## trk file as tract annotations
In order to load a trk, tck, or tract file as a precomputed annotation run the followign

    load "(path to .trk file)"

## unknown file as tract annotations
If we have a file that has the format of a .trk file but not one of the above extensions we will need to run the following

    load "trk://(path to file)"

## trk file as skeleton
If we wish to load .trk file as a skeleton layer run the following

    load "tractsv1://(path to file)"
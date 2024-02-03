import neuroglancer as ng
import numpy as np
import itertools


def to_square(affine):
    """Ensure an affine matrix is in square (homogeneous) form"""
    if affine.shape[0] == affine.shape[1]:
        return affine
    new_affine = np.eye(affine.shape[-1])
    new_affine[:-1, :] = affine
    return new_affine


letter2full = {
    'r': 'right',
    'l': 'left',
    'a': 'anterior',
    'p': 'posterior',
    'i': 'inferior',
    's': 'superior',
}


def _get_neuronames():
    letters = (('r', 'l'), ('a', 'p'), ('i', 's'))
    all_neurospaces = []
    for perm in itertools.permutations(range(3)):
        for flip in itertools.product([0, 1], repeat=3):
            space = ''.join([letters[p][f] for p, f in zip(perm, flip)])
            all_neurospaces.append(space)
    return all_neurospaces


def _get_src2ras(name):
    xyz2ras = {'x': 'r', 'y': 'a', 'z': 's'}
    name = [xyz2ras.get(x, x) for x in name.lower()]
    pos = list('ras')
    neg = list('lpi')
    mat = np.eye(4)
    # permutation
    perm = [pos.index(x) if x in pos else neg.index(x) for x in name]
    perm += [3]
    mat = mat[:, perm]
    # flip
    flip = [-1 if x in neg else 1 for x in name]
    flip += [1]
    mat *= np.asarray(flip)
    return mat


def _get_src2dst(src, dst):
    src_to_ras = _get_src2ras(src)
    dst_to_ras = _get_src2ras(dst)
    return np.linalg.inv(dst_to_ras) @ src_to_ras


# Build coordinate spaces
neurospaces = {
    name: ng.CoordinateSpace(
        names=[letter2full[letter.lower()] for letter in name],
        scales=[1]*3,
        units=['mm']*3,
    )
    for name in _get_neuronames()
}


defaultspaces = {
    ''.join(name): ng.CoordinateSpace(
        names=name,
        scales=[1]*3,
        units=['mm']*3,
    )
    for name in itertools.permutations(['x', 'y', 'z'])
}
default = defaultspaces['xyz']
neurospaces.update(defaultspaces)


# Build coordiinate transforms
neurotransforms = {
    (neurospaces[src], neurospaces[dst]): ng.CoordinateSpaceTransform(
        matrix=_get_src2dst(src, dst)[:3, :4],
        input_dimensions=neurospaces[src],
        output_dimensions=neurospaces[dst],
    )
    for src in _get_neuronames()
    for dst in _get_neuronames()
}


for name in itertools.permutations(['x', 'y', 'z']):
    name = ''.join(name)
    for neuroname in _get_neuronames():
        matrix = _get_src2dst(neuroname, name)
        neurotransforms[(neurospaces[neuroname], defaultspaces[name])] \
            = ng.CoordinateSpaceTransform(
                matrix=matrix[:3, :4],
                input_dimensions=neurospaces[neuroname],
                output_dimensions=defaultspaces[name],
            )
        neurotransforms[(defaultspaces[name], neurospaces[neuroname])] \
            = ng.CoordinateSpaceTransform(
                matrix=np.linalg.inv(matrix)[:3, :4],
                input_dimensions=defaultspaces[name],
                output_dimensions=neurospaces[neuroname],
            )


def compose(left, right):
    """Compose two transforms"""
    input_dimensions = right.input_dimensions
    output_dimensions = left.output_dimensions
    right_dimensions = right.output_dimensions
    left_dimensions = left.input_dimensions

    onames = output_dimensions.names
    ounits = output_dimensions.units
    oscles = output_dimensions.scales
    inames = input_dimensions.names
    iunits = input_dimensions.units
    iscles = input_dimensions.scales
    lnames = left_dimensions.names
    lunits = left_dimensions.units
    lscles = left_dimensions.scales
    rnames = right_dimensions.names
    runits = right_dimensions.units
    rscles = right_dimensions.scales

    lmatrix = left.matrix
    if lmatrix is None:
        lmatrix = np.eye(len(onames)+1)[:-1]
    lmatrix = to_square(lmatrix)
    rmatrix = right.matrix
    if rmatrix is None:
        rmatrix = np.eye(len(inames)+1)[:-1]
    rmatrix = to_square(rmatrix)

    # delete output axes that do not exist in the input space
    delnames = [n for n in lnames if n not in rnames]
    odel = [onames.index(n) for n in delnames]
    ldel = [lnames.index(n) for n in delnames]
    lkeep = [i for i in range(len(lnames)) if i not in ldel]
    okeep = [i for i in range(len(onames)) if i not in odel]
    lnames = [lnames[i] for i in lkeep]
    lunits = [lunits[i] for i in lkeep]
    lscles = [lscles[i] for i in lkeep]
    onames = [onames[i] for i in okeep]
    ounits = [ounits[i] for i in okeep]
    oscles = [oscles[i] for i in okeep]
    lmatrix = lmatrix[okeep + [-1], :][:, lkeep + [-1]]

    # add axes that exist in the input space but not in the output space
    extnames = [n for n in rnames if n not in lnames]
    onames += extnames
    ounits += [runits[rnames.index(n)] for n in extnames]
    oscles += [rscles[rnames.index(n)] for n in extnames]
    lnames += extnames
    lunits += [runits[rnames.index(n)] for n in extnames]
    lscles += [rscles[rnames.index(n)] for n in extnames]
    lmatrix0 = lmatrix
    lmatrix = np.eye(len(onames)+1)
    n0 = len(lmatrix0)-1
    # copy compatible part
    lmatrix[:n0, :n0] = lmatrix0[:-1, :-1]
    lmatrix[:n0, -1] = lmatrix0[:-1, -1]
    # copy extra part
    rsub = [rnames.index(n) for n in extnames]
    isub = [inames.index(n) for n in extnames]
    lmatrix[n0:-1, n0:-1] = rmatrix[rsub, :][:, isub]
    lmatrix[n0:-1, -1] = rmatrix[rsub, -1]
    # reorder right side
    lmatrix = lmatrix[:, [lnames.index(n) for n in rnames] + [-1]]

    matrix = lmatrix @ rmatrix
    oscles = np.abs(lmatrix[:-1, :-1] @ rscles)

    T = ng.CoordinateSpaceTransform(
        input_dimensions=ng.CoordinateSpace(
            names=inames,
            units=iunits,
            scales=iscles,
        ),
        output_dimensions=ng.CoordinateSpace(
            names=onames,
            units=ounits,
            scales=oscles,
        ),
        matrix=matrix[:-1],
    )
    return T

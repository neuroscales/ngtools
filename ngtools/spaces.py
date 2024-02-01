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

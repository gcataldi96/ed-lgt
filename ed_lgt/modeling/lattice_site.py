# %%
"""
Bases types
-----------
- flat:   `[(l, s)], i -> j` = indicex where the i-th basis entries end)
- sparse: `i -> [(l, s)]`
- dense:  `i, l, s -> 0 or 1`
- mixed:  `i, l -> s`  // -1 means no bond

Legend
------
- `i` = local state index
- `o` = space orientation
- `d` = space direction
- `s` = specie
- `l` = `(o, d)` = link
- `b` = `(l, s) = ((o, d), s)` = bond
- `x` = `(x1, ..., xD)` = coordinate

For any basis but the flat one, `basis[configs]` replaces `i` with `x`,
yielding the corresponding config representation.

For cubic lattices, switching from a site to a link basis amounts 
to replacing `l = (1, d)` with `d` (and reshaping arrays).
"""

from collections import defaultdict
from itertools import count, chain, combinations, product
from functools import cache

import numpy as np
import scipy as sp


def tuple_or_range(iterable_or_max, endpoint):
    """
    Casts iterable_or_max as a tuple if it is iterable,
    otherwise returns `range(iterable_or_max + endpoint)`.
    """
    try:
        return tuple(iterable_or_max)
    except TypeError:
        return range(iterable_or_max + endpoint)


@cache
def lbl(*lbls):
    """
    Generates a label for an operator or a basis state.
    """
    return "_".join(map(str, lbls)).replace(" ", "")


def get_composite_site(num_bonds, links, species):
    """
    Builds the composite local Hilbert space of a polymer model.

    Parameters
    ----------
    num_bonds : int or iterable
        maximum number of bonds per site or list of valid bond counts;
        num_bonds can be used to control the kind of polymers we find, e.g.
        for space-filling num_bonds >= 1, for self-avoiding num_bonds <= 2

    links : int or iterable
        number of links attached to each site (lattice connectivity)
        or list of link names

    species : int or iterable
        number of species or list of species names

    Returns
    -------
    list of tuple, dict of str : scipy.sparse array
        local Hilbert space basis in sparse form and some operators
    """
    # parse args
    num_bonds = tuple_or_range(num_bonds, endpoint=True)
    links = tuple_or_range(links, endpoint=False)
    species = tuple_or_range(species, endpoint=False)

    # encode basis states as lists of active bonds
    enum = count()
    basis = []
    projs = defaultdict(list)
    diags = defaultdict(list)
    for n in num_bonds:
        for ll, ss in product(combinations(links, n), product(species, repeat=n)):
            i = next(enum)
            state = tuple(zip(ll, ss))
            basis.append(state)

            # BUILDING OPERATORS
            # quantum_green_tea is slow at initializing operators, we remove surios ones
            # should a new operator be nedded look for it in the commented lines below
            diags["num_bond"].append(n)
            diags["num_monomer"].append(int(n > 0))
            # diags["num_bulk"].append(int(n > 1))
            diags["num_edge"].append(int(n == 1))
            # projs[lbl("is", state)].append(i)
            # projs[lbl("is_l", ll)].append(i)
            projs[lbl("is_s", tuple(sorted(ss)))].append(i)
            for s in species:
                diags[lbl("num_s", s)].append(ss.count(s))
            # should replace set() with powerset() for generic has_ operators
            # https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.powerset
            for s in set(ss):
                # projs[lbl("has_s", s)].append(i)
                for not_l in set(links) - set(ll):
                    projs[lbl("hasnt_l", not_l, "has_s", s)].append(i)
            for l in set(ll):
                projs[lbl("has_l", l)].append(i)
            for ls in state:
                projs[lbl("has", ls)].append(i)  # to be deleted afterwards

    d = len(basis)
    eye = sp.sparse.eye(d)

    ops = {}
    ops["id"] = eye
    for k, ijs in projs.items():
        ops[k] = sp.sparse.coo_array(([1] * len(ijs), (ijs, ijs)), shape=(d, d))
    for k, diag in diags.items():
        ops[k] = sp.sparse.dia_array(([diag], [0]), shape=(d, d))
    for ls in product(links, species):
        # .pop() of no longer needed operator
        ops[lbl("penalty_l", ls)] = eye - 2 * ops.pop(lbl("has", ls))

    return basis, ops


def basis_sparse_to_mixed(sbasis, links):
    """See module docstrig"""
    basis = basis = [[e.get(l, -1) for l in links] for e in map(dict, sbasis)]
    return np.asarray(basis, dtype=int)


def basis_sparse_to_flat(sbasis):
    """See module docstrig"""
    splits = np.cumsum([len(bonds) for bonds in sbasis[:-1]])
    basis = list(chain.from_iterable(sbasis))
    return basis, splits


def basis_flat_to_sparse(fbasis, splits):
    """See module docstrig"""
    basis = [[(d, s) for d, s in bonds] for bonds in np.split(fbasis, splits)]
    return np.fromiter(basis, object)  # prevent iteration ofer inner lists


def basis_sparse_to_dense(sbasis, *inds):
    """See module docstrig"""
    basis = [[tuple(j) in e for j in product(*inds)] for e in sbasis]
    return np.asarray(basis, dtype=int).reshape(len(sbasis), *map(len, inds))


def basis_dense_to_sparse(dbasis):
    """See module docstrig"""
    basis = [[tuple(j) for j in np.argwhere(e)] for e in dbasis]
    return np.fromiter(basis, object)  # prevent iteration ofer inner lists


def get_bonds(sconfig):
    """
    Returns a list `[(x, d, s)]` of the bonds in a (single) sparse config.
    Similar to `np.argwhere(dconfig)` from dense config.
    """
    return [(x, *b) for x, bonds in np.ndenumerate(sconfig) for b in bonds]


# %%

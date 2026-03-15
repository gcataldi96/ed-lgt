import argparse
from time import perf_counter

import numpy as np

from edlgt.modeling.lattice_geometry import get_lattice_link_site_pairs
from edlgt.symmetries import nbody_term, symmetry_sector_configs


def _time_call(callable_fn, repeats: int):
    # Warmup (includes JIT compilation on first call).
    callable_fn()
    times = np.zeros(repeats, dtype=np.float64)
    result = None
    for rep in range(repeats):
        t0 = perf_counter()
        result = callable_fn()
        times[rep] = perf_counter() - t0
    return result, times


def _stats(times: np.ndarray) -> str:
    return (
        f"min={times.min():.6f}s "
        f"mean={times.mean():.6f}s "
        f"median={np.median(times):.6f}s "
        f"max={times.max():.6f}s"
    )


def _build_sector_inputs(n_sites: int, loc_dim: int, sector_target: float):
    loc_dims = np.full(n_sites, loc_dim, dtype=np.int32)

    # One global U(1)-like generator: local labels [0, 1, ..., d-1].
    glob_op_diags = np.zeros((1, n_sites, loc_dim), dtype=np.float64)
    local_labels = np.arange(loc_dim, dtype=np.float64)
    for site_idx in range(n_sites):
        glob_op_diags[0, site_idx, :] = local_labels
    glob_sectors = np.array([sector_target], dtype=np.float64)

    # Link generator values all set to zero -> always satisfies sector 0.
    # This keeps link-check cost in the benchmark without adding extra filtering.
    link_op_diags = np.zeros((1, 2, n_sites, loc_dim), dtype=np.float64)
    link_sectors = np.array([0.0], dtype=np.float64)
    pair_list = get_lattice_link_site_pairs([n_sites], [False])

    return loc_dims, glob_op_diags, glob_sectors, link_op_diags, link_sectors, pair_list


def _build_representative_two_site_operator(
    n_sites: int, loc_dim: int, op_sites: np.ndarray
) -> np.ndarray:
    op_list = np.zeros((2, n_sites, loc_dim, loc_dim), dtype=np.complex128)

    # Dense-ish local matrices to trigger non-trivial transition enumeration.
    mat0 = np.eye(loc_dim, dtype=np.complex128)
    mat1 = np.eye(loc_dim, dtype=np.complex128)
    for ii in range(loc_dim):
        for jj in range(loc_dim):
            if ii != jj:
                mat0[ii, jj] = (0.05 + 0.01j) * (1 + ii + jj)
                mat1[ii, jj] = (-0.03 + 0.02j) * (1 + ii + jj)

    op_list[0, int(op_sites[0])] = mat0
    op_list[1, int(op_sites[1])] = mat1
    return op_list


def main():
    parser = argparse.ArgumentParser(
        description="Micro-benchmark: symmetry sector build + representative Hamiltonian term."
    )
    parser.add_argument("--n-sites", type=int, default=10)
    parser.add_argument("--loc-dim", type=int, default=2)
    parser.add_argument("--sector-target", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.n_sites < 2:
        raise ValueError("n_sites must be >= 2.")
    if args.loc_dim < 2:
        raise ValueError("loc_dim must be >= 2.")
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1.")

    if args.sector_target is None:
        # Half-filling-like target for loc_dim=2.
        # For larger local dimensions this still gives a deterministic baseline.
        sector_target = float(args.n_sites // 2)
    else:
        sector_target = float(args.sector_target)

    (
        loc_dims,
        glob_op_diags,
        glob_sectors,
        link_op_diags,
        link_sectors,
        pair_list,
    ) = _build_sector_inputs(
        n_sites=args.n_sites,
        loc_dim=args.loc_dim,
        sector_target=sector_target,
    )

    sector_fn = lambda: symmetry_sector_configs(
        loc_dims=loc_dims,
        glob_op_diags=glob_op_diags,
        glob_sectors=glob_sectors,
        sym_type_flag="U",
        link_op_diags=link_op_diags,
        link_sectors=link_sectors,
        pair_list=pair_list,
    )
    sector_configs, sector_times = _time_call(sector_fn, repeats=args.repeats)

    op_sites = np.array([0, 1], dtype=np.int32)
    op_list = _build_representative_two_site_operator(
        n_sites=args.n_sites, loc_dim=args.loc_dim, op_sites=op_sites
    )
    term_fn = lambda: nbody_term(
        op_list=op_list,
        op_sites_list=op_sites,
        sector_configs=sector_configs,
        momentum_basis=None,
    )
    term_triplets, term_times = _time_call(term_fn, repeats=args.repeats)

    print("====================================================")
    print("Benchmark: symmetry sector + representative 2-site term")
    print("----------------------------------------------------")
    print(
        f"Inputs: n_sites={args.n_sites}, loc_dim={args.loc_dim}, "
        f"sector_target={sector_target}, repeats={args.repeats}"
    )
    print("----------------------------------------------------")
    print(
        f"Symmetry sector: dim={int(sector_configs.shape[0])} "
        f"({sector_configs.shape[1]} sites), {_stats(sector_times)}"
    )
    print(
        f"2-site term kernel: nnz={int(term_triplets[0].shape[0])}, "
        f"{_stats(term_times)}"
    )
    print("====================================================")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate the spin-polarized (zeta-aware) pretrain-data file for a cluster run.

A `python -m xcquinox.alec.cluster submit ... --polarized` run pretrains
spin-polarization-aware networks and therefore expects a
``pretrain_data_polarized.npz`` (carrying a ``zeta_all`` column) in its
``pretrain.data_dir`` -- ``run_pretrain`` selects that filename automatically for
a polarized architecture and fails fast if it is absent. Run this once to stage
it alongside the unpolarized ``pretrain_data.npz``.

Usage::

    python scripts/generate_polarized_pretrain_data.py --out-dir /path/to/pretrain_data

The default molecules and basis match the standard pretrain-data generator
(H, He, O, N; def2-svp); the grid level is the production 3, because the set
contains the spatially degenerate O atom. The file carries spin-resolved Fx/Fc
targets, Becke ``weights_all`` (integration-mode loss), descriptor columns
(``cusp_all``/``dm_all``), and the ``zeta_all`` polarization column.
"""
import argparse

from xcquinox.alec.pretrain_data_gen import (
    generate_pretrain_data_npz, DEFAULT_PRETRAIN_ATOMS,
    DEFAULT_BASIS, COARSE_DEGENERATE_MIN_GRID_LEVEL,
)

#: Grid level this script writes at. The default atom set contains O, a
#: spatially degenerate free atom whose rows below
#: ``COARSE_DEGENERATE_MIN_GRID_LEVEL`` are one arbitrary member of the P-term
#: manifold rather than the reproducible quantity the manifest records, so the
#: default here is the PRODUCTION level rather than the library's historical 1.
DEFAULT_SCRIPT_GRID_LEVEL = COARSE_DEGENERATE_MIN_GRID_LEVEL


def _parse_atoms(spec):
    """Parse ``H:1,He:0,O:2,N:3`` -> ((\"H\",1), ...)."""
    out = []
    for tok in spec.split(","):
        sym, _, spin = tok.partition(":")
        out.append((sym.strip(), int(spin)))
    return tuple(out)


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True,
                   help="directory to write pretrain_data_polarized.npz into "
                        "(the run's pretrain.data_dir)")
    p.add_argument("--basis", default=DEFAULT_BASIS)
    p.add_argument("--grid-level", type=int,
                   default=DEFAULT_SCRIPT_GRID_LEVEL,
                   help=f"integration grid level (default "
                        f"{DEFAULT_SCRIPT_GRID_LEVEL}, the production level: "
                        "below it a spatially degenerate free atom's rows are "
                        "not reproducible between processes and the generator "
                        "refuses to write them)")
    p.add_argument("--atoms", default=None,
                   help="override pretrain atoms as 'H:1,He:0,O:2,N:3' "
                        "(symbol:spin); default matches the standard generator")
    p.add_argument("--no-descriptors", action="store_true",
                   help="skip cusp/dm descriptor columns (smaller file; only valid "
                        "if no run uses descriptor architectures)")
    p.add_argument("--density-fit", action="store_true",
                   help="density-fit the per-atom SCF Coulomb build (auxbasis "
                        "auto-selected from the basis) so a large basis stays "
                        "within node RAM")
    p.add_argument("--allow-irreproducible-degenerate", action="store_true",
                   help="build a spatially degenerate free atom's rows even "
                        "though the identity is not reproducible (a coarse "
                        "grid, or the orientation lock off); the manifest "
                        "records that the permission was exercised")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    atoms = _parse_atoms(args.atoms) if args.atoms else DEFAULT_PRETRAIN_ATOMS
    path = generate_pretrain_data_npz(
        args.out_dir, atoms=atoms, basis=args.basis, grid_level=args.grid_level,
        polarized=True, descriptors=not args.no_descriptors,
        density_fit=args.density_fit,
        allow_irreproducible_degenerate=args.allow_irreproducible_degenerate)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

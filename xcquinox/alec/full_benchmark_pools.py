"""Full GMTKN55-BH76 + W4-11 held-out benchmark pools.

This module exposes the full 76-reaction BH76 (barrier heights) and
140-reaction W4-11 (atomization energies) benchmark sets in the same dict
schema that :func:`reaction_mae_kcalmol` (originally in
``notebooks/analysis/local_reeval.py``) consumes, so the cluster eval can
write apples-to-apples held-out MAE rows without re-deriving the math.

Source data lives at
``scripts/script_data/gmtkn55/{BH76,W4-11}/``:

  * ``<set>/.res``: shell-script reaction list (one ``tmer ...`` line per
    reaction). BH76 uses signed stoichiometry (``-1 -1 +1``) for barrier
    forward energies; W4-11 uses ``-1 +n_atom1 +n_atom2 ...`` for
    atomization energies.
  * ``<set>/<species>/struc.xyz``: Cartesian geometry in ANGSTROM (standard
    .xyz convention; the sibling ``coord`` file is the bohr/TURBOMOLE copy).
  * ``<set>/<species>/coord``: TURBOMOLE-format coords (bohr) + ``$eht
    charge=N unpaired=M`` for spin (2S) and charge metadata.

Performance: parsing the ``.res`` files + reading 200+ ``struc.xyz`` +
``coord`` files on every SLURM task adds ~5 s of startup per spec. We
sidestep that by pre-parsing once into ``xcquinox/alec/data/{bh76,w411}_
full_pool.json`` (committed to the repo via
``scripts/rebuild_full_benchmark_pools.py``) and reading the JSON at module
import. Set ``XCQUINOX_REBUILD_FULL_POOLS=1`` to force re-parsing from the
GMTKN55 source instead (useful if the JSON falls out of sync).

Reaction-dict schema (matches
``xcquinox.alec.eval_probes.PROBE_C_BH76_OUT_OF_TRAINING``):

    {
        "name":               str,                   # e.g. "bh76_h_n2o_n2ohts"
        "source_pool":        "bh76" | "w411",
        "reactants":          list[str],             # species names
        "products":           list[str],             # species names
        "coeffs":             list[float],           # signed Σc_i·E_i convention
        "reaction_energy_ref": float,                # kcal/mol
        "species_spins":      dict[str, int],        # 2S per species
        "species_charges":    dict[str, int],        # charge per species
        "source":             str,                   # citation
    }
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from xcquinox.alec.config import MoleculeSpec

# Per-module-import caches so a second call to load_full_bh76 / load_full_w411
# does not re-read the JSON.
_BH76_CACHE: Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]] | None = None
_W411_CACHE: Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]] | None = None

# Bohr -> angstrom (CODATA 2018). NOTE: GMTKN55 ``struc.xyz`` files are already
# in ANGSTROM (the bohr copy is the sibling TURBOMOLE ``coord`` file), so the
# regen path does NOT convert struc.xyz: it stores those angstrom coordinates
# verbatim. The constant is kept only for reference/round-trips; struc.xyz must
# NOT be divided by it (doing so shrinks every held-out molecule ~1.89x).
BOHR_PER_ANGSTROM = 1.8897261246257702


# ---------------------------------------------------------------------------
# JSON cache path resolution
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"
BH76_JSON_PATH = _DATA_DIR / "bh76_full_pool.json"
W411_JSON_PATH = _DATA_DIR / "w411_full_pool.json"

# GMTKN55 source root, the regen script reads from here; runtime loaders
# only touch it when XCQUINOX_REBUILD_FULL_POOLS=1.
_GMTKN55_ROOT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "script_data" / "gmtkn55"
)
BH76_SOURCE_DIR = _GMTKN55_ROOT / "BH76"
W411_SOURCE_DIR = _GMTKN55_ROOT / "W4-11"


# ---------------------------------------------------------------------------
# GMTKN55 .res parser, used at JSON build time, NOT at every cluster task
# ---------------------------------------------------------------------------

# BH76 lines: "$tmer  h/$f  n2o/$f  n2ohts/$f  x  -1  -1  1  $w  17.7"
# Capture: species tokens before 'x', integer coeffs after 'x', float ref after $w.
_RE_BH76_LINE = re.compile(
    r"^\$tmer\s+(?P<spec>.+?)\s+x\s+(?P<coeffs>-?\d+(?:\s+-?\d+)*)\s+\$w\s+"
    r"(?P<ref>-?\d+(?:\.\d+)?)\s*$"
)

# W4-11 lines: "$tmer {h2,h}/$f  x -1 2 $w 109.493"
# The brace group is "molecule,atom1,atom2,..." (atomization decomposition).
_RE_W411_LINE = re.compile(
    r"^\$tmer\s+\{(?P<species>[^}]+)\}/\$f\s+x\s+(?P<coeffs>-?\d+(?:\s+-?\d+)*)"
    r"\s+\$w\s+(?P<ref>-?\d+(?:\.\d+)?)\s*$"
)

# Per-species metadata extractor from <species>/coord (TURBOMOLE eht block):
# "$eht charge=0 unpaired=1"
_RE_EHT = re.compile(r"\$eht\s+charge\s*=\s*(-?\d+)\s+unpaired\s*=\s*(\d+)")


def _read_coord_meta(species_dir: Path) -> Tuple[int, int]:
    """Return (charge, 2S) parsed from ``<species>/coord``'s ``$eht`` block.

    Defaults to (0, 0) if the file is missing or the block is absent,
    closed-shell neutral is the right fallback for the molecule majority.
    """
    coord_file = species_dir / "coord"
    if not coord_file.is_file():
        return 0, 0
    text = coord_file.read_text(encoding="utf-8", errors="ignore")
    m = _RE_EHT.search(text)
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _read_struc_xyz_angstrom(species_dir: Path) -> List[Tuple[str, float, float, float]]:
    """Read ``<species>/struc.xyz`` and return list of (element, x, y, z) in ANGSTROM.

    GMTKN55 ``struc.xyz`` follows the standard .xyz convention (angstroms); the
    bohr copy lives in the sibling TURBOMOLE ``coord`` file. The coordinates are
    handed to PySCF (whose default unit is angstrom) verbatim, no conversion.
    """
    xyz_file = species_dir / "struc.xyz"
    if not xyz_file.is_file():
        raise FileNotFoundError(
            f"struc.xyz missing for species at {species_dir}"
        )
    lines = xyz_file.read_text(encoding="utf-8").splitlines()
    n_atoms = int(lines[0].strip())
    out: List[Tuple[str, float, float, float]] = []
    # struc.xyz: line 0 = N, line 1 = comment (blank), lines 2..2+N = atoms.
    for line in lines[2: 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        out.append((elem, x, y, z))
    if len(out) != n_atoms:
        raise ValueError(
            f"struc.xyz at {xyz_file} declared {n_atoms} atoms but parsed "
            f"{len(out)}"
        )
    return out


def _atoms_to_pyscf_str(atoms_ang: Sequence[Tuple[str, float, float, float]]) -> str:
    """Format (element, x, y, z) angstrom tuples as a PySCF ``atom`` string in
    ANGSTROMS (the PySCF default unit). Coordinates are passed through verbatim
: struc.xyz is already angstrom (see ``_read_struc_xyz_angstrom``).

    Returns ``'H 0.000000 0.000000 0.000000; O ...'``.
    """
    parts = []
    for elem, x, y, z in atoms_ang:
        parts.append(f"{elem} {x:.10f} {y:.10f} {z:.10f}")
    return "; ".join(parts)


def _atom_composition(
    atoms: Sequence[Tuple[str, float, float, float]],
) -> Tuple[Tuple[str, int], ...]:
    """Return Hill-ordered composition tuple ((element, count), ...).

    MoleculeSpec uses a tuple-of-pairs composition for hashability.
    """
    counts: Dict[str, int] = {}
    for elem, _, _, _ in atoms:
        counts[elem] = counts.get(elem, 0) + 1
    return tuple(sorted(counts.items()))


def _build_species_dict(
    species_name: str, base_dir: Path,
) -> Dict[str, Any]:
    """Build a JSON-serializable dict for one species.

    Output keys: name, atom (PySCF string, Angstroms), atom_composition,
    charge, spin (= 2S, the number of unpaired electrons).
    """
    species_dir = base_dir / species_name
    if not species_dir.is_dir():
        raise FileNotFoundError(
            f"species directory missing: {species_dir}"
        )
    atoms_ang = _read_struc_xyz_angstrom(species_dir)
    charge, spin = _read_coord_meta(species_dir)
    return {
        "name": species_name,
        "atom": _atoms_to_pyscf_str(atoms_ang),
        "atom_composition": list(_atom_composition(atoms_ang)),
        "charge": charge,
        "spin": spin,
    }


# ---------------------------------------------------------------------------
# Reaction parsers
# ---------------------------------------------------------------------------

def _parse_bh76_res(
    res_path: Path,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Parse BH76/.res into (species_names, reactions).

    Each reaction:
      * The line specifies a sequence of "species/$f" tokens BEFORE the 'x'
        marker, then integer coefficients matching the species order. The
        sign of each coefficient identifies reactant (negative) vs product
        (positive). E.g. ``h n2o n2ohts -1 -1 +1`` means
        ``-E(h) -E(n2o) +E(n2ohts) = 17.7 kcal/mol`` (forward barrier).
    """
    text = res_path.read_text(encoding="utf-8")
    reactions: List[Dict[str, Any]] = []
    seen_species: Dict[str, None] = {}
    rxn_idx = 0
    for line_raw in text.splitlines():
        line = line_raw.strip()
        m = _RE_BH76_LINE.match(line)
        if not m:
            continue
        spec_tokens = m.group("spec").split()
        # Each token is "<species>/$f"; strip the /$f suffix.
        species_in_order: List[str] = []
        for tok in spec_tokens:
            sp = tok.split("/")[0]
            species_in_order.append(sp)
            seen_species.setdefault(sp, None)
        coeffs_raw = [int(c) for c in m.group("coeffs").split()]
        if len(coeffs_raw) != len(species_in_order):
            raise ValueError(
                f"BH76 line {rxn_idx}: {len(species_in_order)} species but "
                f"{len(coeffs_raw)} coefficients in {line!r}"
            )
        ref = float(m.group("ref"))
        reactants = [sp for sp, c in zip(species_in_order, coeffs_raw) if c < 0]
        products = [sp for sp, c in zip(species_in_order, coeffs_raw) if c > 0]
        # coeffs ordered as reactants-then-products to match PROBE_C convention.
        coeffs = [float(c) for c in coeffs_raw if c < 0] + \
                 [float(c) for c in coeffs_raw if c > 0]
        reactants_ordered = [sp for sp, c in zip(species_in_order, coeffs_raw) if c < 0]
        products_ordered = [sp for sp, c in zip(species_in_order, coeffs_raw) if c > 0]
        name = "bh76_" + "_".join(reactants_ordered + ["to"] + products_ordered)
        reactions.append({
            "name": name,
            "source_pool": "bh76",
            "reactants": reactants_ordered,
            "products":  products_ordered,
            "coeffs":    coeffs,
            "reaction_energy_ref": ref,
            "source": (
                "GMTKN55-BH76 forward barrier heights (Goerigk, Hansen, Bauer, "
                "Ehrlich, Najibi, Grimme, PCCP 19 32184 (2017); "
                "scripts/script_data/gmtkn55/BH76/.res)"
            ),
        })
        rxn_idx += 1
        # consume products too
        for sp in products:
            seen_species.setdefault(sp, None)
    return list(seen_species.keys()), reactions


def _parse_w411_res(
    res_path: Path,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Parse W4-11/.res into (species_names, reactions).

    Each W4-11 line is an atomization energy decomposition::

        $tmer {h2,h}/$f  x -1 2 $w 109.493

    The braced list is ``{target_molecule, atom_type_1, atom_type_2, ...}``
    in species-table order. Coefficients are signed: -1 for the molecule,
    +n_i for each atom type (count of that atom in the molecule). So the
    atomization energy ``2·E(h) - E(h2) = 109.493 kcal/mol``.
    """
    text = res_path.read_text(encoding="utf-8")
    reactions: List[Dict[str, Any]] = []
    seen_species: Dict[str, None] = {}
    rxn_idx = 0
    for line_raw in text.splitlines():
        line = line_raw.strip()
        m = _RE_W411_LINE.match(line)
        if not m:
            continue
        species_in_order = [s.strip() for s in m.group("species").split(",")]
        for sp in species_in_order:
            seen_species.setdefault(sp, None)
        coeffs_raw = [int(c) for c in m.group("coeffs").split()]
        if len(coeffs_raw) != len(species_in_order):
            raise ValueError(
                f"W4-11 line {rxn_idx}: {len(species_in_order)} species but "
                f"{len(coeffs_raw)} coefficients in {line!r}"
            )
        ref = float(m.group("ref"))
        reactants = [sp for sp, c in zip(species_in_order, coeffs_raw) if c < 0]
        products = [sp for sp, c in zip(species_in_order, coeffs_raw) if c > 0]
        coeffs = [float(c) for c in coeffs_raw if c < 0] + \
                 [float(c) for c in coeffs_raw if c > 0]
        target_mol = reactants[0] if reactants else species_in_order[0]
        reactions.append({
            "name": f"w411_{target_mol}_atomization",
            "source_pool": "w411",
            "reactants": reactants,
            "products":  products,
            "coeffs":    coeffs,
            "reaction_energy_ref": ref,
            "source": (
                "GMTKN55-W4-11 zero-point-exclusive nonrelativistic atomization "
                "energies (Karton, Daon, Martin, Chem. Phys. Lett. 510, 165 "
                "(2011); scripts/script_data/gmtkn55/W4-11/.res)"
            ),
        })
        rxn_idx += 1
    return list(seen_species.keys()), reactions


# ---------------------------------------------------------------------------
# Build-time helpers (called by scripts/rebuild_full_benchmark_pools.py)
# ---------------------------------------------------------------------------

def build_bh76_pool_dict() -> Dict[str, Any]:
    """Parse the BH76 source and return a JSON-serializable dict.

    Output schema::

        {"species": [...species dicts...], "reactions": [...rxn dicts...]}

    The species dicts cover every species name referenced across the 76
    reactions; the reaction dicts each carry their ``species_spins`` and
    ``species_charges`` filled from the parsed coord files for fast
    PROBE_C-schema consumption at cluster runtime.
    """
    species_names, reactions = _parse_bh76_res(BH76_SOURCE_DIR / ".res")
    species_dicts: List[Dict[str, Any]] = []
    spin_lookup: Dict[str, int] = {}
    charge_lookup: Dict[str, int] = {}
    for sp_name in species_names:
        sd = _build_species_dict(sp_name, BH76_SOURCE_DIR)
        species_dicts.append(sd)
        spin_lookup[sp_name] = sd["spin"]
        charge_lookup[sp_name] = sd["charge"]
    for r in reactions:
        r["species_spins"] = {sp: spin_lookup.get(sp, 0)
                              for sp in r["reactants"] + r["products"]}
        r["species_charges"] = {sp: charge_lookup.get(sp, 0)
                                 for sp in r["reactants"] + r["products"]}
    return {"species": species_dicts, "reactions": reactions}


def build_w411_pool_dict() -> Dict[str, Any]:
    """Parse the W4-11 source and return a JSON-serializable dict.

    Same schema as :func:`build_bh76_pool_dict`.
    """
    species_names, reactions = _parse_w411_res(W411_SOURCE_DIR / ".res")
    species_dicts: List[Dict[str, Any]] = []
    spin_lookup: Dict[str, int] = {}
    charge_lookup: Dict[str, int] = {}
    for sp_name in species_names:
        sd = _build_species_dict(sp_name, W411_SOURCE_DIR)
        species_dicts.append(sd)
        spin_lookup[sp_name] = sd["spin"]
        charge_lookup[sp_name] = sd["charge"]
    for r in reactions:
        r["species_spins"] = {sp: spin_lookup.get(sp, 0)
                              for sp in r["reactants"] + r["products"]}
        r["species_charges"] = {sp: charge_lookup.get(sp, 0)
                                 for sp in r["reactants"] + r["products"]}
    return {"species": species_dicts, "reactions": reactions}


# ---------------------------------------------------------------------------
# Runtime loaders, what the cluster eval task imports
# ---------------------------------------------------------------------------

def _resolve_refs_dir(refs_dir: str | os.PathLike | None) -> str | None:
    """Benchmark CCSD reference dir resolution: explicit argument first, then
    the ``XCQUINOX_BENCH_REFS_DIR`` environment variable (how the cluster eval
    task and the parallel shard workers pick it up without a config-schema
    change -- the env propagates into worker subprocesses), else None
    (historical behavior: no external density references)."""
    if refs_dir is not None:
        return str(refs_dir)
    return os.environ.get("XCQUINOX_BENCH_REFS_DIR") or None


def _dict_to_mol_spec(
    sd: Dict[str, Any],
    basis: str,
    grid_level: int | None,
    refs_dir: str | None = None,
) -> MoleculeSpec:
    """Build a MoleculeSpec from a JSON-cached species dict.

    When ``refs_dir`` holds ``<name>.npz`` (a density-only benchmark CCSD
    reference from ``xcquinox.alec.benchmark_refs``), it is wired via
    ``external_data_path`` so the precompute loads ``rho_ref_grid`` and the
    held-out eval can report density-vs-CCSD errors (pattern:
    ``cluster.spec_builder.atoms_to_mol_spec``). Absent file -> None, the
    historical no-reference behavior."""
    ext: str | None = None
    if refs_dir:
        cand = os.path.join(refs_dir, f"{sd['name']}.npz")
        ext = cand if os.path.isfile(cand) else None
    return MoleculeSpec(
        name=sd["name"],
        atom=sd["atom"],
        basis=basis,
        charge=int(sd["charge"]),
        spin=int(sd["spin"]),
        atom_composition=tuple(
            (str(e), int(n)) for e, n in sd["atom_composition"]
        ),
        external_data_path=ext,
        grid_level=grid_level,
    )


def _load_pool_from_json(
    json_path: Path,
    builder,
    basis: str,
    grid_level: int | None,
    refs_dir: str | None = None,
) -> Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]]:
    """Load a pool from the JSON cache, rebuilding on the fly if missing
    or if XCQUINOX_REBUILD_FULL_POOLS=1."""
    rebuild = os.environ.get("XCQUINOX_REBUILD_FULL_POOLS") == "1"
    if not rebuild and json_path.is_file():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = builder()
        if rebuild:
            # Caller can persist via the regen script; do not write here.
            pass
    mol_specs: Dict[str, MoleculeSpec] = {}
    for sd in data["species"]:
        mol_specs[sd["name"]] = _dict_to_mol_spec(sd, basis, grid_level,
                                                  refs_dir)
    return mol_specs, list(data["reactions"])


def load_full_bh76(
    basis: str = "def2-svp",
    grid_level: int | None = 1,
    refs_dir: str | os.PathLike | None = None,
) -> Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]]:
    """Return ``({species_name: MoleculeSpec}, [reaction_dict, ...])`` for the
    full GMTKN55-BH76 set (76 reactions, ~50 species).

    Pulls from the JSON cache when present (the cluster path). Falls back
    to live ``.res`` parsing when the cache is missing or
    ``XCQUINOX_REBUILD_FULL_POOLS=1`` is set in the environment.

    Args:
        basis: PySCF basis name to bake into every returned MoleculeSpec.
            Default ``def2-svp`` matches the existing cluster sweep.
        grid_level: PySCF DFT grid level for every MoleculeSpec. Default 1.
        refs_dir: directory of density-only benchmark CCSD reference
            ``<name>.npz`` files (``xcquinox.alec.benchmark_refs``); falls
            back to ``$XCQUINOX_BENCH_REFS_DIR``, else no references
            (see :func:`_resolve_refs_dir`).
    """
    global _BH76_CACHE
    resolved_refs = _resolve_refs_dir(refs_dir)
    key = (basis, grid_level, resolved_refs)
    if _BH76_CACHE is not None and _BH76_CACHE[0] == key:  # pyright: ignore
        return _BH76_CACHE[1]
    mol_specs, reactions = _load_pool_from_json(
        BH76_JSON_PATH, build_bh76_pool_dict, basis, grid_level,
        resolved_refs,
    )
    _BH76_CACHE = (key, (mol_specs, reactions))  # type: ignore[assignment]
    return mol_specs, reactions


def load_full_w411(
    basis: str = "def2-svp",
    grid_level: int | None = 1,
    refs_dir: str | os.PathLike | None = None,
) -> Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]]:
    """Return ``({species_name: MoleculeSpec}, [reaction_dict, ...])`` for the
    full GMTKN55-W4-11 set (140 atomization reactions, ~150 species).

    Same caching + fallback + ``refs_dir`` semantics as
    :func:`load_full_bh76`.
    """
    global _W411_CACHE
    resolved_refs = _resolve_refs_dir(refs_dir)
    key = (basis, grid_level, resolved_refs)
    if _W411_CACHE is not None and _W411_CACHE[0] == key:  # pyright: ignore
        return _W411_CACHE[1]
    mol_specs, reactions = _load_pool_from_json(
        W411_JSON_PATH, build_w411_pool_dict, basis, grid_level,
        resolved_refs,
    )
    _W411_CACHE = (key, (mol_specs, reactions))  # type: ignore[assignment]
    return mol_specs, reactions


def load_full_held_out_pools(
    basis: str = "def2-svp",
    grid_level: int | None = 1,
    refs_dir: str | os.PathLike | None = None,
) -> Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]]:
    """Convenience: union of BH76 + W4-11.

    Species dicts merge by name (e.g. ``h``, ``c``, ``o``, ``n``, ``f`` appear
    in both sets, same MoleculeSpec for both). Reactions concatenate (BH76
    first, then W4-11). Total: 76 + 140 = 216 reactions over 214 unique
    species (79 BH76 + 152 W4-11, 17 overlap). ``refs_dir`` semantics as
    :func:`load_full_bh76`.
    """
    bh76_mols, bh76_rxns = load_full_bh76(basis=basis, grid_level=grid_level,
                                          refs_dir=refs_dir)
    w411_mols, w411_rxns = load_full_w411(basis=basis, grid_level=grid_level,
                                          refs_dir=refs_dir)
    merged_mols: Dict[str, MoleculeSpec] = dict(bh76_mols)
    for sp_name, ms in w411_mols.items():
        if sp_name in merged_mols:
            # When the same species appears in both sets the geometries may
            # differ marginally (different GMTKN55 source dirs). Keep the
            # BH76 version, it covers the barrier-height species set which
            # matters most for the eval comparison. Document the conflict
            # at debug-print level for the operator.
            continue
        merged_mols[sp_name] = ms
    return merged_mols, list(bh76_rxns) + list(w411_rxns)

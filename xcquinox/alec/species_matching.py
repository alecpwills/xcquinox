"""Composition-level identity between training names and pool names.

Training molecules carry ASE Hill-formula names from the DFS pool builder
(``dfs_pool``: ``HO``, ``CHN``, ``H3N``, ``CH2``, ``C+``); the benchmark
pools name the same physical species in GMTKN55 style (``oh``, ``hcn``,
``nh3``, ``ch2-trip``). A name-equality membership test -- even case-folded
-- cannot connect the two vocabularies, so a strict held-out filter built
on names alone keeps trained molecules' reactions in the "held-out" set.

The identity used here is ``(element composition, charge, spin)``:

- pool side: taken from each ``MoleculeSpec`` (``atom_composition``,
  ``charge``, ``spin``) -- no name parsing of common names ever happens;
- training side: the Hill name is parsed (it was generated as a formula, so
  parsing is exact), the charge from its ``+``/``-`` suffix, and the spin
  from the DFS ground-state tables (``DFS_AE_SPIN``,
  ``ATOMIC_GROUND_STATE_SPIN``). A training name with no tabulated spin
  matches any pool spin (conservative: over-matching can only shrink a
  held-out set, never leak into it).

``trained_pool_aliases`` returns the pool names identical to some training
molecule; callers add them to the name-based membership set, which keeps
every existing filter signature unchanged.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional, Set, Tuple

# Standard element symbols (through Og); the tokenizer validates every token
# against this set so a non-formula name fails loudly instead of parsing as
# nonsense ("methanol" is not Me-Th-An-O-L).
_ELEMENT_SYMBOLS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I
Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt
Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

Composition = Tuple[Tuple[str, int], ...]


def parse_formula_name(name: str) -> Optional[Tuple[Composition, int]]:
    """``(composition, charge)`` for a Hill-formula name, else ``None``.

    Accepts exactly ``<element tokens><one optional trailing + or ->``
    (``H3N`` -> ((('H', 3), ('N', 1)), 0); ``C+`` -> ((('C', 1),), 1)).
    Any residue, unknown symbol, or empty parse returns ``None`` -- pool
    common names (``methanol``, ``ch2-trip``, ``RKT01``) are intentionally
    unparseable here; their identity comes from the pool specs instead.
    """
    if not name:
        return None
    charge = 0
    body = str(name)
    if body.endswith("+"):
        charge, body = 1, body[:-1]
    elif body.endswith("-"):
        charge, body = -1, body[:-1]
    if not body:
        return None
    counts: Dict[str, int] = {}
    pos = 0
    for m in _TOKEN_RE.finditer(body):
        if m.start() != pos:
            return None
        sym, digits = m.group(1), m.group(2)
        if sym not in _ELEMENT_SYMBOLS:
            return None
        counts[sym] = counts.get(sym, 0) + (int(digits) if digits else 1)
        pos = m.end()
    if pos != len(body) or not counts:
        return None
    return tuple(sorted(counts.items())), charge


def is_atomic(composition: Composition) -> bool:
    """True for a single atom of a single element (any charge)."""
    return len(composition) == 1 and composition[0][1] == 1


def trained_species_key(name: str
                        ) -> Optional[Tuple[Composition, int, Optional[int]]]:
    """``(composition, charge, spin)`` for one training-side name.

    Spin (2S, PySCF convention) comes from the DFS ground-state tables the
    pool builder itself used, so a training ``CH2`` (built triplet) matches
    the pool's ``ch2-trip`` and not ``ch2-sing``; names absent from both
    tables get ``spin=None`` (matches any pool spin). ``None`` when the name
    is not a formula.
    """
    parsed = parse_formula_name(name)
    if parsed is None:
        return None
    comp, charge = parsed
    # dfs_pool pulls ASE at module level; imported lazily so this module
    # stays cheap for figure-layer callers that never touch geometries.
    from xcquinox.alec.dfs_pool import (ATOMIC_GROUND_STATE_SPIN,
                                        DFS_AE_SPIN)
    spin: Optional[int] = None
    if name in DFS_AE_SPIN:
        spin = int(DFS_AE_SPIN[name])
    elif name in ATOMIC_GROUND_STATE_SPIN:
        spin = int(ATOMIC_GROUND_STATE_SPIN[name])
    return comp, charge, spin


def pool_species_key(spec: Any) -> Optional[Tuple[Composition, int, int]]:
    """``(composition, charge, spin)`` for one pool ``MoleculeSpec`` (or a
    mapping with the same fields). ``None`` when composition is missing."""
    get = (spec.get if isinstance(spec, Mapping)
           else lambda k, d=None: getattr(spec, k, d))
    comp = get("atom_composition")
    if not comp:
        return None
    comp_t = tuple(sorted((str(e), int(n)) for e, n in comp))
    return comp_t, int(get("charge", 0) or 0), int(get("spin", 0) or 0)


def _parse_atom_string(atom: str) -> Optional[Tuple[Tuple[str, ...],
                                                    Tuple[Tuple[float, ...],
                                                          ...]]]:
    """``(symbols, positions)`` from a PySCF-style ``atom`` string
    (``"C x y z; N x y z; ..."``); ``None`` on any malformed entry."""
    syms = []
    pos = []
    for chunk in str(atom or "").split(";"):
        parts = chunk.split()
        if not parts:
            continue
        if len(parts) != 4:
            return None
        try:
            pos.append(tuple(float(x) for x in parts[1:]))
        except ValueError:
            return None
        syms.append(parts[0])
    if not syms:
        return None
    return tuple(syms), tuple(pos)


def _distance_signature(symbols, positions
                        ) -> Dict[Tuple[str, str], Tuple[float, ...]]:
    """Sorted interatomic distances per unordered element pair -- a
    rotation/translation/permutation-invariant connectivity fingerprint.
    Isomers (hcn vs hnc) differ by ~1 Angstrom in their H-N / H-C legs, far
    beyond cross-source geometry scatter for these small rigid species."""
    out: Dict[Tuple[str, str], list] = {}
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            key = tuple(sorted((str(symbols[i]), str(symbols[j]))))
            d = sum((positions[i][k] - positions[j][k]) ** 2
                    for k in range(3)) ** 0.5
            out.setdefault(key, []).append(d)
    return {k: tuple(sorted(v)) for k, v in out.items()}


def geometries_match(sig_a: Mapping[Tuple[str, str], Tuple[float, ...]],
                     sig_b: Mapping[Tuple[str, str], Tuple[float, ...]],
                     tol: float = 0.3) -> bool:
    """True when the two distance signatures agree pairwise within ``tol``
    Angstrom (same pair types, same counts, every sorted distance close)."""
    if set(sig_a) != set(sig_b):
        return False
    for k, da in sig_a.items():
        db = sig_b[k]
        if len(da) != len(db):
            return False
        if any(abs(x - y) > tol for x, y in zip(da, db)):
            return False
    return True


_TRAINED_GEOMETRY_CACHE: Optional[Dict[str, Any]] = None


def _dfs_trained_geometry(name: str):
    """``(symbols, positions)`` of the trained molecule ``name`` from the
    DFS pool's own geometry source (the G2/97 trajectory ``dfs_pool``
    builds from, same last-wins keying), or ``None`` when unavailable.
    Cached; the trajectory is read once per process."""
    global _TRAINED_GEOMETRY_CACHE
    if _TRAINED_GEOMETRY_CACHE is None:
        try:
            from ase.io import read
            from xcquinox.alec.dfs_pool import _g297_traj_path
            traj = read(str(_g297_traj_path()), ":")
            _TRAINED_GEOMETRY_CACHE = {
                a.get_chemical_formula(): (
                    tuple(a.get_chemical_symbols()),
                    tuple(tuple(float(x) for x in p)
                          for p in a.get_positions()))
                for a in traj}
        except Exception as exc:
            print(f"  (species aliasing: trained geometries unavailable "
                  f"({exc}); composition-ambiguous matches kept "
                  "conservatively)")
            _TRAINED_GEOMETRY_CACHE = {}
    return _TRAINED_GEOMETRY_CACHE.get(name)


def trained_pool_aliases(training_names: Iterable[str],
                         pool_specs: Mapping[str, Any], *,
                         verbose: bool = True,
                         _geometry_provider=None) -> Set[str]:
    """Pool species names physically identical to some training molecule.

    A pool species aliases a training name when compositions and charges are
    equal and the spins are equal (or the training spin is untabulated).
    Exact/case-only name matches are already handled by the name-based
    filters and are not repeated here; only differently-named twins are
    returned.

    A training name matching SEVERAL pool species (isomers share
    composition+charge+spin: hcn vs hnc) is resolved by geometry -- the
    trained molecule's own distance signature against each candidate's --
    keeping only the geometric matches. When no geometry is available for
    the trained name, all candidates are kept (conservative: over-matching
    only shrinks a held-out set, never leaks into it) and reported.
    ``_geometry_provider`` is a test seam (name -> (symbols, positions) or
    None); the default reads the DFS pool's G2/97 trajectory."""
    provider = (_geometry_provider if _geometry_provider is not None
                else _dfs_trained_geometry)
    keys: Dict[str, Tuple[Composition, int, Optional[int]]] = {}
    trained_names_cf = {str(n).casefold() for n in training_names}
    for n in training_names:
        k = trained_species_key(str(n))
        if k is not None:
            keys[str(n)] = k
    # Ambiguity is assessed over ALL composition matches, INCLUDING pool
    # names the name-based filters already see: removing the name-visible
    # candidate first would leave its composition-degenerate isomer as a
    # lone "unambiguous" match that skips the geometry check (trained
    # acetylene C2H2 -> pool {c2h2, ch2c}: dropping c2h2 early wrongly
    # aliases vinylidene). Name-visible matches drop only from the RETURNED
    # set, at the end.
    matches_by_trained: Dict[str, Set[str]] = {}
    for pool_name, spec in pool_specs.items():
        pk = pool_species_key(spec)
        if pk is None:
            continue
        p_comp, p_charge, p_spin = pk
        for t_name, (t_comp, t_charge, t_spin) in keys.items():
            if (t_comp == p_comp and t_charge == p_charge
                    and (t_spin is None or t_spin == p_spin)):
                matches_by_trained.setdefault(t_name, set()).add(
                    str(pool_name))
    aliases: Set[str] = set()
    unresolved: Dict[str, list] = {}
    for t_name, pool_names in matches_by_trained.items():
        if len(pool_names) == 1:
            aliases.update(pool_names)
            continue
        geo = provider(t_name)
        resolved = set()
        if geo is not None:
            t_sig = _distance_signature(*geo)
            for pn in pool_names:
                get = (pool_specs[pn].get
                       if isinstance(pool_specs[pn], Mapping)
                       else lambda k, d=None, s=pool_specs[pn]: getattr(
                           s, k, d))
                parsed = _parse_atom_string(get("atom"))
                if parsed is not None and geometries_match(
                        t_sig, _distance_signature(*parsed)):
                    resolved.add(pn)
        if resolved:
            aliases.update(resolved)
        else:
            # no geometry, or none matched: keep all candidates rather than
            # risk a trained twin staying "held-out"
            aliases.update(pool_names)
            unresolved[t_name] = sorted(pool_names)
    if verbose and unresolved:
        print("  (species aliasing: composition-ambiguous training names "
              f"kept conservatively (all candidates excluded): {unresolved})")
    return {a for a in aliases if a.casefold() not in trained_names_cf}

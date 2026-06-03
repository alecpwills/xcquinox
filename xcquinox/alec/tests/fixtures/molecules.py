"""Reusable MoleculeSpec factory functions for test fixtures."""
from xcquinox.alec.config import MoleculeSpec


def h_atom() -> MoleculeSpec:
    """Single hydrogen atom (spin=1, open-shell, exercises the UKS branch)."""
    return MoleculeSpec(
        name="H", atom="H 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("H", 1),),
    )


def h2_molecule() -> MoleculeSpec:
    """H2 (spin=0, closed-shell, smallest RKS diatomic)."""
    return MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
    )


def o_atom() -> MoleculeSpec:
    """Oxygen atom (spin=2, open-shell, UKS path with a 2-channel DM)."""
    return MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),),
    )


def h2o_molecule() -> MoleculeSpec:
    """Water molecule (spin=0, closed-shell, the canonical notebook subject)."""
    return MoleculeSpec(
        name="H2O", atom="O 0 0 0; H 0 0 0.96; H 0.96 0 0", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2), ("O", 1)),
    )


def c2h2_molecule(basis: str = "def2-svp") -> MoleculeSpec:
    """Acetylene (linear D∞h symmetry, exercises the near-degenerate π MO
    regression path in oneshot_dm_prediction_fast. Without symmetry-breaking
    in the Fock matrix before eigh, the reverse-mode gradient of eigh hits
    1/(λ_i - λ_j) at the degenerate π_x / π_y pair and returns NaN.

    Default basis is def2-svp (matches the step6 case-study notebook where
    the bug was originally observed). sto-3g is too small to reliably
    surface the numerical tie."""
    return MoleculeSpec(
        name="C2H2",
        atom=(
            "H 0 0 1.666650; "
            "C 0 0 0.603250; "
            "C 0 0 -0.603250; "
            "H 0 0 -1.666650"
        ),
        basis=basis, charge=0, spin=0,
        atom_composition=(("C", 2), ("H", 2)),
    )

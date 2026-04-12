"""Reusable MoleculeSpec factory functions for test fixtures."""
from xcquinox.alec.config import MoleculeSpec


def h_atom() -> MoleculeSpec:
    """Single hydrogen atom (spin=1, open-shell — exercises the UKS branch)."""
    return MoleculeSpec(
        name="H", atom="H 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("H", 1),),
    )


def h2_molecule() -> MoleculeSpec:
    """H2 (spin=0, closed-shell — smallest RKS diatomic)."""
    return MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
    )


def o_atom() -> MoleculeSpec:
    """Oxygen atom (spin=2, open-shell — UKS path with a 2-channel DM)."""
    return MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),),
    )


def h2o_molecule() -> MoleculeSpec:
    """Water molecule (spin=0, closed-shell — the canonical notebook subject)."""
    return MoleculeSpec(
        name="H2O", atom="O 0 0 0; H 0 0 0.96; H 0.96 0 0", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2), ("O", 1)),
    )

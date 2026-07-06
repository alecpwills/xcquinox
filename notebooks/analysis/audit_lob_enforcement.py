"""Audit Lieb-Oxford bound enforcement across every trained checkpoint.

For each ``model.eqx`` under ``notebooks/checkpoints_step6/<run>/group*/``,
samples F_x on the CH4 PBE grid and reports:

  - max(F_x): if > 1.804, the Lieb-Oxford bound (Lieb & Oxford,
              IJQC 19, 427 (1981); PBE convention 1+kappa = 1.804 per
              Perdew/Burke/Ernzerhof PRL 77, 3865 (1996) eq. 14)
              has been violated -- a real architectural bug.
  - min(F_x): if it reaches 0, the network has hit the sigmoid floor
              of the _AlecLOB clamp; F_x = 0 means zero exchange at
              that grid point (unphysical for real systems but allowed
              by the current clamp).
  - asymptotic F_x at s > 5: how close the trained NN's F_x asymptote
              is to PBE's 1+kappa value. The LOB only requires
              F_x <= 1.804 globally; it does NOT pin F_x(s -> inf) to
              that value. Specs that drift below ~ 1.5 at large s have
              learned a softer F_x asymptote than PBE.

Run: ``python notebooks/analysis/audit_lob_enforcement.py``
Optional: ``--run-dir notebooks/checkpoints_step6/integration``
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
from pathlib import Path

# Force x64 + CPU before importing jax.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import xcquinox.alec as alec  # noqa: E402
from xcquinox.alec.descriptors import assemble_descriptor_features  # noqa: E402
from xcquinox.alec.models import AlecGGAModel  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
BASIS, GRID_LEVEL = "def2-svp", 1

# Lieb-Oxford constant.
# Lieb & Oxford 1981 IJQC 19, 427: E_x[rho] >= -C_LO * integral(rho^4/3).
# PBE chooses kappa = 0.804 so F_x <= 1 + kappa = 1.804 globally.
LOB_CEILING = 1.804


def pbe_fx_analytic(s: np.ndarray) -> np.ndarray:
    """PBE-1996 analytic F_x(s) -- eq. (14) of Perdew/Burke/Ernzerhof PRL 77, 3865.

    F_x(s) = 1 + kappa - kappa / (1 + mu * s^2 / kappa)
    """
    kappa, mu = 0.804, 0.21951
    return 1.0 + kappa - kappa / (1.0 + mu * s**2 / kappa)


def build_ch4_grid(arch_name: str):
    """Build CH4 (rho, sigma, features, s) on the PBE grid for one arch.

    Cached by arch -- the descriptor features depend on the arch's
    descriptor list, but the grid coords/density themselves don't.
    """
    arch_cfg = alec.get_architecture(arch_name)
    skel = AlecGGAModel.from_arch(arch_cfg, seed=0)
    required = tuple({k for d in skel.descriptors for k in d.required_mol_keys})
    mol_spec = alec.MoleculeSpec(
        name="CH4",
        atom=(
            "C 0 0 0; "
            "H  0.628099  0.628099  0.628099; "
            "H -0.628099 -0.628099  0.628099; "
            "H  0.628099 -0.628099 -0.628099; "
            "H -0.628099  0.628099 -0.628099"
        ),
        basis=BASIS, charge=0, spin=0, grid_level=GRID_LEVEL,
        atom_composition=(("C", 1), ("H", 4)),
    )
    mol_data = alec.precompute_fixed_density_data(
        mol_spec, required_keys=required, descriptors=skel.descriptors,
    )
    rho = np.asarray(mol_data["rho_grid"])
    sigma = np.asarray(mol_data["sigma_grid"])
    features = np.asarray(assemble_descriptor_features(skel.descriptors, mol_data))
    kF = (3.0 * np.pi**2) ** (1.0 / 3.0)
    s = np.sqrt(np.clip(sigma, 0.0, None)) / (
        2.0 * kF * np.clip(rho, 1e-12, None) ** (4.0 / 3.0)
    )
    return rho, sigma, features, s


def fx_for_checkpoint(ckpt_path: str, rho, sigma, features, arch_name: str):
    """Load one model.eqx and return F_x on every grid point as a numpy array."""
    arch_cfg = alec.get_architecture(arch_name)
    model = AlecGGAModel.from_arch(arch_cfg, seed=0)
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    rho_j, sigma_j, feat_j = jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(features)

    def _fx_one(r, sg, f):
        return model.xnet(jnp.concatenate(
            [jnp.atleast_1d(r), jnp.atleast_1d(sg), f],
        ))

    return np.asarray(jax.vmap(_fx_one)(rho_j, sigma_j, feat_j))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir", type=Path,
        default=REPO / "notebooks" / "checkpoints_step6" / "unweighted",
    )
    args = ap.parse_args()

    caches = {a: build_ch4_grid(a)
              for a in ("deep_combined", "deep_combined_attn")}

    rows = []
    ckpts = sorted(glob.glob(str(args.run_dir / "group*" / "*" / "L*" / "*" / "model.eqx")))
    print(f"scanning {len(ckpts)} checkpoints under {args.run_dir} ...")
    for ckpt in ckpts:
        parts = ckpt.split(os.sep)
        group, arch_name, loss, solver = parts[-5], parts[-4], parts[-3], parts[-2]
        rho, sigma, feat, s = caches[arch_name]
        fx = fx_for_checkpoint(ckpt, rho, sigma, feat, arch_name)
        high_s = s > 5.0
        rows.append({
            "group": group, "arch": arch_name, "loss": loss, "solver": solver,
            "fx_max": float(fx.max()), "fx_min": float(fx.min()),
            "fx_at_s5plus": float(fx[high_s].mean()) if high_s.any() else float("nan"),
        })

    arr_max = np.array([r["fx_max"] for r in rows])
    arr_min = np.array([r["fx_min"] for r in rows])
    n_violate = int((arr_max > LOB_CEILING + 1e-4).sum())
    n_saturate = int((arr_max > LOB_CEILING - 1e-3).sum())

    print()
    print("=" * 70)
    print(f"LIEB-OXFORD BOUND ENFORCEMENT -- {args.run_dir.name} pretrain-origin")
    print("=" * 70)
    print(f"  Specs scanned:              {len(rows)}")
    print(f"  max(F_x) global:            {arr_max.max():.6f}  (LOB ceiling = {LOB_CEILING})")
    print(f"  min(F_x) global:            {arr_min.min():.6f}  (LOB floor at sigmoid = 0)")
    print(f"  specs violating F_x > 1.804: {n_violate} / {len(rows)}")
    print(f"  specs saturating LOB ceiling: {n_saturate} / {len(rows)}")
    print()
    print("  Lieb-Oxford bound (Lieb & Oxford IJQC 19, 427, 1981):")
    print("    E_x[rho] >= -C_LO * integral(rho^(4/3) dr)")
    print("  PBE convention (Perdew/Burke/Ernzerhof PRL 77, 3865, 1996, eq. 14):")
    print("    kappa = 0.804  =>  F_x(s) <= 1 + kappa = 1.804")
    print()
    if n_violate == 0:
        print(f"  ✓ LOB IS ENFORCED.  All {len(rows)} specs respect F_x <= 1.804.")
    else:
        print(f"  ✗ LOB VIOLATED in {n_violate} specs -- investigate the _AlecLOB clamp.")

    arr_asym = np.array([r["fx_at_s5plus"] for r in rows])
    arr_asym = arr_asym[~np.isnan(arr_asym)]
    pbe_asym = float(pbe_fx_analytic(np.array([5.0, 10.0, 15.0])).mean())
    print()
    print("  ASYMPTOTIC F_x at s > 5 on CH4 grid:")
    print(f"    PBE analytic asymptote at s=5..15: {pbe_asym:.4f}  (-> 1.804 as s -> inf)")
    print(f"    NN mean: {arr_asym.mean():.4f}    range [{arr_asym.min():.4f}, {arr_asym.max():.4f}]")
    print()
    print("  By loss strategy (asymptotic mean F_x at s > 5):")
    by_loss = collections.defaultdict(list)
    for r in rows:
        if not np.isnan(r["fx_at_s5plus"]):
            by_loss[r["loss"]].append(r["fx_at_s5plus"])
    for loss in sorted(by_loss):
        a = np.array(by_loss[loss])
        print(f"    {loss:<26}: mean {a.mean():.4f}, range [{a.min():.4f}, {a.max():.4f}]")
    print()
    print("  INTERPRETATION:")
    print("    The Lieb-Oxford bound is an UPPER bound; it does NOT require")
    print("    F_x(s -> inf) = 1.804. PBE's choice F_x(s -> inf) = 1+kappa is")
    print("    a model design that meets the bound TIGHTLY at large s. Trained")
    print("    NNs may asymptote BELOW 1.804 -- this is allowed by the LOB and")
    print("    not a violation. Specs with V_xc fitting (L3, L4) keep F_x near")
    print("    the PBE asymptote because the V_xc target encodes that shape;")
    print("    energy-only losses (L1, L2, L5) drift to lower asymptotes")
    print("    because nothing in the loss penalizes asymptote-from-PBE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

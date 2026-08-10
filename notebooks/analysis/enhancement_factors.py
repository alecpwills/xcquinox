#!/usr/bin/env python
"""Exchange / correlation enhancement-factor curves from trained checkpoints.

Mirrors SI Figs 7-10 of Navarro-Rodriguez et al. (*Constraint-aware
functional cloning*, MLXC_Constraints 2026): plot each trained network's
learned exchange enhancement F_x(s) and correlation enhancement
F_c(s, r_s; zeta=0) against the analytic PBE reference, as a function of the
reduced density gradient ``s = |grad rho| / (2 (3 pi^2)^{1/3} rho^{4/3})``.

For each architecture we load the most-trained representative checkpoint
(largest subset_size with a materialized ``model.eqx``), reusing the canonical
loader :func:`xcquinox.alec.eval_holdout.load_trained_model`
(``AlecGGAModel.from_arch(spec.arch) -> eqx.tree_deserialise_leaves``), and
forward-evaluate ``model.eval_Fx`` / ``model.eval_Fc`` on a synthetic
descriptor grid.

Definitions used (consistent with ``AlecGGAModel`` and ``pbe_anchor``):
  * F_x reference  : analytic PBE, ``pbe_anchor._fx_pbe_analytic(s)``
    (Perdew-Burke-Ernzerhof, PRL 77, 3865 (1996), eq. 14;
    kappa=0.804, mu=0.21951).
  * F_c reference  : libxc GGA_C_PBE eps_c / LDA_C_PW eps_c -- i.e. the PBE
    correlation enhancement over the same PW92 baseline the network's
    ``eval_Fc`` enhances (``_ec_baseline`` -> PW92). Same per-electron ratio,
    so the network and reference are directly comparable.

Caveats stamped on the figure:
  * Pre-``dm_entropy``-fix run (2026-05-29 forensic review).
  * **Zero-descriptor slice**: for descriptor architectures (cusp / dm /
    combined) the extra features are set to 0, so these curves are the
    F(s) slice at zero auxiliary descriptors -- a well-defined cut, not the
    full descriptor-dependent surface. ``deep``/``deep_attn``/``*notransform``
    have no extra descriptors, so the cut is exact for them.

Usage:
    python notebooks/analysis/enhancement_factors.py \
        [--run-dir <pulled run dir>] \
        [--outdir notebooks/analysis/figures_ablation_notransform]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the arch palette/order + run discovery from the sibling module.
_SIB_PATH = Path(__file__).resolve().parent / "make_ablation_arch_figure.py"
_sib_spec = importlib.util.spec_from_file_location(
    "make_ablation_arch_figure", _SIB_PATH)
sib = importlib.util.module_from_spec(_sib_spec)  # type: ignore[arg-type]
sys.modules["make_ablation_arch_figure"] = sib
_sib_spec.loader.exec_module(sib)  # type: ignore[union-attr]

ARCH_ORDER = sib.ARCH_ORDER
ARCH_COLOR = sib.ARCH_COLOR
ccp = sib.ccp

_RS_PANELS: Tuple[float, ...] = (0.5, 2.0, 5.0)  # Wigner-Seitz radii for F_c
#: Iso-orbital alpha values the meta-GGA panel sweeps. Spans the single-orbital
#: corner (0), the uniform-gas point (1), and the overlap/tail region a
#: diffuse basis reaches; alpha is clipped to 100 upstream (metagga.py).
_ALPHA_PANELS: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 100.0)
#: The dm_entropy label leak was fixed on 2026-05-29; runs stamped at or before
#: that date carry the caveat, later ones must NOT (the stamp used to be
#: unconditional, which printed a false provenance claim on every newer run).
_DM_ENTROPY_FIX_DATE = "20260529"


def _provenance(run_dir: Path, mgga_alpha: bool) -> str:
    """Provenance stamp for ``run_dir``: the descriptor-slice convention the
    panels actually use, plus the dm_entropy caveat only where it applies."""
    parts = []
    stamp = run_dir.name.replace("run_", "")[:8]
    if stamp.isdigit() and stamp <= _DM_ENTROPY_FIX_DATE:
        parts.append("Pre-dm_entropy-fix run (2026-05-29).")
    parts.append("Descriptor archs shown at the zero-descriptor slice "
                 "(extras=0)")
    parts.append("except the meta-GGA archs, whose alpha column is swept in "
                 "the alpha panel (their extras=0 curve IS the alpha=0 slice)."
                 if mgga_alpha else ".")
    return " ".join(parts).replace(" .", ".")


# ---------------------------------------------------------------------------
# Descriptor-grid geometry
# ---------------------------------------------------------------------------

def s_to_sigma(rho: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Invert ``s = sqrt(sigma)/(2 k_F rho)`` -> ``sigma`` at fixed rho.
    ``k_F = (3 pi^2 rho)^{1/3}``."""
    rho = np.asarray(rho, dtype=float)
    k_F = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    return (2.0 * k_F * rho * np.asarray(s, dtype=float)) ** 2


def rs_to_rho(rs: float) -> float:
    """Wigner-Seitz radius -> uniform density ``rho = 3/(4 pi rs^3)``."""
    return 3.0 / (4.0 * np.pi * rs ** 3)


# ---------------------------------------------------------------------------
# Checkpoint selection + loading
# ---------------------------------------------------------------------------

def representative_specs(run_dir: Path) -> Dict[str, int]:
    """``{arch: spec_idx}`` -- per arch, the largest-subset trained spec
    (has ``model.eqx``). The most-trained representative per architecture."""
    cells = ccp._read_manifest_cells(run_dir)
    best: Dict[str, Tuple[int, int]] = {}  # arch -> (subset_size, idx)
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        if not (spec_dir / "model.eqx").is_file():
            continue
        cell = cells.get(idx, {})
        arch, ss = cell.get("arch"), cell.get("subset_size")
        if arch is None or ss is None:
            continue
        if arch not in best or ss > best[arch][0]:
            best[arch] = (ss, idx)
    return {a: idx for a, (_ss, idx) in best.items()}


def load_trained_model(run_dir: Path, spec_idx: int):
    """Load ``(spec, AlecGGAModel)`` for ``spec_idx`` via the canonical
    cluster loader. Heavy: imports jax/equinox/pyscf on first call."""
    import pickle  # local, trusted file produced by this codebase
    from xcquinox.alec import eval_holdout

    manifest = ccp._read_manifest_cells(run_dir)
    width = 4
    mpath = run_dir / "manifest.json"
    if mpath.is_file():
        try:
            width = int(json.loads(mpath.read_text()).get("width", 4))
        except (json.JSONDecodeError, OSError, ValueError):
            width = 4
    spec_path = run_dir / "specs" / f"spec_{spec_idx:0{width}d}.spec"
    model_path = (run_dir / "checkpoints" / f"spec_{spec_idx:0{width}d}"
                  / "model.eqx")
    with spec_path.open("rb") as f:
        spec = pickle.load(f)
    model = eval_holdout.load_trained_model(spec, model_path)
    return spec, model


# ---------------------------------------------------------------------------
# Enhancement-factor curves
# ---------------------------------------------------------------------------

def is_meta_gga(model) -> bool:
    """Whether ``model``'s X-net carries the meta-GGA alpha channel."""
    return bool(getattr(model.xnet, "meta_gga", False))


def model_fx_curve(model, s_grid: np.ndarray, rho: float = 1.0,
                   alpha: Optional[float] = None) -> np.ndarray:
    """``F_x(s)`` from a loaded model at fixed rho, zero extra descriptors.

    ``alpha`` (meta-GGA archs only) places that value in the descriptor column
    the X-net reads as the iso-orbital indicator instead of leaving it at 0.
    Left None the behaviour is the historical zero-descriptor slice.

    Why the alpha argument exists: for a meta-GGA, "zero extra descriptors"
    means alpha = 0 -- the SINGLE-ORBITAL corner, where F_x sits at its
    Lieb-Oxford ceiling. That is the least representative point of the domain,
    not a neutral cut, so a meta-GGA plotted only there says nothing about the
    functional where molecules actually sample it (alpha ~ 0-2).
    """
    import jax.numpy as jnp
    n = s_grid.shape[0]
    rho_arr = np.full(n, rho, dtype=float)
    sigma = s_to_sigma(rho_arr, s_grid)
    n_extra = int(getattr(model.xnet, "n_extra_features", 0))
    feats = np.zeros((n, n_extra), dtype=float)
    if alpha is not None and n_extra > 0:
        idx = int(getattr(model.xnet, "metagga_alpha_index", 0))
        feats[:, idx] = float(alpha)
    fx = model.eval_Fx(jnp.asarray(rho_arr), jnp.asarray(sigma),
                       jnp.asarray(feats))
    return np.asarray(fx, dtype=float)


def scan_fx_curve(s_grid: np.ndarray, alpha: float,
                  rho: float = 1.0) -> Optional[np.ndarray]:
    """SCAN exchange enhancement ``F_x(s)`` at fixed ``alpha`` from libxc.

    The meta-GGA archs pretrain to SCAN (``pretrain.py`` selects
    ``Fx_scan_all`` for ``meta_gga`` archs), so SCAN -- not PBE -- is the
    reference they should be read against. ``tau`` is solved from the requested
    alpha by inverting ``alpha = (tau - tau_W)/tau_unif``. ``None`` when libxc
    has no SCAN (older pyscf), so the panel degrades to no reference line.
    """
    try:
        from pyscf import dft
    except Exception:  # noqa: BLE001 - optional reference
        return None
    out = []
    tau_unif_c = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    e_x_lda = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0)
    for s in np.asarray(s_grid, dtype=float):
        k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
        sigma = (s * 2.0 * k_f * rho) ** 2
        tau = alpha * tau_unif_c * rho ** (5.0 / 3.0) + sigma / (8.0 * rho)
        arr = np.array([[rho], [np.sqrt(sigma)], [0.0], [0.0], [0.0], [tau]])
        try:
            e_xc = dft.libxc.eval_xc("MGGA_X_SCAN", arr, spin=0, deriv=0)[0][0]
        except Exception:  # noqa: BLE001
            return None
        out.append(float(e_xc / e_x_lda))
    return np.asarray(out, dtype=float)


def model_fc_curve(model, s_grid: np.ndarray, rs: float,
                   zeta: float = 0.0) -> np.ndarray:
    """``F_c(s; r_s, zeta)`` from a loaded model (zero extra descriptors)."""
    import jax.numpy as jnp
    n = s_grid.shape[0]
    rho = rs_to_rho(rs)
    rho_arr = np.full(n, rho, dtype=float)
    sigma = s_to_sigma(rho_arr, s_grid)
    n_extra = int(getattr(model.cnet, "n_extra_features", 0))
    feats = np.zeros((n, n_extra), dtype=float)
    fc = model.eval_Fc(jnp.asarray(rho_arr), jnp.asarray(sigma),
                       jnp.asarray(feats), zeta=zeta)
    return np.asarray(fc, dtype=float)


def pbe_fx_curve(s_grid: np.ndarray) -> np.ndarray:
    """Analytic PBE F_x(s) (reuses ``pbe_anchor._fx_pbe_analytic``)."""
    from xcquinox.alec.pbe_anchor import _fx_pbe_analytic
    return np.asarray(_fx_pbe_analytic(np.asarray(s_grid, dtype=float)),
                      dtype=float)


def pbe_fc_curve(s_grid: np.ndarray, rs: float) -> Optional[np.ndarray]:
    """PBE correlation enhancement ``eps_c^{PBE}/eps_c^{PW92}`` via libxc, at
    fixed rho(rs) over the s grid. Returns None if libxc is unavailable.

    Mirrors the libxc call convention in ``pbe_anchor._pbe_fx_libxc``: pack a
    (4, N) GGA input with ``rho_input[1] = sqrt(sigma)`` so that the libxc
    contracted gradient equals the target sigma.
    """
    try:
        from pyscf import dft as _pyscf_dft
    except ImportError:  # pragma: no cover - pyscf always present in env
        return None
    eval_xc = _pyscf_dft.libxc.eval_xc
    n = s_grid.shape[0]
    rho = rs_to_rho(rs)
    rho_arr = np.full(n, rho, dtype=float)
    sigma = s_to_sigma(rho_arr, s_grid)

    rho_input = np.zeros((4, n), dtype=np.float64)
    rho_input[0, :] = rho_arr
    rho_input[1, :] = np.sqrt(np.clip(sigma, 0.0, None))
    eps_c_pbe, *_ = eval_xc("GGA_C_PBE", rho_input, spin=0, deriv=0)
    eps_c_pw92, *_ = eval_xc("LDA_C_PW", rho_arr, spin=0, deriv=0)
    eps_c_pbe = np.asarray(eps_c_pbe, dtype=float)
    eps_c_pw92 = np.asarray(eps_c_pw92, dtype=float)
    return np.where(np.abs(eps_c_pw92) > 1e-30, eps_c_pbe / eps_c_pw92, 1.0)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_enhancement_factors(run_dir: Path, out_path: Path, *,
                             s_max: float = 3.0, n_points: int = 240) -> Path:
    """Figure D -- F_x(s) and F_c(s; r_s) per architecture, each against the
    reference it was PRETRAINED to (PBE for the GGA archs, SCAN for the
    meta-GGA ones), plus an alpha sweep for the meta-GGA family."""
    reps = representative_specs(run_dir)
    archs = [a for a in ARCH_ORDER if a in reps]
    s_grid = np.linspace(1e-3, s_max, n_points)

    # Load each arch once; compute its Fx + Fc(rs) curves.
    fx_curves: Dict[str, np.ndarray] = {}
    fc_curves: Dict[str, Dict[float, np.ndarray]] = {}
    # meta-GGA archs additionally get an alpha family. Their zero-descriptor
    # curve IS the alpha=0 slice, which is the single-orbital corner -- shown
    # alone it misrepresents the functional everywhere molecules sample it.
    mgga_alpha: Dict[str, Dict[float, np.ndarray]] = {}
    for arch in archs:
        try:
            _spec, model = load_trained_model(run_dir, reps[arch])
        except Exception as exc:  # noqa: BLE001 - report and skip a bad ckpt
            print(f"  [warn] could not load {arch} "
                  f"(spec {reps[arch]}): {exc}", flush=True)
            continue
        fx_curves[arch] = model_fx_curve(model, s_grid)
        fc_curves[arch] = {rs: model_fc_curve(model, s_grid, rs)
                           for rs in _RS_PANELS}
        if is_meta_gga(model):
            mgga_alpha[arch] = {a: model_fx_curve(model, s_grid, alpha=a)
                                for a in _ALPHA_PANELS}

    with plt.rc_context(sib._STYLE):
        # A meta-GGA arch earns a fifth panel (its alpha sweep). Without one the
        # layout is the historical 2x2, so GGA-only runs render unchanged.
        if mgga_alpha:
            fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
            ax_fx, ax_alpha = axes[0, 0], axes[0, 1]
            fc_axes = [axes[0, 2], axes[1, 0], axes[1, 1]]
            axes[1, 2].axis("off")
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            ax_fx, ax_alpha = axes[0, 0], None
            fc_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]

        # Panel 1: F_x(s) ---------------------------------------------------
        # NOTE the meta-GGA curves here are the alpha=0 slice (zero extra
        # descriptors) -- the single-orbital corner. Their representative
        # behaviour is in the alpha panel; this one is kept so all archs
        # appear on one axis, and the caption says which slice it is.
        for arch in archs:
            if arch in fx_curves:
                ax_fx.plot(s_grid, fx_curves[arch], linewidth=1.4,
                           color=ARCH_COLOR[arch], label=arch)
        ax_fx.plot(s_grid, pbe_fx_curve(s_grid), "k--", linewidth=1.8,
                   label="PBE (analytic)")
        if mgga_alpha:
            # SCAN at alpha=0 is the reference for the meta-GGA archs on THIS
            # slice; PBE is the wrong oracle for them (they pretrain to SCAN).
            scan0 = scan_fx_curve(s_grid, 0.0)
            if scan0 is not None:
                ax_fx.plot(s_grid, scan0, ls="-.", color="#8c564b",
                           linewidth=1.8, label=r"SCAN (libxc, $\alpha$=0)")
        ax_fx.axhline(1.804, ls=":", color="0.5", linewidth=1.0,
                      label="Lieb-Oxford bound (1.804)")
        ax_fx.set_xlabel("reduced gradient  s")
        ax_fx.set_ylabel(r"$F_x(s)$")
        ax_fx.set_title("Exchange enhancement"
                        + (r"  ($\alpha$=0 slice for meta-GGA)"
                           if mgga_alpha else ""))
        ax_fx.legend(fontsize=6, ncol=2)
        ax_fx.grid(True, alpha=0.3)

        # Panel: meta-GGA alpha sweep vs SCAN -------------------------------
        if ax_alpha is not None:
            cmap = plt.get_cmap("viridis")
            n_a = max(1, len(_ALPHA_PANELS) - 1)
            for arch, curves in mgga_alpha.items():
                for i, a in enumerate(_ALPHA_PANELS):
                    ax_alpha.plot(s_grid, curves[a], linewidth=1.5,
                                  color=cmap(i / n_a),
                                  label=fr"{arch} $\alpha$={a:g}")
                    ref = scan_fx_curve(s_grid, a)
                    if ref is not None:
                        ax_alpha.plot(s_grid, ref, ls="--", linewidth=1.0,
                                      color=cmap(i / n_a), alpha=0.75)
            ax_alpha.set_xlabel("reduced gradient  s")
            ax_alpha.set_ylabel(r"$F_x(s;\alpha)$")
            ax_alpha.set_title(r"meta-GGA: $F_x$ vs iso-orbital $\alpha$"
                               "\n(solid = net, dashed = SCAN target)")
            ax_alpha.legend(fontsize=5, ncol=2)
            ax_alpha.grid(True, alpha=0.3)

        # Panels 2-4: F_c(s) at three r_s ----------------------------------
        for ax, rs in zip(fc_axes, _RS_PANELS):
            for arch in archs:
                if arch in fc_curves:
                    ax.plot(s_grid, fc_curves[arch][rs], linewidth=1.3,
                            color=ARCH_COLOR[arch], label=arch)
            pbe_fc = pbe_fc_curve(s_grid, rs)
            if pbe_fc is not None:
                ax.plot(s_grid, pbe_fc, "k--", linewidth=1.8, label="PBE")
            ax.set_xlabel("reduced gradient  s")
            ax.set_ylabel(r"$F_c(s)$  (enh. over PW92)")
            ax.set_title(fr"Correlation enhancement, $r_s={rs:g}$ ($\zeta=0$)")
            ax.grid(True, alpha=0.3)
        fc_axes[0].legend(fontsize=6, ncol=2)

        fig.suptitle(
            ("Learned enhancement factors vs the reference each arch was "
             f"PRETRAINED to · {run_dir.name}" if mgga_alpha else
             f"Learned enhancement factors vs PBE · {run_dir.name}"),
            fontsize=12)
        loaded = [a for a in archs if a in fx_curves]
        missing = [a for a in ARCH_ORDER if a not in reps]
        cov = (f"Archs shown: {len(loaded)}/{len(ARCH_ORDER)} "
               f"({', '.join(loaded)}).")
        if missing:
            cov += f"  NOT TRAINED in this run: {', '.join(missing)}."
        if mgga_alpha:
            # Which oracle belongs to which arch. Reading a meta-GGA against PBE
            # mistakes SCAN's genuine flatness in s for a defect, so the split
            # is stated on the figure rather than left to the reader.
            cov += ("  PRETRAIN TARGET: " + ", ".join(sorted(mgga_alpha))
                    + " clone SCAN (pretrain.py selects Fx_scan_all for "
                    "meta_gga archs); every other arch clones PBE -- read each "
                    "against its own reference. F_x panel is the alpha=0 "
                    "(single-orbital) slice for the meta-GGA archs; their "
                    "representative behaviour is the alpha panel.")
        # The pretrain-target sentence makes this caption long; wrap it and
        # give the band room rather than letting it run off the canvas.
        fig.text(0.5, 0.055 if mgga_alpha else 0.028, cov, ha="center",
                 fontsize=7, color="#a33", wrap=True)
        fig.text(0.5, 0.005, _provenance(run_dir, bool(mgga_alpha)),
                 ha="center", fontsize=7, color="#777777")
        fig.tight_layout(rect=(0, 0.10 if mgga_alpha else 0.05, 1, 0.96))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None,
                   help="pulled run dir (default: latest ablation_notransform)")
    p.add_argument("--outdir", default=str(
        Path(__file__).resolve().parent / "figures_ablation_notransform"))
    args = p.parse_args(argv)

    run_dir = sib._resolve_run_dir(args.run_dir)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir: {run_dir}")
    out = plot_enhancement_factors(run_dir, outdir / "ablation_enhancement_factors.png")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

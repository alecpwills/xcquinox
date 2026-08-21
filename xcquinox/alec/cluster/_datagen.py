"""Datagen-stage entrypoint: the FIRST stage of the cluster job graph.

The graph is ``datagen -> pretrain -> preflight -> train -> eval``. This stage
generates the per-atom Fx/Fc pretrain-target data (``pretrain_data[_polarized].npz``)
into ``cfg.pretrain.data_dir`` BEFORE the pretrain stage consumes it. Previously
that generation lived inside ``inputs.prepare_inputs`` which only runs in the
preflight stage (``afterok:pretrain``), i.e. AFTER the pretrain stage that needs
the data, so pretrain raised ``FileNotFoundError`` and the whole ``afterok``
chain went ``DependencyNeverSatisfied``. Running it here, gated before pretrain,
fixes that ordering.

The generator is idempotent (``ensure_pretrain_data`` skips a file whose manifest
already matches the requested basis/grid_level), so a re-submit is a cheap no-op.

It produces EVERY pretrain-data file the sweep's architectures require: the set
of required files is the set of distinct polarization flags of the swept archs
(after applying the run-level ``use_polarized_correlation`` patch exactly as
``spec_builder`` does), each named through
``pretrain_data_gen.pretrain_data_filename`` -- the one naming function
``run_pretrain`` also reads through. ``descriptors=True`` writes the
``cusp_all`` / ``dm_all`` columns the descriptor archs (deep_cusp / deep_dm /
deep_combined*) need, so one file serves base, attn, cusp, dm, combined, and
notransform archs.

JAX precision
-------------
The generator's kinetic-energy density, iso-orbital indicator, rung-3.5
occupancies and cusp feature are JAX computations, and the parent density
reaches it as a ``jnp`` array out of ``data.precompute_fixed_density_data``, so
the worker must compute in float64. ``_route_jax_env`` sets ``JAX_ENABLE_X64``
and flips the live configuration -- under ``python -m`` the package
initializers import jax before this module's body runs, so the environment
variable alone is read too late -- and ``main`` refuses to generate unless a
float64 host array keeps its dtype on entering JAX. Before this was explicit
the worker computed in float64 only because ``import pyscfad`` enables x64 as a
side effect of being imported.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

from xcquinox.alec.config import get_architecture
from xcquinox.alec.cluster.grid_config import load_grid_config


# ---------------------------------------------------------------------------
# Mockable heavy-call seam, tests monkeypatch ``_datagen._ensure_pretrain_data``
# to assert the generation calls without running real SCFs. Bound lazily in
# ``main`` rather than at import, because importing the generator pulls in
# jax.numpy and the precision routing below must run first; a test that patches
# the name still wins, since the rebind only fires while the value is None.
# ---------------------------------------------------------------------------
_ensure_pretrain_data = None


def _route_jax_env():
    """Pin JAX to float64: the environment variable and the live configuration.

    ``JAX_ENABLE_X64=1`` is honored by a jax that has not been imported yet and
    is inherited by any child process; ``cluster._pretrain`` and
    ``cluster._eval_one_spec`` open the same way. It is not sufficient here:
    ``python -m xcquinox.alec.cluster._datagen`` runs ``xcquinox/__init__``
    first, which imports jax, so by the time this function runs the variable
    is read too late and the live switch ``jax.config.update`` is the
    effective one. JAX reads the flag when an array is created, so every array
    the generator builds after this call is float64. ``JAX_PLATFORMS`` is left
    untouched so the sbatch-requested device is honored.
    """
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    jax.config.update("jax_enable_x64", True)


def _require_x64():
    """``None`` when JAX keeps float64, else a message naming the defect.

    The guarantee must not rest on a third-party import side effect (``import
    pyscfad`` enables x64), so the live behavior is checked: a float64 host
    array must enter JAX as float64.
    """
    import jax
    import jax.numpy as jnp
    flag = bool(jax.config.jax_enable_x64)
    dtype = jnp.asarray(np.ones(1, dtype=np.float64)).dtype
    if flag and dtype == np.float64:
        return None
    return (
        f"JAX is not computing in float64 (jax_enable_x64={flag}; a float64 "
        f"host array enters JAX as {dtype}); the pretrain-data tau, alpha, "
        "rung-3.5 and cusp columns would be written in single precision"
    )


def _log(msg: str) -> None:
    """Emit a legible progress line to the datagen SLURM log (project rule:
    long-running steps must show progress so a running job isn't mistaken for a
    hang)."""
    print(f"[datagen] {msg}", flush=True)


def _required_polarized_flags(cfg) -> list[bool]:
    """The distinct ``polarized`` flags the sweep's archs actually consume.

    Mirrors ``spec_builder``: each swept arch is patched with the run-level
    ``use_polarized_correlation`` before its flag is read. The flag is read
    directly rather than parsed back out of a filename suffix, which stops
    working as soon as a name carries a second qualifier (the parent-density
    suffix of ``pretrain_data_gen.pretrain_data_filename``); the name is built
    from this flag by that same function, which ``run_pretrain`` reads through.
    Returns a deterministic list of distinct flags (one per distinct required
    file), normally ``[True]`` or ``[False]`` since the polarization flag is
    run-level, but a future per-arch/mixed sweep yields both.
    """
    run_polarized = bool(getattr(cfg, "use_polarized_correlation", False))
    flags: dict[bool, None] = {}
    for name in cfg.sweep.arch:
        arch = get_architecture(name)
        if run_polarized:
            arch = dataclasses.replace(arch, use_polarized_correlation=True)
        flags.setdefault(
            bool(getattr(arch, "use_polarized_correlation", False)), None)
    return sorted(flags)  # deterministic: [False] < [True] < [False, True]


def main(argv=None) -> int:
    """Datagen-job entrypoint. Returns a process exit code (0 = success).

    ``argv[0]`` is the run dir. Returns 1 on any failure so the pretrain array's
    ``afterok:datagen`` dependency blocks (rather than letting pretrain run
    against missing/partial data).
    """
    # Precision first: the generator import below pulls in jax.numpy, and the
    # refusal that follows is what makes the float64 guarantee explicit.
    _route_jax_env()
    global _ensure_pretrain_data
    from xcquinox.alec import pretrain_data_gen as _pdg
    if _ensure_pretrain_data is None:
        _ensure_pretrain_data = _pdg.ensure_pretrain_data
    problem = _require_x64()
    if problem is not None:
        _log(f"ERROR: {problem}")
        return 1
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        _log("ERROR: no run directory given; usage: _datagen <run_dir>")
        return 1
    run_dir = os.path.abspath(argv[0])
    _log(f"starting datagen for run_dir={run_dir}")

    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(cfg_path):
        _log(f"ERROR: resolved_config.yaml not found at {cfg_path}")
        return 1
    try:
        cfg = load_grid_config(cfg_path)
    except (ValueError, ImportError, OSError) as exc:
        _log(f"ERROR: failed to load resolved config: {exc}")
        return 1

    data_dir = cfg.pretrain.data_dir
    flags = _required_polarized_flags(cfg)
    required = [_pdg.pretrain_data_filename(p) for p in flags]
    _log(
        f"archs={list(cfg.sweep.arch)} -> required: {required} | "
        f"basis={cfg.inputs.basis} grid_level={cfg.inputs.grid_level} "
        f"density_fit={cfg.inputs.density_fit} data_dir={data_dir}"
    )
    try:
        for polarized in flags:
            path = _ensure_pretrain_data(
                data_dir,
                basis=cfg.inputs.basis,
                grid_level=cfg.inputs.grid_level,
                density_fit=cfg.inputs.density_fit,
                auxbasis=cfg.inputs.auxbasis,
                polarized=polarized,
                descriptors=True,
                # empty config tuple -> the generator's DEFAULT_PRETRAIN_ATOMS
                **({"atoms": tuple(tuple(a) for a in cfg.pretrain.atoms)}
                   if getattr(cfg.pretrain, "atoms", ()) else {}),
            )
            _log(f"ensured pretrain data (polarized={polarized}): {path}")
    except Exception as exc:  # noqa: BLE001, fail the stage loudly + non-zero.
        _log(f"ERROR: pretrain-data generation failed: {type(exc).__name__}: {exc}")
        return 1

    _log("datagen complete, all required pretrain-data files present.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())

"""Datagen-stage entrypoint — the FIRST stage of the cluster job graph.

The graph is ``datagen -> pretrain -> preflight -> train -> eval``. This stage
generates the per-atom Fx/Fc pretrain-target data (``pretrain_data[_polarized].npz``)
into ``cfg.pretrain.data_dir`` BEFORE the pretrain stage consumes it. Previously
that generation lived inside ``inputs.prepare_inputs`` which only runs in the
preflight stage (``afterok:pretrain``) — i.e. AFTER the pretrain stage that needs
the data — so pretrain raised ``FileNotFoundError`` and the whole ``afterok``
chain went ``DependencyNeverSatisfied``. Running it here, gated before pretrain,
fixes that ordering.

The generator is idempotent (``ensure_pretrain_data`` skips a file whose manifest
already matches the requested basis/grid_level), so a re-submit is a cheap no-op.

It produces EVERY pretrain-data file the sweep's architectures require: the set of
required filenames is computed per-arch via ``pretrain._pretrain_data_filename``
(polarized vs unpolarized), after applying the run-level ``use_polarized_correlation``
patch exactly as ``spec_builder`` does. ``descriptors=True`` writes the ``cusp_all``
/ ``dm_all`` columns the descriptor archs (deep_cusp / deep_dm / deep_combined*)
need, so one file serves base, attn, cusp, dm, combined, and notransform archs.
"""
from __future__ import annotations

import dataclasses
import os
import sys

from xcquinox.alec.config import get_architecture
from xcquinox.alec.cluster.grid_config import load_grid_config
from xcquinox.alec.pretrain import _pretrain_data_filename
from xcquinox.alec import pretrain_data_gen as _pretrain_data_gen


# ---------------------------------------------------------------------------
# Mockable heavy-call seam — tests monkeypatch ``_datagen._ensure_pretrain_data``
# to assert the generation calls without running real PBE SCFs.
# ---------------------------------------------------------------------------
_ensure_pretrain_data = _pretrain_data_gen.ensure_pretrain_data


def _log(msg: str) -> None:
    """Emit a legible progress line to the datagen SLURM log (project rule:
    long-running steps must show progress so a running job isn't mistaken for a
    hang)."""
    print(f"[datagen] {msg}", flush=True)


def _required_polarized_flags(cfg) -> list[bool]:
    """The distinct ``polarized`` flags the sweep's archs actually consume.

    Mirrors ``spec_builder``: each swept arch is patched with the run-level
    ``use_polarized_correlation`` before its required pretrain-data filename is
    resolved. Returns a deterministic list of distinct flags (one per distinct
    required file) — normally ``[True]`` or ``[False]`` since the polarization
    flag is run-level, but a future per-arch/mixed sweep yields both.
    """
    run_polarized = bool(getattr(cfg, "use_polarized_correlation", False))
    flags: dict[bool, None] = {}
    for name in cfg.sweep.arch:
        arch = get_architecture(name)
        if run_polarized:
            arch = dataclasses.replace(arch, use_polarized_correlation=True)
        # _pretrain_data_filename -> "pretrain_data_polarized.npz" iff polarized.
        is_polarized = _pretrain_data_filename(arch).endswith("_polarized.npz")
        flags.setdefault(is_polarized, None)
    return sorted(flags)  # deterministic: [False] < [True] < [False, True]


def main(argv=None) -> int:
    """Datagen-job entrypoint. Returns a process exit code (0 = success).

    ``argv[0]`` is the run dir. Returns 1 on any failure so the pretrain array's
    ``afterok:datagen`` dependency blocks (rather than letting pretrain run
    against missing/partial data).
    """
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
    required = ["pretrain_data_polarized.npz" if p else "pretrain_data.npz"
                for p in flags]
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
            )
            _log(f"ensured pretrain data (polarized={polarized}): {path}")
    except Exception as exc:  # noqa: BLE001 — fail the stage loudly + non-zero.
        _log(f"ERROR: pretrain-data generation failed: {type(exc).__name__}: {exc}")
        return 1

    _log("datagen complete — all required pretrain-data files present.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())

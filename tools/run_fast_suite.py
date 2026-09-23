"""Run the fast suite in chunks that stay under the kernel's memory-mapping ceiling.

Under jaxlib 0.10.2 one pytest process running the whole fast suite reaches
``vm.max_map_count`` (65530 on a stock Linux kernel) inside XLA's compiler and aborts:
the differentiated-SCF tests add several thousand mappings each and a process never
gives them back (measured 2026-09-21, the count at 65235
when the fourth alphabetical quarter of the pipeline tests aborted in one process). The
suite therefore runs in ten interpreters, one chunk each, and every chunk ends well below
the ceiling: the mapping count at each chunk's end, measured on 2026-09-22 after the suite was
pruned to its core, is recorded beside the chunk.
A hosted runner with root raises the ceiling instead (the workflow's ``sysctl`` step);
this runner is for the workstation and the cluster, where the ceiling is what it is.

The chunks partition the tracked test modules of the five test paths of
``pyproject.toml`` (``tools/test_run_fast_suite.py`` holds that): the four small directories
in one chunk, the pipeline tests in alphabetical quarters, the fourth quarter in three
parts, and its heaviest module -- ``test_training_gradient_consistency.py``, 63174
mappings over its 25 tests in one interpreter -- alone in three parts by test function.
A test module added to the tree fails the partition test until it is assigned here.

Usage::

    python -m tools.run_fast_suite [--dry-run] [--root DIR] [--basetemp DIR]
                                   [--log-dir DIR] [--python EXE] [--marker EXPR]

Each chunk's output goes to ``<log-dir>/<chunk>.log`` and its summary line is printed
as it finishes; the exit status is non-zero if any chunk's is. ``--basetemp`` is one
fixed directory pytest clears at the start of every chunk, so the runs never accumulate.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: The marker expression of the fast suite: the slow items are deselected here only.
MARKER = "not slow and not oracle"

_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = "xcquinox/pipeline/tests/"
_TRAINING_GRADIENT = _PIPELINE + "test_training_gradient_consistency.py"
_SUMMARY = re.compile(r"^=+ .*\b(passed|failed|error|errors|no tests ran)\b.*=+$")


@dataclass(frozen=True)
class Chunk:
    """One interpreter's share of the suite.

    ``paths`` are repository-relative test directories or modules; ``keyword`` is a
    pytest ``-k`` expression selecting test functions of the one module in ``paths``;
    ``peak`` is the interpreter's mapping count at the end of the chunk as measured on
    2026-09-22 (informative; the ceiling is 65530).
    """
    name: str
    paths: tuple[str, ...]
    keyword: str | None = None
    peak: int | None = None


def _pipeline(*names: str) -> tuple[str, ...]:
    return tuple(_PIPELINE + name for name in names)


CHUNKS: tuple[Chunk, ...] = (
    Chunk("A", ("xcquinox/tests", "tools", "notebooks", "hpcjobs"), peak=2223),
    Chunk("B", _pipeline(
        "test_arch_names.py", "test_balancing.py", "test_benchmark_refs.py",
        "test_bh76w411_pool.py", "test_certificate_two_tier_gate.py",
        "test_checkpoint_class.py", "test_closed_shell_byte_identity.py",
        "test_cluster_analyze.py", "test_cluster_cli.py", "test_cluster_datagen.py",
        "test_cluster_domain.py", "test_cluster_eval_basis.py",
        "test_cluster_eval_worker.py", "test_cluster_examples.py",
        "test_cluster_fidelity.py", "test_cluster_grid_config.py",
        "test_cluster_inputs.py", "test_cluster_job_tracking.py",
        "test_cluster_materialize.py", "test_cluster_preflight.py",
        "test_cluster_pretrain.py", "test_cluster_spec_builder.py",
        "test_cluster_spec_golden.py", "test_cluster_submit_eval.py",
        "test_cluster_submit.py", "test_cluster_sync.py", "test_cluster_train_task.py",
        "test_cluster_workflow_matrix.py", "test_coldstart_retro.py",
        "test_compile_counter_fixture.py", "test_compile_ratchet.py", "test_config.py",
        "test_conftest_hard_exit.py"), peak=23740),
    Chunk("C", _pipeline(
        "test_constraints.py",
        "test_converged_channel.py", "test_cusp_descriptor_bounded.py",
        "test_cusp_log_transform_skew.py", "test_data_cderi.py", "test_data.py",
        "test_defused_grad.py", "test_descriptor_coordinates.py", "test_descriptors.py",
        "test_df_jk.py", "test_df_memory.py", "test_dfs_pretrain_set.py",
        "test_discovery_probes.py", "test_eval_holdout_parallel.py",
        "test_eval_holdout.py", "test_eval_probes.py", "test_evaluation.py",
        "test_external_refs_df.py", "test_external_refs.py", "test_features.py",
        "test_full_benchmark_pools.py", "test_generate_polarized_script.py",
        "test_generate_step7_subsets.py", "test_integration_end_to_end.py",
        "test_losses.py", "test_losses_step7.py",
        "test_metagga_indicator_domain.py", "test_metagga_pretrain.py",
        "test_metagga.py", "test_models.py",
        "test_networks.py", "test_oep_per_species_emit_overrides.py"), peak=27624),
    Chunk("D", _pipeline(
        "test_oep_per_species_tune.py", "test_oep.py", "test_oep_uks.py",
        "test_oneshot.py", "test_orientation_lock.py", "test_padding.py",
        "test_parallel.py", "test_parent_anchor.py", "test_parents.py",
        "test_parents_scan.py", "test_pbe_anchor.py", "test_pretrain_board_local.py",
        "test_pretrain_cloning_protocol.py", "test_pretrain_data_basis.py",
        "test_pretrain_data_gen.py", "test_pretrain_energy_term.py",
        "test_pretrain_mesh.py", "test_pretrain.py", "test_pretrain_schema.py",
        "test_pretrain_set.py", "test_pretrain_systems.py", "test_pretrain_weighted.py",
        "test_procmem.py", "test_pyscfad_gradflow.py", "test_pyscf_determinism.py",
        "test_refinalize_verbatim.py", "test_required_keys_df.py", "test_rung35.py",
        "test_rungs.py", "test_scf_backends.py", "test_scf_diff.py", "test_scf_tail.py",
        "test_seed_cache.py", "test_self_attention.py", "test_shape_padding.py",
        "test_smoke_preflight_uks_oep.py"), peak=24315),
    Chunk("E0", _pipeline(
        "test_solv01_split_xc.py", "test_solver_config_df.py",
        "test_solver_manual_checkpoint.py", "test_solver_manual_df.py",
        "test_solver_manual.py", "test_solver.py", "test_species_matching.py",
        "test_spin_scaling_oracles.py", "test_spin_scaling_precompute.py",
        "test_spin_scaling_pyscfad.py", "test_spin_scaling_solver_manual.py"),
        peak=8942),
    Chunk("E1a", _pipeline(
        "test_notebooks_end_to_end.py",
        "test_subset_selection_gpu.py",
        "test_subset_selection_parallel.py", "test_subset_selection.py",
        "test_sym_break_shift.py", "test_t1_diagnostic.py",
        "test_trained_checkpoint_loaders.py"), peak=1793),
    Chunk("TG1", (_TRAINING_GRADIENT,),
          keyword="test_vxc_equals_whole_energy_gradient_elementwise or "
                  "test_training_gradient_matches_fd_uks_mgga", peak=9522),
    Chunk("TG2", (_TRAINING_GRADIENT,),
          keyword="test_training_gradient_matches_fd_energy_and_dm_loss", peak=34319),
    Chunk("TG3", (_TRAINING_GRADIENT,),
          keyword="test_energy_training_descends or test_density_training_descends",
          peak=33296),
    # The tracked slow module of the pipeline tree rides with the last chunk: the marker
    # deselects it in the fast suite, and the partition rule counts every tracked module.
    Chunk("E2", _pipeline(
        "test_training_points.py", "test_train_one_spec.py", "test_train.py",
        "test_uks_atom_gradflow.py", "test_uks_oneshot.py", "test_uks_scf.py",
        "test_uks_vxc_integration.py", "test_validate_run.py",
        "test_vxc_padding_neutrality.py", "test_worker_hard_exit.py",
        "test_workers.py", "slow/test_scf_train.py"), peak=17500),
)


def commands(root, basetemp, log_dir, marker: str = MARKER, python: str | None = None
             ) -> list[list[str]]:
    """One pytest command line per chunk, in :data:`CHUNKS` order, to be run from
    ``root``: the interpreter (``python`` or the running one), ``-m`` with the marker,
    the chunk's paths, ``-k`` with its keyword when it has one, and ``--basetemp``.
    ``log_dir`` is where :func:`main` sends each chunk's output; the command line itself
    carries no log path, so a caller can redirect as it likes."""
    del root, log_dir     # named for the caller's clarity; the argv carries neither
    exe = python or sys.executable
    out = []
    for chunk in CHUNKS:
        argv = [exe, "-m", "pytest", "-m", marker, *chunk.paths]
        if chunk.keyword:
            argv += ["-k", chunk.keyword]
        argv.append(f"--basetemp={basetemp}")
        out.append(argv)
    return out


def summary_line(log_path) -> str:
    """pytest's own summary line of a chunk's log, or a statement that there is none
    (a process that aborted inside XLA leaves no summary line, which is the signature
    of the ceiling)."""
    found = "(no summary line: the process ended without one)"
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if _SUMMARY.match(line.rstrip("\n")):
                found = line.strip()
    return found


def _parser() -> argparse.ArgumentParser:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache = Path(os.environ.get("XCQ_CACHE_ROOT", Path.home() / ".cache" / "xcquinox"))
    parser = argparse.ArgumentParser(
        prog="python -m tools.run_fast_suite",
        description="Run the fast suite in chunks that stay under vm.max_map_count.")
    parser.add_argument("--root", default=str(_ROOT),
                        help="the repository root the chunks run from")
    parser.add_argument("--basetemp", default=str(cache / "pytest_tmp" / "suite"),
                        help="pytest's --basetemp, one fixed directory for every chunk")
    parser.add_argument("--log-dir", default=str(cache / "logs" / f"suite_{stamp}"),
                        help="where each chunk's output goes, one file per chunk")
    parser.add_argument("--python", default=None,
                        help="the interpreter to run pytest with (default: this one)")
    parser.add_argument("--marker", default=MARKER, help="pytest's -m expression")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command lines and exit")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    lines = commands(args.root, args.basetemp, args.log_dir, marker=args.marker,
                     python=args.python)
    if args.dry_run:
        for line in lines:
            print(shlex.join(line))
        return 0
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for chunk, line in zip(CHUNKS, lines):
        log = log_dir / f"{chunk.name}.log"
        with open(log, "w", encoding="utf-8") as fh:
            status = subprocess.run(line, cwd=args.root, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
        print(f"{chunk.name}: {summary_line(log)} (exit {status}; {log})", flush=True)
        if status != 0:
            failed.append(chunk.name)
    if failed:
        print(f"chunks that did not pass: {', '.join(failed)}", flush=True)
        return 1
    print(f"every chunk passed; logs under {log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

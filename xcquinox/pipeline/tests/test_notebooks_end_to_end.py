"""The tracked training notebooks execute end to end.

Each tracked notebook is regenerated from its builder with a narrow configuration and executed
through nbclient in the kernel of the interpreter running the tests. These are the notebooks'
tests: what a reader runs is what is executed here, and the cells' contents are not pinned
separately (the products are held to their builders by ``notebooks/test_notebook_hygiene.py``).
Slow-marked: a run takes ten minutes.

The kernel named ``python3`` must be the interpreter running the tests: jupyter resolves that
name to the running interpreter's own ipykernel only when nothing else registers the name, and
a kernel from another environment would execute the notebook on another stack (a
run whose kernel comes from another environment can die in glibc's heap
consolidation after the training cell). The resolution is checked before any cell runs.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import nbformat
import pytest

_REPO = Path(__file__).resolve().parents[3]
_BUILDERS = _REPO / "notebooks"


def _load_builder(name: str):
    """The builder module loaded from its file (``notebooks/`` is not a package)."""
    path = _BUILDERS / name
    if not path.is_file():
        pytest.fail(f"notebook builder not found at {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _client(nb, *, timeout: int, cwd: Path | None = None):
    """An nbclient client whose kernel command launches the running interpreter."""
    import nbclient

    resources = {"metadata": {"path": str(cwd)}} if cwd is not None else None
    client = nbclient.NotebookClient(nb, timeout=timeout, kernel_name="python3",
                                     resources=resources)
    # The launch command, not the raw spec: ipykernel's own spec names a bare ``python``,
    # which jupyter substitutes with the running interpreter at launch.
    manager = client.create_kernel_manager()
    command = manager.format_kernel_cmd()
    assert command[0] == sys.executable, (
        f"the notebook kernel 'python3' launches {command[0]} "
        f"({manager.kernel_spec.resource_dir}), not the interpreter running the tests, "
        f"{sys.executable}; the test extra's ipykernel provides the kernel of the running "
        "interpreter, and a kernelspec registered under that name elsewhere takes "
        "precedence over it")
    return client


@pytest.mark.slow
def test_gga_training_pipeline_notebook_runs_end_to_end(tmp_path):
    """The library-driven training notebook on a 1-arch x 1-loss configuration: every cell
    executes and the files it promises exist. Numerical correctness is not the claim (a
    250-step training run is not converged)."""
    gen = _load_builder("_build_gga_training_pipeline_notebook.py")
    nb_path = tmp_path / "pipeline_smoke.ipynb"
    checkpoint_base = str(tmp_path / "ckpt")
    gen.main(str(nb_path), arch_names=("shallow",), loss_names=("A_atomization",),
             checkpoint_base=checkpoint_base)
    nb = nbformat.read(str(nb_path), as_version=4)
    _client(nb, timeout=900, cwd=tmp_path).execute()
    for relative in ("pretrain_data/pretrain_data.npz", "pretrain/shallow/xnet.eqx",
                     "pretrain/shallow/cnet.eqx", "external_data/H.npz",
                     "external_data/O.npz", "external_data/H2O.npz",
                     "external_data/H_metadata.json", "external_data/O_metadata.json"):
        assert os.path.isfile(f"{checkpoint_base}/{relative}"), relative


@pytest.mark.slow
def test_gga_training_anchor_transfer_notebook_runs_end_to_end(tmp_path):
    """The anchor-and-transfer notebook on a 1-arch x 1-loss x 1-solver configuration with
    small step counts (the builder's parameters): every cell executes, the fidelity
    certificate its specs cell writes admits the pretrained pair to training, and the
    evaluation and transfer cells complete."""
    gen = _load_builder("_build_gga_training_anchor_transfer_notebook.py")
    nb = gen.main(
        arch_names=("deep_combined",),
        loss_names=("L1_B",),
        solver_labels=("oneshot",),
        checkpoint_base=str(tmp_path / "checkpoints_anchor_transfer_smoke"),
        output_path=str(tmp_path / "anchor_transfer_smoke.ipynb"),
        pretrain_n_steps=100, train_n_steps_short=10, train_n_steps_long=20,
    )
    _client(nb, timeout=600).execute()

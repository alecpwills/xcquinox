"""The cold-start convergence census (cluster/coldstart_census).

The census runs the training solver of a spec from the functional-free atomic
guess, once per training species, forward only, and records what the SCF did:
whether it converged, how many cycles it took, the energy it reached, the last
energy step and the whole trace. It is the pre-training pass of the published
script written as a record, and it is a REPORT -- nothing in the harness gates
on its verdict.

Every assertion is made against an oracle outside the census: the solver's own
``run_scf`` result on the same model and the same cold-start record, and the
``SolverConfig`` the spec carries. A census that produced its numbers some
other way, or ran a protocol other than the spec's own, is separated from one
that drives the spec's solver.
"""
import json
import os

import numpy as np
import pytest

from xcquinox.pipeline.config import ArchitectureConfig, TrainingSpec
from xcquinox.pipeline.solver import SolverConfig, SolverMode
from xcquinox.pipeline.tests.fixtures.molecules import (
    h_atom,
    h2_molecule,
    h2o_molecule,
    o_atom,
)


# The keys every row states, the record the analysis reads.
_ROW_KEYS = frozenset({
    "name", "converged", "cycles_run", "total_energy", "last_delta_e",
    "energy_trace", "wall_s",
})


def _census_arch():
    """A small network: the census measures the SCF, not the architecture."""
    return ArchitectureConfig(
        name="census_t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )


def _census_spec(tmp_path, *, molecules, targets, atom_energies,
                 solver_config, name="ckpt"):
    """A per-molecule training spec carrying ``solver_config`` where the loop
    reads it (``loss_kwargs["solver_config"]``)."""
    return TrainingSpec.from_dicts(
        arch=_census_arch(),
        molecules=molecules,
        targets=targets,
        atom_energies=atom_energies,
        loss_name="L5_gradnorm_vxc_step7",
        loss_kwargs={"solver_config": solver_config},
        n_steps=1,
        lr_start=1e-3,
        lr_end=1e-5,
        lr_decay_start=0.0,
        grad_clip=1.0,
        checkpoint_dir=os.path.join(str(tmp_path), name),
        seed=0,
        update_scheme="per_molecule",
        require_atom_anchors=False,
    )


def _three_species_spec(tmp_path, solver_config):
    """H, O and H2O: two open-shell atoms and one closed-shell compound, so the
    census is exercised on both SCF branches."""
    return _census_spec(
        tmp_path,
        molecules=(h_atom(), o_atom(), h2o_molecule()),
        targets={"H": 0.0, "O": 0.0, "H2O": 0.2},
        atom_energies={"H": -0.5, "O": -74.0},
        solver_config=solver_config,
    )


def _h2_spec(tmp_path, solver_config, name="ckpt"):
    """One closed-shell compound, for the single-species checks."""
    return _census_spec(
        tmp_path,
        molecules=(h2_molecule(),),
        targets={"H2": 0.2},
        atom_energies={"H": -0.5},
        solver_config=solver_config,
        name=name,
    )


def _fresh_model(spec):
    from xcquinox.pipeline.models import AlecGGAModel
    return AlecGGAModel.from_arch(spec.arch, seed=spec.seed)


def _solver_oracle(spec, model, census_cfg):
    """``{name: SCFResult}`` from running ``census_cfg`` on the cold-start
    record of every species of ``spec``, outside the census."""
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.solver import run_scf

    out = {}
    for mol in spec.molecules:
        md = precompute_fixed_density_data(
            mol, required_keys=("eri",),
            descriptors=spec.arch.materialize_descriptors(),
            seed_source=census_cfg.seed_source,
            seed_cache_dir=census_cfg.seed_cache_dir,
        )
        out[mol.name] = run_scf(census_cfg, model, md, forward_only=True)
    return out


def test_the_census_runs_the_training_solver_from_the_cold_start(tmp_path):
    """``census_solver_config`` moves the seed and nothing else, and
    ``census_rows`` reports, per species, what that solver did.

    The first block holds the override to the spec's own protocol: only
    ``seed_source`` and ``seed_cache_dir`` may differ, so the census measures
    the cycles, tolerance, mixer and tail the run trains under rather than the
    held-out channel's separate protocol. The second block compares every
    reported field against the solver's own result on the same model and the
    same cold-start record. The third separates ``converged`` -- the solver's
    latched flag -- from the cycle count: with the convergence freeze off a
    species converges while still running every cycle, so a verdict derived
    from ``cycles_run`` reports the opposite.
    """
    import dataclasses
    from xcquinox.pipeline.cluster.coldstart_census import (
        census_rows, census_solver_config)

    trained = SolverConfig(
        mode=SolverMode.FULL, max_cycles=3, conv_tol=1e-7,
        mixer_name="linear", mixer_kwargs=(("alpha", 0.3),),
        scf_loss_use_tail=True, scf_loss_tail=2, scf_loss_weight_power=1.0,
        seed_source="pbe", seed_cache_dir="/shared/seed_cache",
    )
    census_cfg = census_solver_config(trained)
    assert census_cfg.seed_source == "minao"
    assert census_cfg.seed_cache_dir is None
    for f in dataclasses.fields(trained):
        if f.name in ("seed_source", "seed_cache_dir"):
            continue
        assert getattr(census_cfg, f.name) == getattr(trained, f.name), f.name

    # The rows against the solver's own result, species by species.
    sc = SolverConfig(mode=SolverMode.FULL, max_cycles=3, conv_tol=1e-12)
    spec = _three_species_spec(tmp_path, sc)
    model = _fresh_model(spec)
    rows = census_rows(spec, model=model)

    assert [r["name"] for r in rows] == [m.name for m in spec.molecules]
    oracle = _solver_oracle(spec, model, census_solver_config(sc))
    for row in rows:
        assert set(row) >= _ROW_KEYS, sorted(row)
        want = oracle[row["name"]]
        trace = [float(x) for x in np.asarray(want.energy_trace)]
        assert isinstance(row["converged"], bool)
        assert row["converged"] == bool(want.converged)
        assert isinstance(row["cycles_run"], int)
        assert row["cycles_run"] == int(want.cycles_run)
        assert 0 < row["cycles_run"] <= sc.max_cycles
        assert row["total_energy"] == pytest.approx(float(want.total_energy),
                                                    rel=1e-9)
        assert row["energy_trace"] == pytest.approx(trace, rel=1e-9)
        # a magnitude: the row's own value, unsigned, against the last step
        assert row["last_delta_e"] >= 0.0
        assert row["last_delta_e"] == pytest.approx(
            abs(trace[-1] - trace[-2]), rel=1e-9, abs=1e-14)
        assert row["finite"] is True
        assert row["wall_s"] >= 0.0
        # the record is JSON: no array type survives into the document.
        json.dumps(row)

    # converged is the solver's flag, not a reading of the cycle count: with
    # the freeze off every cycle runs and the flag still latches.
    loose = SolverConfig(mode=SolverMode.FULL, max_cycles=3, conv_tol=1e6,
                         freeze_on_convergence=False)
    loose_spec = _h2_spec(tmp_path, loose, name="loose")
    loose_rows = census_rows(loose_spec, model=_fresh_model(loose_spec))
    assert len(loose_rows) == 1
    assert loose_rows[0]["converged"] is True
    assert loose_rows[0]["cycles_run"] == loose.max_cycles


def test_the_census_cli_writes_the_record(tmp_path):
    """``main`` writes one cell per spec file, in the order given, and returns
    0.

    The two specs differ only in the convergence criterion they carry, so the
    per-cell converged count is an oracle rather than a restatement of the
    rows: the loose spec converges its one species, the strict spec converges
    none.
    """
    from xcquinox.pipeline.cluster import coldstart_census
    from xcquinox.pipeline.cluster.materialize import write_spec_atomic

    loose_spec = _h2_spec(
        tmp_path, SolverConfig(mode=SolverMode.FULL, max_cycles=3,
                               conv_tol=1e6, freeze_on_convergence=False),
        name="loose")
    strict_spec = _h2_spec(
        tmp_path, SolverConfig(mode=SolverMode.FULL, max_cycles=3,
                               conv_tol=1e-12),
        name="strict")
    paths = []
    for tag, spec in (("loose", loose_spec), ("strict", strict_spec)):
        path = os.path.join(str(tmp_path), f"spec_{tag}.spec")
        write_spec_atomic(spec, path)
        paths.append(path)
    out_path = os.path.join(str(tmp_path), "coldstart_census.json")

    assert coldstart_census.main([*paths, "--out", out_path]) == 0

    with open(out_path) as fh:
        doc = json.load(fh)
    cells = doc["cells"]
    assert len(cells) == len(paths)
    assert [c["spec_path"] for c in cells] == paths
    for cell, spec in zip(cells, (loose_spec, strict_spec)):
        assert cell["arch"] == spec.arch.name
        assert cell["n_species"] == len(spec.molecules)
        assert len(cell["rows"]) == cell["n_species"]
        assert [r["name"] for r in cell["rows"]] == [m.name
                                                     for m in spec.molecules]
        for row in cell["rows"]:
            assert set(row) >= _ROW_KEYS, sorted(row)
    assert cells[0]["n_converged"] == 1
    assert cells[1]["n_converged"] == 0


def test_the_document_is_rewritten_after_every_cell(tmp_path):
    """``write_census`` writes the document after each cell, so a census cut
    short at a wall leaves the cells completed: when a cell's summary line
    is logged, the document on disk already holds that cell.

    Oracle: the file read from inside the ``log`` callback, which the writer
    calls once per cell after that cell's write.
    """
    from xcquinox.pipeline.cluster import coldstart_census
    from xcquinox.pipeline.cluster.materialize import write_spec_atomic

    sc = SolverConfig(mode=SolverMode.FULL, max_cycles=2, conv_tol=1e6,
                      freeze_on_convergence=False)
    paths = []
    for tag in ("first", "second"):
        path = os.path.join(str(tmp_path), f"spec_{tag}.spec")
        write_spec_atomic(_h2_spec(tmp_path, sc, name=tag), path)
        paths.append(path)
    out_path = os.path.join(str(tmp_path), "census.json")
    seen = []

    def _log(line):
        with open(out_path) as fh:
            seen.append(len(json.load(fh)["cells"]))

    coldstart_census.write_census(paths, out_path, log=_log)
    assert seen == [1, 2]


def test_a_cell_whose_backend_refuses_the_seed_is_recorded(tmp_path):
    """A FULL cell on the pyscfad backend cannot start from the atomic guess
    (``run_scf`` refuses a non-pbe seed there); the census records the
    refusal as the cell's error, with no rows, and the document is written.

    Oracle: the solver's own refusal, which names the backend. A census that
    ran the trained solver instead of the cold one would report numbers for
    this cell.
    """
    from xcquinox.pipeline.cluster import coldstart_census
    from xcquinox.pipeline.cluster.materialize import write_spec_atomic
    from xcquinox.pipeline.solver import SolverBackend

    sc = SolverConfig(mode=SolverMode.FULL, max_cycles=3,
                      backend=SolverBackend.PYSCFAD)
    path = os.path.join(str(tmp_path), "spec_pyscfad.spec")
    write_spec_atomic(_h2_spec(tmp_path, sc), path)
    out_path = os.path.join(str(tmp_path), "census.json")

    doc = coldstart_census.write_census([path], out_path)

    cell = doc["cells"][0]
    assert cell["rows"] == [] and cell["n_converged"] == 0
    assert cell["n_species"] == 1
    assert cell["error"] is not None
    assert "NotImplementedError" in cell["error"]
    assert "manual-backend only" in cell["error"]
    with open(out_path) as fh:
        assert json.load(fh)["cells"][0]["error"] == cell["error"]


def test_a_diverged_energy_is_written_as_null_and_flagged():
    """A non-finite energy (a diverged SCF) is written as null with the
    row's ``finite`` flag false, and the last step is None when it cannot be
    formed, so the document stays strict JSON.

    Oracle: ``json.dumps(..., allow_nan=False)`` accepting the row, and the
    fields against a fabricated result carrying NaN beside a finite one.
    """
    from types import SimpleNamespace
    from xcquinox.pipeline.cluster.coldstart_census import _row

    diverged = SimpleNamespace(
        converged=np.bool_(False), cycles_run=np.int32(3),
        total_energy=np.float64("nan"),
        energy_trace=np.array([-1.0, -1.5, float("nan")]))
    row = _row("X", diverged, 0.1)
    assert row["total_energy"] is None
    assert row["finite"] is False
    assert row["energy_trace"] == [-1.0, -1.5, None]
    assert row["last_delta_e"] is None
    json.dumps(row, allow_nan=False)

    fine = SimpleNamespace(
        converged=np.bool_(True), cycles_run=np.int32(2),
        total_energy=np.float64(-1.5), energy_trace=np.array([-1.0, -1.5]))
    good = _row("Y", fine, 0.1)
    assert good["finite"] is True
    assert good["total_energy"] == -1.5
    assert good["last_delta_e"] == pytest.approx(0.5)


def test_a_cell_that_cannot_be_read_is_recorded_and_the_next_measured(
        tmp_path):
    """A spec file that cannot be read is recorded as a cell with its error
    and no architecture, and the cells after it are still measured: the
    census is a report, and a report records its failures.

    Oracle: the document's cell order, the failed cell's error text and the
    measured cell's rows.
    """
    from xcquinox.pipeline.cluster import coldstart_census
    from xcquinox.pipeline.cluster.materialize import write_spec_atomic

    good = os.path.join(str(tmp_path), "spec_good.spec")
    write_spec_atomic(
        _h2_spec(tmp_path, SolverConfig(mode=SolverMode.FULL, max_cycles=2)),
        good)
    missing = os.path.join(str(tmp_path), "spec_missing.spec")
    out_path = os.path.join(str(tmp_path), "census.json")

    assert coldstart_census.main([missing, good, "--out", out_path]) == 0

    with open(out_path) as fh:
        cells = json.load(fh)["cells"]
    assert [c["spec_path"] for c in cells] == [missing, good]
    assert cells[0]["arch"] is None and cells[0]["rows"] == []
    assert "FileNotFoundError" in cells[0]["error"]
    assert [r["name"] for r in cells[1]["rows"]] == ["H2"]

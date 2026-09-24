"""Two trained checkpoints of the v7 campaign, loaded and evaluated.

The guard the coordinate and gate fields are added against: a checkpoint
written by an earlier campaign must keep loading into the class its record
names and must keep returning the same numbers, to the last bit. Every field
those runs did not state defaults to what they were -- the legacy uniform-gas
gate, the recorded coordinates -- so a default that moved, or a static field
that reached the leaf stream, shows up here as a changed output or a changed
byte count rather than as a silent reinterpretation of the parameters.

The fixtures beside this module carry, per architecture, the checkpoint, the
model-class record the training stage wrote with it, and the outputs recorded
from it: six ``(rho, sigma, zeta)`` rows spanning four decades of density,
the closed-shell and both fully polarized limits, and the zero-gradient row
where the uniform-gas gate pins both factors at one.
"""
import dataclasses
import json
import pathlib

import equinox as eqx
import jax.numpy as jnp
import pytest

from xcquinox.pipeline.checkpoint_class import (model_class_of_arch,
                                            require_matching_class)
from xcquinox.pipeline.cluster.grid_config import ModelConfig
from xcquinox.pipeline.config import apply_model_block, get_architecture
from xcquinox.pipeline.models import AlecGGAModel


_FIXTURES = (pathlib.Path(__file__).resolve().parent / "fixtures"
             / "v7_checkpoints")
_ARCHS = ("deep_3x16", "deep_attn_3x16")


def _fixture(name):
    """The checkpoint path, its class record and its recorded outputs."""
    directory = _FIXTURES / name
    checkpoint = directory / "model.eqx"
    record = json.loads((directory / "model.eqx.class.json").read_text())
    outputs = json.loads((directory / "outputs.json").read_text())
    return checkpoint, record, outputs


@pytest.mark.parametrize("name", _ARCHS)
def test_the_v7_checkpoints_load_and_reproduce_their_recorded_outputs(
        name, tmp_path):
    """A v7 checkpoint loads into the class its record names and returns the
    recorded enhancement factors exactly.

    Oracle: ``outputs.json`` beside each checkpoint -- the values the loaded
    model produced on those rows when the fixture was recorded -- compared
    with ``==`` rather than a tolerance, and the checkpoint's own bytes,
    compared against a re-serialization of the loaded model. The architecture
    is rebuilt the way the recording did: the registry entry, the polarized
    flag the record states, and the run's model block.
    """
    checkpoint, record, outputs = _fixture(name)
    assert checkpoint.is_file(), checkpoint

    arch = dataclasses.replace(
        get_architecture(name),
        use_polarized_correlation=bool(outputs["use_polarized_correlation"]))
    arch = apply_model_block(arch, ModelConfig(
        parent_anchor=bool(record["parent_anchor"]),
        descriptor_coordinates=str(record["descriptor_coordinates"])))

    skeleton = AlecGGAModel.from_arch(arch, seed=0)
    require_matching_class(str(checkpoint), model_class_of_arch(arch))
    model = eqx.tree_deserialise_leaves(str(checkpoint), skeleton)

    for row, fx, fc in zip(outputs["rows"], outputs["fx"], outputs["fc"]):
        rho, sigma, zeta = (float(v) for v in row)
        got_fx = float(model.xnet(jnp.array([rho, sigma])).squeeze())
        got_fc = float(model.cnet(jnp.array([rho, sigma, zeta])).squeeze())
        assert got_fx == fx, (name, row, got_fx, fx)
        assert got_fc == fc, (name, row, got_fc, fc)

    round_trip = tmp_path / "model.eqx"
    eqx.tree_serialise_leaves(str(round_trip), model)
    assert round_trip.read_bytes() == checkpoint.read_bytes()


"""The exchange network with the gradient-expansion coefficient fixed by
construction (``ArchitectureConfig.gea_mu``, ``networks.AlecGGA_XNet``).

Under the published clone's gate the gated output is O(s^3), so the clone has
no s^2 term; with the field the pre-image gains mu a/(a - 1) (1 - e^{-s^2}),
a being the bounded map's limit, and F_x = 1 + mu s^2 + O(s^3) for every
network state. The oracles: the curvature at s -> 0 against 2 mu by two
routes, the closed form with the network at zero, the interface contract of
the field, the registry entry against its twin, the class record and the
loaders, and the submit-time check of a run's model block.
"""
from __future__ import annotations

import dataclasses
import os
import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.pipeline import config as C
from xcquinox.pipeline import networks as N
from xcquinox.pipeline.gradient_expansion import MU_GE, PBE_MU, resolve
from xcquinox.pipeline.parents import pbe_fx

_A = 1.804
_LP0 = (_A - 1.0) / _A
#: Relative tolerance on 2 mu. F''(s) = 2 mu + 6 L'(0) net0 s + ... under the
#: x2 gate (net0 the MLP at input 0), so the autodiff route reads a bias of
#: 7.4e-7 at s0 = 1e-6 on seed 2 (1.7e-6 relative) and at most 3.0e-9
#: relative at s0 = 1e-9; the Richardson route reads at most 4.7e-9.
_RTOL = 1e-6
_S0 = 1e-9
_S_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0)
#: 1 + L(mu' (1 - e^{-s^2})) at _S_GRID for mu = PBE's: the form saturates at
#: 1 + L(mu') = 1.2096825, against PBE's own F_x(1) = 1.172435.
_CLOSED_PBE = (1.0, 1.002183624746416, 1.048223433467951, 1.135378368274788,
               1.206081823471181, 1.209682492472494, 1.209682514562016)


def _kf(rho):
    return (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)


def _fx_of_s(xnet, rho):
    """F_x as a function of s at fixed rho: sigma = (2 k_F rho s)^2."""
    return lambda s: xnet(jnp.array([rho, (2.0 * _kf(rho) * rho * s) ** 2]))


def _closed(s, mu):
    """1 + L(mu' (1 - e^{-s^2})) in plain floating point."""
    x = mu * _A / (_A - 1.0) * (1.0 - math.exp(-s * s))
    return _A / (1.0 + math.exp(-(x - math.log(_A - 1.0))))


def _curvature(xnet, rho):
    """``(autodiff at s0, Richardson)`` estimates of F''(0). sqrt(sigma) is
    not differentiable at sigma = 0, so the autodiff route differentiates at
    s0; the difference D(h) = (F(2h) - 2F(h) + F(0))/h^2 = 2c2 + 6c3 h + ...
    is extrapolated three times at ratio 2 from h = 0.01."""
    f = _fx_of_s(xnet, rho)
    auto = float(jax.grad(jax.grad(f))(_S0))
    f0 = float(f(0.0))
    row = [(float(f(2 * h)) - 2 * float(f(h)) + f0) / h ** 2
           for h in (0.01 / 2 ** i for i in range(4))]
    for k in (1, 2, 3):
        row = [(2 ** k * row[i + 1] - row[i]) / (2 ** k - 1)
               for i in range(len(row) - 1)]
    return auto, row[0]


def _xnet(arch, seed):
    return N.create_network_pair(arch, seed=seed)[0]


def _plain_x2():
    """deep_3x16 under the x2 gate: deep_gea_3x16 without the coefficient."""
    return dataclasses.replace(C.get_architecture("deep_3x16"), ueg_gate="x2")


def test_the_curvature_at_the_uniform_gas_is_the_coefficient():
    """d^2 F_x / ds^2 at s -> 0 is 2 mu to 1e-6 relative by both routes, seeds
    0 to 2, rho 0.1 and 1.0: PBE's mu for the registry entry, 10/81 for
    ``gea_mu="gea"``, 0.3 for a number. The control, deep_3x16 under the x2
    gate, reads 0 (at most 7.4e-10); under tanh2 it reads 2 L'(0) net0, the
    network's value at s = 0 (6.35e-2, -3.41e-2 and 2.46e-1 at seeds 0 to 2)."""
    for seed in (0, 1, 2):
        for rho in (0.1, 1.0):
            for value in _curvature(_xnet(_plain_x2(), seed), rho):
                assert abs(value) < _RTOL * 2 * PBE_MU, (seed, rho, value)
            t2net = _xnet(C.get_architecture("deep_3x16"), seed)
            want = 2 * _LP0 * float(t2net.net(jnp.array([0.0]))[0])
            for value in _curvature(t2net, rho):
                assert value == pytest.approx(want, rel=_RTOL), (seed, rho)
    cases = [(C.get_architecture("deep_gea_3x16"), PBE_MU)]
    for name, mu in (("gea", MU_GE), (0.3, 0.3)):
        cases.append((C.ArchitectureConfig.from_spec(
            "deep_gea_t_3x16", 3, 16, dm_entropy_intensive=True,
            descriptor_log_transform=True, ueg_gate="x2", gea_mu=name), mu))
    for arch, mu in cases:
        assert arch.resolved_gea_mu == mu
        for seed in (0, 1, 2):
            for rho in (0.1, 1.0):
                for value in _curvature(_xnet(arch, seed), rho):
                    assert value == pytest.approx(2 * mu, rel=_RTOL), (
                        arch.gea_mu, seed, rho, value)


def test_the_fixed_term_is_the_damped_quadratic():
    """With the final layer zeroed F_x = 1 + L(mu' (1 - e^{-s^2})) to 1e-12 at
    s in {0, 0.1, 0.5, 1, 2, 4, 10}, below a everywhere and not PBE (1.135378
    against 1.172435 at s = 1); with the network live F_x(1) is the closed
    form of x2 net(x2) + mu' (1 - e^{-1}) through the map, and the network's
    share of it exceeds 1e-3 at every seed (1.2e-2, -8.3e-3, 4.8e-2)."""
    for value, s in zip(_CLOSED_PBE, _S_GRID):
        assert abs(_closed(s, PBE_MU) - value) < 1e-15, s
    gea = C.get_architecture("deep_gea_3x16")
    xnet = _xnet(dataclasses.replace(gea, zero_init_final_layer=True), 0)
    for rho in (0.1, 1.0):
        f = _fx_of_s(xnet, rho)
        for s, want in zip(_S_GRID, _CLOSED_PBE):
            assert abs(float(f(s)) - want) < 1e-12, (rho, s, float(f(s)))
            assert float(f(s)) < _A
    pbe_1 = float(pbe_fx(1.0, (2.0 * _kf(1.0)) ** 2))
    assert abs(pbe_1 - 1.172435228403129) < 1e-12
    assert abs(float(_fx_of_s(xnet, 1.0)(1.0)) - pbe_1) > 1e-2
    x2 = (1.0 - math.exp(-1.0)) * math.log(2.0)
    fixed = PBE_MU * _A / (_A - 1.0) * (1.0 - math.exp(-1.0))
    for seed in (0, 1, 2):
        live = _xnet(gea, seed)
        f1 = float(_fx_of_s(live, 1.0)(1.0))
        net = float(live.net(jnp.array([x2]))[0])
        by_hand = _A / (1.0 + math.exp(-(x2 * net + fixed - math.log(_A - 1.0))))
        assert abs(f1 - by_hand) < 1e-13, (seed, f1, by_hand)
        assert abs(f1 - _closed(1.0, PBE_MU)) > 1e-3, seed


def test_the_registry_entry_is_deep_3x16_with_the_coefficient_and_the_gate():
    """deep_3x16 field for field plus the coefficient and the gate it
    requires; the shown name is its own; the GGA rung; the expanded key names
    the coefficient; the leaves are deep_3x16's (no parameter is added) and a
    serialized pair loads into a skeleton of another seed bitwise."""
    from xcquinox.pipeline import arch_names, rungs
    entry = C.ARCHITECTURES["deep_gea_3x16"]
    twin = dataclasses.replace(C.ARCHITECTURES["deep_3x16"],
                               name="deep_gea_3x16", gea_mu="pbe", ueg_gate="x2")
    for f in dataclasses.fields(C.ArchitectureConfig):
        assert getattr(entry, f.name) == getattr(twin, f.name), f.name
    assert entry.resolved_gea_mu == PBE_MU == resolve("pbe")
    assert arch_names.DISPLAY_NAME["deep_gea_3x16"] == "deep_gea_3x16"
    assert rungs.rung_of("deep_gea_3x16") == rungs.RUNG_GGA
    assert "gradient expansion" in arch_names.expanded_key("deep_gea_3x16")
    assert "gradient expansion" not in arch_names.expanded_key("deep_3x16")
    xg, cg = N.create_network_pair(entry, seed=0)
    xp, cp = N.create_network_pair(_plain_x2(), seed=0)
    assert xg.gea_mu == PBE_MU and getattr(cg, "gea_mu", None) is None
    for a, b in ((xg, xp), (cg, cp)):
        la = jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array))
        lb = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))
        assert len(la) == len(lb) == 8
        assert all(np.array_equal(np.asarray(u), np.asarray(v))
                   for u, v in zip(la, lb))
    row = jnp.array([1.0, 0.5])
    assert float(xg(row)) != float(xp(row))
    assert float(cg(jnp.array([1.0, 0.5, 0.0]))) == float(cp(jnp.array([1.0, 0.5, 0.0])))


def test_the_field_is_validated():
    """ValueError naming the field for a coefficient that is neither a table
    name nor a positive finite number, the tanh2 gate, the meta-GGA rung, a
    direct construction with the parent anchor, a run's model block resolving
    the gate to tanh2, and the ueg_limit and lieb_oxford exchange constraints
    (they wrap the forward: ueg_limit zeroes the curvature, lieb_oxford with
    the double clamp keeps L'(0) of it); the same in the network's
    constructor, with lob_lim None besides. The anchor protocol drops the
    term instead: ``anchored`` and a block with the anchor return the
    architecture with ``gea_mu`` None."""
    from xcquinox.pipeline.cluster.grid_config import ModelConfig
    from xcquinox.pipeline.constraints import make_constraint
    for bad in (0.0, -0.1, float("nan"), float("inf"), "PBE", "lda", "", True):
        with pytest.raises(ValueError, match="gea_mu"):
            C.ArchitectureConfig.from_spec("t", 3, 16, ueg_gate="x2", gea_mu=bad)
    with pytest.raises(ValueError, match="gea_mu"):
        C.ArchitectureConfig.from_spec("t", 3, 16, descriptors=["metagga"],
                                       meta_gga=True, ueg_gate="x2", gea_mu="pbe")
    for constraint, extra in (("ueg_limit", {}), ("lieb_oxford", {}),
                              ("lieb_oxford", {"allow_double_lob_clamp": True})):
        with pytest.raises(ValueError, match="gea_mu"):
            C.ArchitectureConfig.from_spec(
                "t", 3, 16, x_constraints=[constraint], ueg_gate="x2",
                gea_mu="pbe", **extra)
    gea = C.get_architecture("deep_gea_3x16")
    for change in (dict(ueg_gate="tanh2"), dict(parent_anchor=True)):
        with pytest.raises(ValueError, match="gea_mu"):
            dataclasses.replace(gea, **change)
    dropped = C.anchored(gea)
    assert dropped.parent_anchor and dropped.resolved_gea_mu is None
    with pytest.raises(ValueError, match="gea_mu"):
        C.apply_model_block(gea, ModelConfig(ueg_gate="tanh2"))
    through = C.apply_model_block(gea, ModelConfig(parent_anchor=True,
                                                   ueg_gate="x2"))
    assert through.parent_anchor and through.gea_mu is None
    assert C.apply_model_block(gea, ModelConfig(ueg_gate="x2")).gea_mu == "pbe"
    for kw in (dict(ueg_gate="tanh2"), dict(meta_gga=True),
               dict(parent="pbe", ueg_gate="x2"),
               dict(lob_lim=None, ueg_gate="x2"),
               dict(ueg_gate="x2", constraints=(make_constraint("ueg_limit"),)),
               dict(ueg_gate="x2", constraints=(make_constraint("lieb_oxford"),))):
        with pytest.raises(ValueError, match="gea_mu"):
            N.AlecGGA_XNet(n_extra_features=0, depth=3, nodes=16,
                           gea_mu=PBE_MU, **kw)
    for bad in (0.0, -1.0, float("nan"), "pbe", True):
        with pytest.raises(ValueError, match="gea_mu"):
            N.AlecGGA_XNet(n_extra_features=0, depth=3, nodes=16,
                           ueg_gate="x2", gea_mu=bad)


def test_the_class_record_and_the_loaders_state_the_coefficient(tmp_path):
    """The resolved number rides the class record (read off the architecture
    and off the built model) and the pretraining metadata; the loaders refuse
    the other value. A record that states no coefficient states a network
    without the term: accepted by deep_3x16's class, refused by
    deep_gea_3x16's, whose leaves it would otherwise fill."""
    from xcquinox.pipeline import checkpoint_class as K
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.pretrain import _metadata_preflight
    from xcquinox.pipeline.train import _require_matching_model_class
    plain, gea = _plain_x2(), C.get_architecture("deep_gea_3x16")
    want_plain, want_gea = K.model_class_of_arch(plain), K.model_class_of_arch(gea)
    assert want_plain["gea_mu"] is None and want_gea["gea_mu"] == PBE_MU
    for arch, want in ((gea, want_gea), (plain, want_plain)):
        assert K.model_class_of_model(AlecGGAModel.from_arch(arch, seed=0)) == want
    paths = {}
    for label, arch in (("gea", gea), ("plain", plain)):
        paths[label] = str(tmp_path / f"{label}.eqx")
        model = AlecGGAModel.from_arch(arch, seed=0)
        eqx.tree_serialise_leaves(paths[label], model)
        K.write_class_record(paths[label], arch)
    assert K.read_class_record(paths["gea"])["gea_mu"] == PBE_MU
    K.require_matching_class(paths["gea"], want_gea)
    K.require_matching_class(paths["plain"], want_plain)
    for path, want in ((paths["gea"], want_plain), (paths["plain"], want_gea)):
        with pytest.raises(K.ModelClassMismatch, match="gea_mu"):
            K.require_matching_class(path, want)
    record_path = Path(K.class_record_path(paths["plain"]))
    record = json.loads(record_path.read_text())
    record.pop("gea_mu")
    record_path.write_text(json.dumps(record))
    K.require_matching_class(paths["plain"], want_plain)
    with pytest.raises(K.ModelClassMismatch, match="gea_mu"):
        K.require_matching_class(paths["plain"], want_gea)
    base = {"depth": 3, "nodes": 16, "parent_anchor": False,
            "descriptor_coordinates": "legacy", "ueg_gate": "x2"}
    hand = tmp_path / "hand"
    hand.mkdir()
    md_path = hand / "pretrain_metadata.json"
    for metadata, accepted, refused in ((base, plain, gea),
                                        (dict(base, gea_mu=MU_GE), None, gea),
                                        (dict(base, gea_mu=PBE_MU), gea, plain)):
        md_path.write_text(json.dumps(metadata))
        if accepted is not None:
            _require_matching_model_class(str(hand), accepted)
            _metadata_preflight(metadata_path=str(md_path), arch=accepted)
        with pytest.raises(ValueError, match="gea_mu"):
            _require_matching_model_class(str(hand), refused)
        with pytest.raises(ValueError, match="gea_mu"):
            _metadata_preflight(metadata_path=str(md_path), arch=refused)


def test_the_submit_time_check_refuses_a_block_that_undoes_the_gate():
    """``validate_grid_semantics`` applies the run's model block to every
    swept architecture on the login node: a block leaving the gate at the
    parser's default refuses deep_gea_3x16 by name; a block stating the x2
    gate passes; a plain architecture passes under the default block."""
    from xcquinox.pipeline.cluster.grid_config import (ModelConfig,
                                                        validate_grid_semantics)
    from xcquinox.pipeline.tests.test_cluster_grid_config import (_StubDomain,
                                                                   _cfg)
    domain = _StubDomain(pool_size=40)
    validate_grid_semantics(_cfg(arch=("deep_3x16",)), domain)
    gea_cfg = _cfg(arch=("deep_3x16", "deep_gea_3x16"))
    with pytest.raises(ValueError, match="deep_gea_3x16") as excinfo:
        validate_grid_semantics(gea_cfg, domain)
    assert "gea_mu" in str(excinfo.value)
    validate_grid_semantics(
        dataclasses.replace(gea_cfg, model=ModelConfig(ueg_gate="x2")), domain)


@pytest.mark.slow
def test_the_run_readers_accept_the_run_they_build(tmp_path, monkeypatch):
    """The certificate stamp, the certificate keep check
    (``model_class_mismatches``, ``certificate_describes_run``,
    ``completed_pretraining``) and ``validate_run`` compare the coefficient
    of the architecture the run builds, the registry entry under the run's
    model block: an anchored run drops the term and its products state
    None, so none of its readers reports a disagreement and the run is kept;
    an unanchored run stamps PBE's mu and is refused by the gate alone after
    two pretraining steps; a certificate or metadata tampered to the other
    value is reported by the keep check and the validator."""
    import pickle
    from types import SimpleNamespace
    from xcquinox.pipeline.cluster import fidelity as fid
    from xcquinox.pipeline.cluster import validate_run as vr
    from xcquinox.pipeline.cluster._pretrain import (completed_pretraining,
                                                     resolve_run_architecture)
    from xcquinox.pipeline.cluster.grid_config import pretrain_checkpoint_dir
    from xcquinox.pipeline.pretrain import run_pretrain
    from xcquinox.pipeline.pretrain_data_gen import generate_pretrain_data_npz
    from xcquinox.pipeline.tests.test_parent_anchor import (_anchored_cfg,
                                                             _tiny_oracle_set)
    from xcquinox.pipeline.tests.test_validate_run import _cfg as _vr_cfg
    from xcquinox.pipeline.tests.test_validate_run import _spec_for
    name = "deep_gea_3x16"
    data = tmp_path / "data"
    data.mkdir()
    generate_pretrain_data_npz(str(data), atoms=(("He", 0),), basis="sto-3g",
                               grid_level=0, polarized=True, descriptors=True,
                               density_fit=False)
    for anchored in (True, False):
        cfg = _anchored_cfg(arch=(name,))
        cfg.model.ueg_gate = "x2"
        cfg.model.parent_anchor = anchored
        arch = resolve_run_architecture(cfg, C.get_architecture(name))
        want = None if anchored else PBE_MU
        assert arch.parent_anchor is anchored and arch.resolved_gea_mu == want
        run_dir = str(tmp_path / ("anchored" if anchored else "plain"))
        pre = pretrain_checkpoint_dir(run_dir, name)
        md = run_pretrain(C.PretrainSpec(arch=arch, data_dir=str(data),
                                         checkpoint_dir=pre, n_steps=2, seed=0))
        payload = fid.fidelity_certificate(cfg, run_dir, name,
                                           oracle_set=_tiny_oracle_set())
        assert md["gea_mu"] == payload["gea_mu"] == want
        assert fid.model_class_mismatches(cfg, payload, name) == []
        assert not any("gea_mu" in line for line in
                       fid.certificate_describes_run(cfg, pre, name, payload))
        keep, reason = completed_pretraining(pre, cfg, name)
        assert "gea_mu" not in reason, (anchored, reason)
        if anchored:
            assert keep is True, reason
        else:
            assert payload["verdict"] == "FAIL" and keep is False, reason
        tampered = dict(payload, gea_mu=(PBE_MU if anchored else None))
        assert [m[0] for m in fid.model_class_mismatches(cfg, tampered, name)
                ] == ["gea_mu"]
        vcfg = _vr_cfg()
        vcfg.sweep.arch = (name,)
        vcfg.inputs.basis = "def2-svp"
        vcfg.inputs.orientation_lock_strength = 0.0
        vcfg.model = cfg.model
        vcfg.pretrain = SimpleNamespace(n_steps=2)
        monkeypatch.setattr(vr, "load_grid_config", lambda path, _c=vcfg: _c)
        spec = _spec_for(name, basis="def2-svp", arch_override=arch)
        os.makedirs(os.path.join(run_dir, "specs"), exist_ok=True)
        with open(os.path.join(run_dir, "specs", "spec_0000.spec"), "wb") as f:
            pickle.dump(spec, f)
        Path(run_dir, "resolved_config.yaml").write_text("placeholder: true\\n")
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump({"width": 4,
                       "xcquinox_version": payload.get("xcquinox_version")}, f)
        failures, warnings, _n = vr.validate_run(run_dir)
        assert not any("gea_mu" in line for line in failures + warnings), (
            anchored, failures, warnings)
        md_path = Path(pre, "pretrain_metadata.json")
        md_path.write_text(json.dumps(dict(json.loads(md_path.read_text()),
                                           gea_mu=(PBE_MU if anchored else None))))
        failures, _warnings, _n = vr.validate_run(run_dir)
        assert any("metadata gea_mu" in line for line in failures), (anchored,
                                                                       failures)

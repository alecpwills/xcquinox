"""The published clone's reference forms and the ``paper`` exchange footing.

The published clone (gitlab.com/saru1799/xcquinox-clone, branch ``public``)
forms its pretraining targets from analytic expressions of its own rather than
from libxc: ``reference_functionals.py`` carries ``__Fx_PBE_unpol``,
``__pw92_eps_c_wospin_point``, ``__Fc_PBE_wospin`` and ``eUEG_LDA_x_unpol``,
all posed on the TOTAL density for every system, open shells included. Its
PW92 amplitudes ``A = (0.031090690869654895, 0.015545, 0.016887)`` are neither
libxc's ``LDA_C_PW_MOD`` set, which the PBE correlation numerator uses, nor
``LDA_C_PW``, which the stored ratio of the ``total`` footing divides by, so
the correlation targets of the ``paper`` footing are not a restatement of a
libxc ratio.

The oracle below is those four expressions typed out again in NumPy from the
published source, constant by constant, so what is pinned is the published
formula rather than a second reading of whatever the library computes. The
file cases then hold a generated ``.npz`` to the same oracle row by row.
"""
import numpy as np
import pytest

import xcquinox.pipeline.pretrain_data_gen as pdg
from xcquinox.pipeline import parents


# ---------------------------------------------------------------------------
# The published forms, typed from reference_functionals.py
# ---------------------------------------------------------------------------

#: ``__pw92_eps_c_wospin_point``: A, ALPHA1, BETA1..BETA4, in that function's
#: own order (the unpolarized, polarized and spin-stiffness branches).
_A = (0.031090690869654895, 0.015545, 0.016887)
#: libxc's ``LDA_C_PW_MOD`` amplitudes, the PBE correlation numerator's.
_A_PW_MOD = (0.0310907, 0.01554535, 0.0168869)
#: libxc's ``LDA_C_PW`` amplitudes: the denominator the ``total`` footing
#: divides its stored ratio by (``pretrain_data_gen`` calls ``",LDA_C_PW"``).
_A_PW = (0.031091, 0.015545, 0.016887)
_ALPHA1 = (0.21370, 0.20548, 0.11125)
_BETA1 = (7.5957, 14.1189, 10.357)
_BETA2 = (3.5876, 6.1977, 3.6231)
_BETA3 = (1.6382, 3.3662, 0.88026)
_BETA4 = (0.49294, 0.62517, 0.49671)

#: ``_PBE_KAPPA``, ``_PBE_MU``, ``_PBE_BETA`` of the published module. ``mu``
#: is a literal there, not ``beta pi^2 / 3``.
_KAPPA = 0.804
_MU = 0.2195149727645171
_BETA = 0.06672455060314922

#: The absolute band beside the relative tolerance where enhancement FACTORS
#: are compared. Both sides form ``1 + x`` and the file stores ``x`` back by
#: a subtraction, so where the factor sits close to one (small ``s``: F_x - 1
#: of order 1e-5) or close to zero (large ``t``: F_c of order 1e-4) the
#: relative difference of two values that agree to the last bit of one
#: exceeds any relative tolerance. Eight ulp of one sits just above the
#: residue measured and near the bottom of the window the separation from the
#: libxc amplitude sets leaves open, that separation binding at 287 ulp: over
#: the 300 oracle rows the two implementations of the factors disagree by 1.0
#: ulp of one (F_x) and 1.5 ulp (F_c), and over 1600000 log-uniform draws
#: inside the protocol's density box, rho in [1e-4, 10], by at most 5.0 ulp
#: with no row past the band. The residue belongs to that box rather than to
#: the formulas: drawn with rho and sigma independent down to rho 1e-6 the
#: worst reaches 10.5 ulp and 16 rows in 200000 pass the band, and down to
#: 1e-12, 412.5 ulp. The generated file the second comparison below reads
#: runs five decades under the box floor -- 248 of its 1200 rows, down to rho
#: 1.5267e-09 -- and stays inside the band, a physical grid correlating sigma
#: with rho where an independent draw does not; that file's own worst is
#: asserted there rather than assumed here.
_ONE_ULP_BAND = 8.0 * float(np.finfo(np.float64).eps)


def _eps_c(rho, zeta, amplitudes=_A):
    """``__pw92_eps_c_wospin_point`` in NumPy, expression for expression, at a
    chosen amplitude set (the published one by default)."""
    rho = np.asarray(rho, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1 / 3)
    g = []
    for k in range(3):
        b = (_BETA1[k] * np.sqrt(rs) + _BETA2[k] * rs
             + _BETA3[k] * rs ** 1.5 + _BETA4[k] * rs ** 2)
        c = 1 + 1 / (2 * amplitudes[k] * b)
        g.append(-2 * amplitudes[k] * (1 + _ALPHA1[k] * rs) * np.log(c))
    c0 = 1.0 / (2.0 ** (4 / 3) - 2.0)
    f = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2.0) * c0
    fpp_0 = c0 * 8.0 / 9.0
    g0, g1, g2 = g
    return g0 - g2 * f / fpp_0 * (1 - zeta ** 4) + (g1 - g0) * f * zeta ** 4


def _published_s(rho, grad):
    """``S``: ``|grad rho| / (2 k_F rho + 1e-30)``, ``k_F = (3 pi^2 rho)^(1/3)``."""
    rho = np.asarray(rho, dtype=np.float64)
    k_f = (3 * np.pi ** 2 * rho) ** (1 / 3)
    return np.asarray(grad, dtype=np.float64) / (2 * k_f * rho + 1e-30)


def _published_fx(rho, grad):
    """``__Fx_PBE_unpol``: ``1 + kappa - kappa / (1 + mu s^2 / kappa)``."""
    s = _published_s(rho, grad)
    return 1 + _KAPPA - _KAPPA / (1 + _MU * s ** 2 / _KAPPA)


def _published_fc(rho, zeta, grad):
    """``__Fc_PBE_wospin``: PBE's ``H`` over the published PW92 ``eps_c``."""
    rho = np.asarray(rho, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    scaling_pol = 0.5 * ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3))
    scaling_pol_3 = scaling_pol ** 3
    k_f = (3 * np.pi ** 2 * rho) ** (1 / 3)
    k_s = np.sqrt((4 * k_f) / np.pi)
    t = np.abs(grad) / (2 * k_s * scaling_pol * rho)
    gamma = (1 - np.log(2)) / (np.pi ** 2)
    e_heg_c = _eps_c(rho, zeta)
    a = (_BETA / gamma) / (np.exp(-e_heg_c / (gamma * scaling_pol_3)) - 1)
    h = gamma * np.log(1 + (_BETA / gamma) * t ** 2
                       * ((1 + a * t ** 2) / (1 + a * t ** 2 + a ** 2 * t ** 4)))
    h = h * scaling_pol_3
    return 1 + (h / e_heg_c)


def _published_lda_x_eps(rho):
    """``eUEG_LDA_x_unpol``: ``-3/4 (3/pi)^(1/3) rho^(1/3)``."""
    rho = np.asarray(rho, dtype=np.float64)
    return -3 / 4 * (3 / np.pi) ** (1 / 3) * rho ** (1 / 3)


def _oracle_rows(n=300, seed=20260923):
    """``(rho, sigma, zeta)`` over the published protocol's range, with the
    three exact polarizations placed on named rows so every branch of the spin
    interpolation is exercised."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(1e-4, 10.0, size=n)
    sigma = rng.uniform(0.0, 100.0, size=n)
    zeta = rng.uniform(-1.0, 1.0, size=n)
    for base in (0, n // 2):
        zeta[base + 0] = 0.0
        zeta[base + 1] = 1.0
        zeta[base + 2] = -1.0
    return rho, sigma, zeta


# ---------------------------------------------------------------------------
# The forms
# ---------------------------------------------------------------------------

def test_the_paper_targets_equal_the_published_formulas():
    """``parents.paper_*`` reproduce the published clone's four reference
    forms.

    Oracle: ``_eps_c`` / ``_published_fx`` / ``_published_fc`` /
    ``_published_lda_x_eps`` above, typed from
    ``reference_functionals.__pw92_eps_c_wospin_point``, ``__Fx_PBE_unpol``,
    ``__Fc_PBE_wospin`` and ``eUEG_LDA_x_unpol``, on 300 rows of a fixed
    stream. The gradient magnitude the published forms take is
    ``sqrt(sigma)``. A relative tolerance of 1e-12 separates the published
    amplitudes from either libxc set (1.8e-7 against ``LDA_C_PW_MOD``, 6.0e-6
    against ``LDA_C_PW``) while admitting a difference of summation order.
    """
    rho, sigma, zeta = _oracle_rows()
    grad = np.sqrt(sigma)

    np.testing.assert_allclose(
        np.asarray(parents.paper_lda_x_eps(rho)),
        _published_lda_x_eps(rho), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(parents.paper_pw92_eps_c(rho, zeta)),
        _eps_c(rho, zeta), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(parents.paper_pbe_fx(rho, sigma)),
        _published_fx(rho, grad), rtol=1e-12, atol=_ONE_ULP_BAND)
    np.testing.assert_allclose(
        np.asarray(parents.paper_pbe_fc(rho, sigma, zeta)),
        _published_fc(rho, zeta, grad), rtol=1e-12, atol=_ONE_ULP_BAND)


def test_the_value_sets_are_stated_once():
    """The library and the login-node parser admit the same values.

    Each string knob of the pretraining protocol is spelled in two modules --
    the library, which pulls JAX, and the harness parser, which must not -- so
    a value one layer admits and the other refuses could otherwise ship.
    Oracle: the module constants themselves, compared pair by pair.
    """
    from xcquinox.pipeline import config as pipeline_config
    from xcquinox.pipeline.cluster import grid_config

    assert pdg.EXCHANGE_FOOTINGS == grid_config._EXCHANGE_FOOTINGS
    assert pipeline_config.UEG_GATES == grid_config._UEG_GATES
    assert (pipeline_config.DESCRIPTOR_COORDINATES
            == grid_config._DESCRIPTOR_COORDINATES)
    assert pipeline_config.PARENT_DENSITIES == grid_config._PARENT_DENSITIES
    assert pipeline_config.MAX_SEED == grid_config._MAX_SEED
    assert "paper" in pdg.EXCHANGE_FOOTINGS
    assert "paper" in pipeline_config.DESCRIPTOR_COORDINATES


# ---------------------------------------------------------------------------
# The file the footing writes
# ---------------------------------------------------------------------------

#: The two-system pretraining set of the schema cases: one closed shell and
#: one fully polarized open shell, the smallest pair that separates a
#: total-density footing from a spin-resolved one.
_TINY = (("He", 0), ("H", 1))


def _gen(tmp_path, **kw):
    """A pretrain ``.npz`` over ``_TINY`` at sto-3g, grid level 0, polarized,
    with descriptors: the configuration ``test_pretrain_schema`` generates."""
    kw.setdefault("atoms", _TINY)
    kw.setdefault("basis", "sto-3g")
    kw.setdefault("grid_level", 0)
    kw.setdefault("polarized", True)
    kw.setdefault("descriptors", True)
    path = pdg.generate_pretrain_data_npz(str(tmp_path), **kw)
    with np.load(path) as z:
        return path, {k: np.array(z[k]) for k in z.files}


def _system_index(path, symbol):
    """The row index the manifest gives ``symbol`` in the generated file."""
    manifest = pdg.read_pretrain_manifest(path)
    names = [str(row[0]) for row in manifest["atoms"]]
    return names.index(symbol)


def test_a_paper_file_carries_the_published_targets_for_every_system(tmp_path):
    """Under ``exchange_footing: paper`` every total-density row of every
    system carries the published targets, the open shell's included.

    Oracle: the published forms above, evaluated on the file's own
    ``rho_all``, ``sigma_all`` and ``zeta_all``. The file stores ``F - 1``, so
    the stored column plus one is the enhancement factor; the LDA columns are
    the published energy densities ``rho eps``. The targets are checked to sit
    inside the published forms' own ranges, which are far inside the
    generator's +-5 clip, so no row is pinned at a clipped value; the key set
    is the ``total`` footing's, the published protocol posing no per-channel
    exchange block. The two factors carry the absolute band of
    ``_ONE_ULP_BAND`` beside the relative tolerance; the LDA columns, which
    cancel nothing, are held to the relative tolerance alone.
    """
    path, cols = _gen(tmp_path, exchange_footing="paper")

    rho = cols["rho_all"]
    sigma = cols["sigma_all"]
    zeta = cols["zeta_all"]
    grad = np.sqrt(sigma)

    np.testing.assert_allclose(cols["Fx_all"] + 1.0,
                               _published_fx(rho, grad), rtol=1e-12,
                               atol=_ONE_ULP_BAND)
    np.testing.assert_allclose(cols["Fc_all"] + 1.0,
                               _published_fc(rho, zeta, grad),
                               rtol=1e-12, atol=_ONE_ULP_BAND)
    np.testing.assert_allclose(cols["e_lda_x_all"],
                               rho * _published_lda_x_eps(rho),
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(cols["e_lda_c_all"],
                               rho * _eps_c(rho, zeta), rtol=1e-12, atol=0.0)

    # This file's rows run five decades below the density box the band was
    # measured over, so its own worst disagreement is recorded against the
    # band here rather than taken on trust from that measurement.
    eps = float(np.finfo(np.float64).eps)
    assert float(rho.min()) < 1e-4
    below = int((rho < 1e-4).sum())
    assert below > 0
    worst = max(
        float(np.max(np.abs(cols["Fx_all"] + 1.0 - _published_fx(rho, grad)))),
        float(np.max(np.abs(cols["Fc_all"] + 1.0
                            - _published_fc(rho, zeta, grad)))))
    assert worst < _ONE_ULP_BAND, (worst / eps, below, float(rho.min()))

    # The open shell is present and carries the same form as the closed one.
    h_rows = cols["system_all"] == _system_index(path, "H")
    assert int(h_rows.sum()) > 0
    np.testing.assert_allclose(cols["Fx_all"][h_rows] + 1.0,
                               _published_fx(rho[h_rows], grad[h_rows]),
                               rtol=1e-12, atol=0.0)

    # No row sits on the clip: F_x - 1 lies in [0, kappa] and F_c - 1 in
    # [-1, 0] for the published forms, both far inside +-5.
    assert float(cols["Fx_all"].min()) >= 0.0
    assert float(cols["Fx_all"].max()) <= _KAPPA
    assert float(cols["Fc_all"].max()) <= 0.0
    assert float(cols["Fc_all"].min()) >= -1.0

    manifest = pdg.read_pretrain_manifest(path)
    assert manifest["exchange_footing"] == "paper"
    assert set(cols) == pdg.pretrain_npz_keys(
        polarized=True, descriptors=True, exchange_footing="total")


def test_the_paper_footing_changes_only_what_the_paper_changes(tmp_path):
    """Against the ``total`` footing the published targets move the open
    shell's exchange and leave the closed shell's where it was, and move the
    correlation of both by no more than the amplitude sets differ.

    Oracle: two generated files. A closed shell's libxc PBE exchange ratio IS
    the analytic total-density form, so He's exchange rows agree to the
    relative tolerance with the ``_ONE_ULP_BAND`` absolute band (the stored
    ``F_x - 1`` is of order 1e-5 where ``s`` is small, and both forms recover
    it by a subtraction from one); an open shell's is the spin-resolved
    ratio, which the published total-density form is not, so H's rows
    separate. The correlation rows move because the
    published amplitudes differ from BOTH libxc sets the stored ratio is built
    from -- ``LDA_C_PW_MOD`` in its numerator and ``LDA_C_PW`` in its
    denominator -- and the move is held to the first-order estimate
    ``2 max(relative amplitude difference) (1 + max|F_c|)`` formed on the
    file's own rows, so a target that changed for any other reason fails it.
    """
    _path_t, total = _gen(tmp_path / "total", exchange_footing="total")
    path_p, paper = _gen(tmp_path / "paper", exchange_footing="paper")

    np.testing.assert_array_equal(total["rho_all"], paper["rho_all"])

    he = paper["system_all"] == _system_index(path_p, "He")
    h = paper["system_all"] == _system_index(path_p, "H")
    assert int(he.sum()) > 0 and int(h.sum()) > 0

    np.testing.assert_allclose(paper["Fx_all"][he], total["Fx_all"][he],
                               rtol=1e-12, atol=_ONE_ULP_BAND)
    assert float(np.max(np.abs(paper["Fx_all"][h] - total["Fx_all"][h]))) > 1e-3

    rho = paper["rho_all"][he]
    zeta = paper["zeta_all"][he]
    published = _eps_c(rho, zeta)
    relative = max(
        float(np.max(np.abs(published / _eps_c(rho, zeta, _A_PW) - 1.0))),
        float(np.max(np.abs(published / _eps_c(rho, zeta, _A_PW_MOD) - 1.0))))
    bound = 2.0 * relative * (1.0 + float(np.max(np.abs(paper["Fc_all"][he]))))
    moved = float(np.max(np.abs(paper["Fc_all"][he] - total["Fc_all"][he])))
    print(f"closed-shell correlation target moved by at most {moved:.3e} "
          f"(first-order bound {bound:.3e}, relative amplitude difference "
          f"{relative:.3e})")
    assert moved <= bound


# ---------------------------------------------------------------------------
# The pretraining run on a paper file
# ---------------------------------------------------------------------------

def test_a_paper_run_records_its_footing_and_refuses_an_energy_term(tmp_path):
    """A pretraining run on a file built on the published footing records
    that footing in its metadata -- the file's key set is the ``total``
    footing's, so the record comes from the manifest and not from the row
    block -- and refuses a per-system energy term at a positive weight, whose
    open-shell exchange target that footing does not carry.

    Oracle: the metadata ``run_pretrain`` returns after two steps on the
    two-system file, and the refusal, seen to fire before any step.
    """
    import dataclasses

    from xcquinox.pipeline.config import PretrainSpec, get_architecture
    from xcquinox.pipeline.pretrain import run_pretrain

    import json

    _gen(tmp_path, exchange_footing="paper")
    arch = dataclasses.replace(get_architecture("deep_3x16"),
                               use_polarized_correlation=True,
                               descriptor_coordinates="paper", ueg_gate="x2")
    spec = dict(arch=arch, data_dir=str(tmp_path),
                checkpoint_dir=str(tmp_path / "ckpt"), n_steps=2,
                lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.5,
                grad_clip=0.0, seed=0, loss_weighting="rho_w_sampled")
    metadata = run_pretrain(PretrainSpec(**spec))
    assert metadata["exchange_footing"] == "paper"
    assert metadata["energy_term_weight"] == 0.0
    # The record names the class the networks were written as, the gate
    # included, on the returned metadata and on the file the readers open.
    assert metadata["ueg_gate"] == "x2"
    assert metadata["descriptor_coordinates"] == "paper"
    on_disk = json.loads((tmp_path / "ckpt" / "pretrain_metadata.json").read_text())
    assert on_disk["ueg_gate"] == "x2"
    assert on_disk["exchange_footing"] == "paper"

    with pytest.raises(ValueError, match="paper"):
        run_pretrain(PretrainSpec(**dict(spec, energy_term_weight=0.1,
                                         checkpoint_dir=str(tmp_path / "w"))))


# ---------------------------------------------------------------------------
# The gate's default on every construction path
# ---------------------------------------------------------------------------

def test_the_gate_defaults_to_tanh2_on_every_construction_path():
    """Every way an architecture, a model block or a network comes into being
    without naming the gate yields ``tanh2``, the gate every model built
    before the field carried, so an existing checkpoint's class is what it
    was on each of them: the dataclass constructor, the ``from_spec``
    factory, the registry, the run's model block and the two network
    constructors each carry their own default.

    Oracle: the field read back on each path.
    """
    from xcquinox.pipeline.cluster.grid_config import ModelConfig
    from xcquinox.pipeline.config import ArchitectureConfig, get_architecture
    from xcquinox.pipeline.networks import AlecGGA_CNet, AlecGGA_XNet

    assert ArchitectureConfig(name="t", depth=2, nodes=8).ueg_gate == "tanh2"
    assert ArchitectureConfig.from_spec("t", 2, 8).ueg_gate == "tanh2"
    assert get_architecture("deep_3x16").ueg_gate == "tanh2"
    assert ModelConfig().ueg_gate == "tanh2"
    assert AlecGGA_XNet(n_extra_features=0, depth=2, nodes=8).ueg_gate == "tanh2"
    assert AlecGGA_CNet(n_extra_features=0, depth=2, nodes=8,
                        use_spin_polarization=True).ueg_gate == "tanh2"


def test_a_manifest_that_contradicts_the_exchange_block_is_refused(tmp_path):
    """The footing a run records is the manifest's, so the manifest is held
    to the block the run reads in both directions: a file carrying the
    per-channel exchange block beside a manifest naming a total-density
    footing is refused before any step, as the converse pairing already was.

    Oracle: a file generated on the spin-channel footing (the open shell
    gives it the per-channel block), its manifest rewritten to name the
    published footing, and ``run_pretrain`` seen to refuse it naming the
    block.
    """
    import dataclasses
    import json

    from xcquinox.pipeline.config import PretrainSpec, get_architecture
    from xcquinox.pipeline.pretrain import run_pretrain

    path, cols = _gen(tmp_path, exchange_footing="spin_channel")
    assert "rho_x" in cols
    manifest_path = pdg._pretrain_manifest_path(path)
    manifest = json.loads(open(manifest_path).read())
    assert manifest["exchange_footing"] == "spin_channel"
    manifest["exchange_footing"] = "paper"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    arch = dataclasses.replace(get_architecture("deep_3x16"),
                               use_polarized_correlation=True)
    spec = PretrainSpec(arch=arch, data_dir=str(tmp_path),
                        checkpoint_dir=str(tmp_path / "ckpt"), n_steps=2,
                        lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.5,
                        grad_clip=0.0, seed=0, loss_weighting="rho_w_sampled")
    with pytest.raises(ValueError, match="per-channel exchange block"):
        run_pretrain(spec)

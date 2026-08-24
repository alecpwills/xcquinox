"""Closed-shell reference record, computed from whichever xcquinox is importable.

rho_a = rho_b makes the three per-channel feature blocks identical by
construction (doubling either channel of [D/2, D/2] reproduces the matrix, and
2 rho_a / 4 sigma_aa are then rho_tot / sigma_tot), so the exact spin scaling
must leave every closed-shell number untouched. This script computes those
numbers so the SAME script, run against an archived tree and against the working
tree, produces two records that can be compared digit for digit.

Two fixtures are kept. ``closed_shell_reference_ae204537e.json`` was produced
against the tree at ae204537e, extracted read-only with ``git archive``; this
script is not in that tree (it postdates it), so it is copied in and run FROM
the archive directory::

    git archive ae204537e | tar -x -C <archive-dir>
    cp xcquinox/alec/tests/record_closed_shell_reference.py \\
       <archive-dir>/xcquinox/alec/tests/
    cd <archive-dir> && PYTHONPATH=<archive-dir> JAX_PLATFORMS=cpu \\
        python xcquinox/alec/tests/record_closed_shell_reference.py \\
        > <repo>/xcquinox/alec/tests/fixtures/closed_shell_reference_ae204537e.json

The working directory matters. With an editable install of the working tree in
the environment, running the same command FROM the repository root loads the
working tree's package however ``PYTHONPATH`` is set (measured: the header line
then names ``<repo>/xcquinox/__init__.py``), and the fixture would be a copy of
the tree it is supposed to be compared against. The header line printed on
stderr names the loaded package; it must point into the archive before the
output is accepted.

``closed_shell_reference_smooth_alpha.json`` was produced by the same script
run from the repository root against the tree in which the iso-orbital
indicator's lower bound became a smooth positive part (``metagga.compute_alpha``,
width 1e-5; DEFERRED_WORK.md entry 27), which moves the five meta-GGA
architectures' closed-shell numbers by the indicator's footprint (at most
1.7e-10 on this record) and nothing else. It is the fixture the live tree is
held to bitwise; the ae204537e fixture is kept and the comparison against it
carries the measured footprint as its tolerance
(``test_closed_shell_byte_identity._SMOOTH_ALPHA_DELTA``).

The record is computed inside :func:`_reproducible_pyscf`, which pins the two
pieces of PySCF state the reference SCF's LAST DIGITS depend on. Both were
measured, not assumed:

* OpenMP thread count. The SCF's threaded reductions are not associative, so
  its density matrix depends on the count: at four threads one molecule gave
  ``E_non_xc`` = -67.00327081852356, -67.0032708185234 and -67.00327081852353
  for three records that must agree, against -67.0032708185235 at one thread.
  The BLAS thread count does not matter here (the one-thread record reproduces
  with ``OMP_NUM_THREADS=4`` in the environment).

* ``lib.param.MAX_MEMORY``. PySCF sizes the grid loop of its exchange-
  correlation quadrature from ``mol.max_memory - lib.current_memory()``, so the
  blocking of the grid -- and with it the order in which the quadrature is
  summed -- depends on how much memory the PROCESS has already used. Running
  the record after other work in the same interpreter moved ``E_non_xc``
  through -67.00327081852353 and -67.00327081852359 at 1.8 to 4.2 GB of
  accumulated usage; pinning the ceiling high enough that the block size
  saturates removes the dependence (verified by re-running the failing
  sequence with the ceiling pinned).

The two inputs (``E_non_xc`` and a digest of the reference density matrix) are
stored beside the six numbers anyway, so that a moved input is still reported
as such rather than as a moved code path.

Both pins hold the record across PROCESSES ON ONE MACHINE; neither can hold it
across machines. The last digits of the reference SCF are those of the BLAS
kernels the CPU selects and of the compiled libraries doing the arithmetic, so
a record taken here is not reproducible bit for bit on another CPU: the same
architecture read -67.00327081852355 against this fixture's -67.0032708185235
on an AMD Milan cluster node -- three ulps, 4.3e-14 Ha, 6.4e-16 relative. Each
fixture
therefore carries a ``platform`` block -- the fields of :data:`PLATFORM_KEYS`,
written by :func:`platform_fingerprint` -- beside its ``records`` block, and
the comparison in ``test_closed_shell_byte_identity`` is bitwise only where
the running platform reproduces that block. The fixture layout is::

    {"platform": {<PLATFORM_KEYS>}, "records": {<arch>: {<RECORD_KEYS>}}}
"""
import contextlib
import hashlib
import inspect
import json
import os
import platform
import sys

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import assemble_descriptor_features
from xcquinox.alec.oneshot import (
    compute_vxc_nn, fixed_density_total_energy, split_exc_energy_uks)

# Closed-shell probe: a molecule that visits every descriptor (three nuclei for
# the cusp feature, a genuine density matrix for the rung-3.5 and DM statistics
# columns, a non-uniform iso-orbital indicator for the meta-GGA column).
_SPEC = dict(name="H2O_closed_shell_reference",
             atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
             basis="def2-svp", charge=0, spin=0,
             atom_composition=(("O", 1), ("H", 2)), grid_level=1)

#: The keys every record carries: the six closed-shell numbers and the two
#: input pins.
RECORD_KEYS = ("E_rks", "V_rks_trace", "V_rks_sq", "E_uks_closed",
               "V_uks_a_trace", "V_uks_a_sq", "E_non_xc", "dm_pbe_sha1")


#: Memory ceiling (MB) pinned on every Mole while the record is computed.
#: Large enough that PySCF's grid block size saturates its own cap for this
#: molecule, so the quadrature is summed in the same order however much memory
#: the process has already used. The value is not a resource request: nothing
#: here allocates near it (the probe is 24 basis functions on a level-1 grid).
_PINNED_MAX_MEMORY_MB = 200000.0


@contextlib.contextmanager
def _reproducible_pyscf():
    """PySCF held at one OpenMP thread and a fixed memory ceiling.

    Both are restored on exit. See the module docstring for the measured
    dependence of the reference SCF's last digits on each.
    """
    from pyscf import lib
    from pyscf.gto.mole import MoleBase
    previous_threads = lib.num_threads()
    # The ceiling every Mole is born with. It is a class attribute of
    # MoleBase, initialised from pyscf.__config__ when the class is defined,
    # so pinning it here reaches the molecule the precompute builds next
    # (lib.param.MAX_MEMORY is read once at class definition and assigning to
    # it later changes nothing -- checked).
    previous_memory = MoleBase.max_memory
    lib.num_threads(1)
    MoleBase.max_memory = _PINNED_MAX_MEMORY_MB
    try:
        yield
    finally:
        lib.num_threads(previous_threads)
        MoleBase.max_memory = previous_memory


#: The platform fields stamped into every fixture beside its records. The
#: record's last digits are a property of the machine as well as of the code:
#: the arithmetic is done by the BLAS kernels the CPU selects and by the
#: compiled libraries around them, none of which the two pins above reach.
#: Six fields name the machine and its numerical libraries; the last two are
#: the recorder's own pins, so a fixture recorded at a different thread count
#: or memory ceiling is not read as if it shared this one's summation order.
PLATFORM_KEYS = ("cpu_model", "numpy_version", "jax_version", "jaxlib_version",
                 "pyscf_version", "blas", "pyscf_threads",
                 "pinned_max_memory_mb")


def _cpu_model():
    """The CPU's model string. ``platform.processor()`` reports only the
    instruction set on Linux ('x86_64'), which does not separate an Intel
    desktop part from an AMD server part, so /proc/cpuinfo is read first."""
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _blas_description():
    """The BLAS numpy reports, as one string.

    numpy 2 exposes its build metadata as a dict; ``get_info`` is kept as the
    fallback for an older numpy. A platform whose BLAS cannot be named reports
    "unknown", which does not compare equal to a named one, so the comparison
    takes its cross-platform branch rather than assuming a shared library.
    """
    try:
        build = np.show_config("dicts")["Build Dependencies"]["blas"]
        fields = [str(build.get("name", "unknown")),
                  str(build.get("version", ""))]
        configuration = build.get("openblas configuration")
        text = " ".join(f for f in fields if f)
        return f"{text} ({configuration})" if configuration else text
    except Exception:
        pass
    try:
        libraries = np.__config__.get_info("blas_opt").get("libraries", [])
        return ",".join(str(name) for name in libraries) or "unknown"
    except Exception:
        return "unknown"


def platform_fingerprint() -> dict:
    """The machine and the pins this record was computed under.

    The thread count and the memory ceiling are read INSIDE
    :func:`_reproducible_pyscf`, i.e. as the record itself sees them, so the
    stamp is a measurement of the pins rather than a copy of the constants
    they are set from.
    """
    import jaxlib
    import pyscf
    from pyscf import lib
    from pyscf.gto.mole import MoleBase
    with _reproducible_pyscf():
        threads = int(lib.num_threads())
        max_memory_mb = float(MoleBase.max_memory)
    return {
        "cpu_model": _cpu_model(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "pyscf_version": pyscf.__version__,
        "blas": _blas_description(),
        "pyscf_threads": threads,
        "pinned_max_memory_mb": max_memory_mb,
    }


def _build_model(arch_name):
    """Production configuration, fixed seed, identical in both trees."""
    import dataclasses
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _split_energy(model, rho_a, rho_b, sig_aa, sig_bb, sig_tot, features, w):
    """Call the split UKS energy under either arity.

    At rho_a = rho_b the three per-channel blocks are the SAME array, so passing
    one block three times is the closed-shell case rather than a compatibility
    shim; the archived tree took a single block because it had no per-channel
    notion at all.
    """
    n_params = len(inspect.signature(split_exc_energy_uks).parameters)
    if n_params == 8:
        return split_exc_energy_uks(model, rho_a, rho_b, sig_aa, sig_bb,
                                    sig_tot, features, w)
    return split_exc_energy_uks(model, rho_a, rho_b, sig_aa, sig_bb, sig_tot,
                                features, features, features, w)


def closed_shell_record(arch_name) -> dict:
    """The six numbers that pin this architecture's closed-shell behavior, and
    the two pins of the record they were computed on."""
    with _reproducible_pyscf():
        return _closed_shell_record(arch_name)


def _closed_shell_record(arch_name) -> dict:
    model = _build_model(arch_name)
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys}))
    md = precompute_fixed_density_data(MoleculeSpec(**_SPEC),
                                       required_keys=keys,
                                       descriptors=model.descriptors)
    features = assemble_descriptor_features(model.descriptors, md)
    ao = jnp.asarray(md["ao_grid"])
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_deriv[1:4]
    w = jnp.asarray(md["grid_weights"])

    e_rks = float(fixed_density_total_energy(model, md))
    v_rks = compute_vxc_nn(model, jnp.asarray(md["rho_grid"]),
                           jnp.asarray(md["sigma_grid"]), features, ao, w,
                           nabla_rho=jnp.asarray(md["nabla_rho_grid"]),
                           ao_grad=ao_deriv)

    # The same molecule fed through the UKS helpers as a closed shell:
    # D_a = D_b = D / 2, so rho_a = rho_b and the spin channels coincide.
    D_half = 0.5 * jnp.asarray(md["dm_pbe"])

    def grid(D):
        rho = jnp.einsum("ij,gi,gj->g", D, ao, ao)
        nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_xyz, ao)
        return rho, nabla, jnp.sum(nabla * nabla, axis=1)

    rho_h, nabla_h, sig_h = grid(D_half)
    nabla_tot = 2.0 * nabla_h
    sig_tot = jnp.sum(nabla_tot * nabla_tot, axis=1)
    e_uks = float(_split_energy(model, rho_h, rho_h, sig_h, sig_h, sig_tot,
                                features, w))
    v_uks_a = compute_vxc_nn(model, 2.0 * rho_h, 4.0 * sig_h, features, ao, w,
                             nabla_rho=2.0 * nabla_h, ao_grad=ao_deriv,
                             part="x") \
        + compute_vxc_nn(model, 2.0 * rho_h, sig_tot, features, ao, w,
                         nabla_rho=nabla_tot, ao_grad=ao_deriv, part="c")
    dm_bytes = np.ascontiguousarray(np.asarray(md["dm_pbe"],
                                               dtype=np.float64)).tobytes()
    return {
        "E_rks": e_rks,
        "V_rks_trace": float(jnp.sum(v_rks)),
        "V_rks_sq": float(jnp.sum(v_rks * v_rks)),
        "E_uks_closed": e_uks,
        "V_uks_a_trace": float(jnp.sum(v_uks_a)),
        "V_uks_a_sq": float(jnp.sum(v_uks_a * v_uks_a)),
        "E_non_xc": float(md["E_non_xc"]),
        "dm_pbe_sha1": hashlib.sha1(dm_bytes).hexdigest(),
    }


def main():
    print(f"# xcquinox loaded from {sys.modules['xcquinox'].__file__}",
          file=sys.stderr)
    document = {
        "platform": platform_fingerprint(),
        "records": {name: closed_shell_record(name)
                    for name in sorted(alec.ARCHITECTURES)},
    }
    json.dump(document, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

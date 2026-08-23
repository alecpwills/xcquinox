"""Record the pretrain-data generator's DEFAULT output as a test fixture.

Not a test module: pytest does not collect it (no ``test_`` prefix). Run it
once, before the pretraining-protocol change touches the generator, to freeze
what the default configuration produced:

    python xcquinox/alec/tests/record_pretrain_data_reference.py

The fixture is a two-atom (He closed shell, H open shell) file at sto-3g and
grid level 0 with descriptors and the zeta column, which exercises the RKS
branch, the UKS branch, every descriptor column and the (r_s, s, alpha) mesh in
a few hundred kilobytes. Stored compressed; the assertions are on array
contents, not on the zip container, whose headers carry write timestamps.

Both atoms carry a single contracted s function in this basis, so the
self-consistent density matrix is fixed by normalization alone and every stored
column is bit-reproducible across processes and BLAS thread counts; the
orientation-lock bias (a traceless quadrupole) vanishes identically on an
s-only basis, so the recording is independent of the lock strength as well.
"""
import os
import sys
import tempfile

import numpy as np

from xcquinox.alec.pretrain_data_gen import generate_pretrain_data_npz

_ATOMS = (("He", 0), ("H", 1))
_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                        "pretrain_data_default_reference.npz")


def main(out_path=_FIXTURE):
    with tempfile.TemporaryDirectory() as tmp:
        path = generate_pretrain_data_npz(
            tmp, atoms=_ATOMS, basis="sto-3g", grid_level=0, polarized=True,
            descriptors=True, density_fit=False)
        with np.load(path) as z:
            payload = {k: np.array(z[k]) for k in z.files}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"wrote {out_path}")
    for k in sorted(payload):
        print(f"  {k}: shape={payload[k].shape} dtype={payload[k].dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))

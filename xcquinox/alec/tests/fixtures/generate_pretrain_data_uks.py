"""Generate UKS pretrain data fixture for H2O triplet.

This script will produce pretrain_data_uks_tiny.npz containing UKS
(unrestricted Kohn-Sham) pretraining data for the H2O molecule in a
triplet spin state.  The fixture is used by test_workers.py
test_uks_pretrain_worker (currently xfailed until this generator is
runnable).

Expected output:
    xcquinox/alec/tests/fixtures/pretrain_data_uks_tiny.npz

Contents (when implemented):
    - rho_a, rho_b      : alpha/beta densities on a grid
    - sigma_aa, sigma_bb : gradient norms squared
    - sigma_ab           : cross gradient dot product
    - exc_ref            : reference XC energy densities
    - weights            : integration weights

To run (once dependencies are available):
    python -m xcquinox.alec.tests.fixtures.generate_pretrain_data_uks

Status: STUB -- not yet runnable.
"""


def main():
    raise NotImplementedError(
        "UKS pretrain data generation not yet implemented. "
        "This stub documents the intended interface; see module docstring."
    )


if __name__ == "__main__":
    main()

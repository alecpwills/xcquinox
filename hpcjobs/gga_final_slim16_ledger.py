#!/usr/bin/env python3
"""Write the Slim16 evaluation ledger of the GGA-final arm.

One ``pretrain`` entry per architecture of the arm's sweep, each pointing
``slim16_eval.py prepare`` at the arm's run directory (its resolved_config.yaml
and ``pretrain/<arch>/{xnet,cnet}.eqx``). The labels carry the ``gf_`` prefix
the figures and tables of the arm's evaluation read.

Usage::

    python3 gga_final_slim16_ledger.py RUN_DIR OUT

RUN_DIR is the run directory the harness submit printed; OUT is the ledger
path the prepare wrapper reads. An existing OUT is overwritten: a resubmitted
arm gets a new run directory and its ledger must follow it, and the only
ledger a prepare consumes is the one written immediately before it.
"""
import json
import sys

# (label, architecture) of the arm's sweep, in evaluation order
NETWORKS = (
    ("gf_atomctx", "deep_atomctx_3x16"),
    ("gf_rung35only", "deep_rung35only_3x16"),
    ("gf_cusp_attn", "deep_cusp_attn"),
    ("gf_dm_attn", "deep_dm_attn"),
)


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        print("usage: gga_final_slim16_ledger.py RUN_DIR OUT", file=sys.stderr)
        return 2
    run_dir, out = argv[1], argv[2]
    entries = [{"label": label, "kind": "pretrain", "arch": arch,
                "run_dir": run_dir}
               for label, arch in NETWORKS]
    with open(out, "w") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")
    print(f"ledger: {out} ({len(entries)} pretrain entries, run {run_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

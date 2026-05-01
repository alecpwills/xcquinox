"""Generate gga_training_example-step7.ipynb.

Pattern matches notebooks/_build_step{4,5,6}_notebook.py: assemble cells
into nbformat structure, write .ipynb, optionally execute end-to-end.

Step-7 plan:
  - Build Dick 2021 SI II training pool (28 entries) via
    xcquinox.alec.dick_pool.build_dick_pool
  - Extract (rho^{1/3}, s, alpha) per species, cache to disk
  - Build reference 3-histogram via build_reference_histograms
  - Sweep r in {1,2,3,4,5,6,7,12,15,18,21} x {l2,jsd} x {oneshot,full_3}
    x {with_hbpt, no_hbpt} = 88 training runs
  - Post-process: 6+1 figures + headline.json

Citations:
  - PBE 1996  (PRL 77, 3865) -- descriptor s
  - SCAN 2015 (PRL 115, 036402) -- descriptor alpha
  - Lin 1991  (IEEE TIT 37, 145) -- JSD
  - Chen 2018 (arXiv:1711.02257) -- GradNorm
  - Dick 2021 (PRB 104, L161109) -- candidate pool
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

NOTEBOOK_OUT = Path(__file__).resolve().parent / "gga_training_example-step7.ipynb"


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


# Constants -------------------------------------------------------------
SUBSET_SIZES = (1, 2, 3, 4, 5, 6, 7, 12, 15, 18, 21)
METRICS = ("l2", "jsd")
SOLVERS = ("oneshot", "full_3")
AUGMENTATIONS = (False, True)  # with_hbpt False/True
ARCH_NAME = "deep_combined_attn"
LOSS_NAME = "L5_gradnorm_vxc_step7"
PRETRAIN_ORIGIN = "integration"  # ONLY origin — unweighted is out of scope per step-6 verdict.
# Pretrain phase: 1000 steps already baked into step-6 integration
# checkpoint (loaded from notebooks/checkpoints_step6/integration/pretrain/
# deep_combined_attn/{xnet.eqx, cnet.eqx}); no re-pretraining in step-7.
# Task-loss phase: 100 steps (matches step-6 group1/group2 short schedule;
# step-6 group3 used 250 but at 88 runs the additional convergence gain
# does not justify the 2.5x cost given step-6 evidence).
TRAIN_N_STEPS = 100
LR_START, LR_END = 1e-2, 1e-5
LR_DECAY_START = 0.2
GRAD_CLIP = 1.0


def build_cells() -> list:
    cells: list = []
    cells.append(_md(
        "# Step-7: Histogram-Matched Subset Selection from Dick 2021 Training Pool\n\n"
        "Generate optimally-representative training subsets (1..21 size sweep) by\n"
        "minimizing distance between candidate-subset and full-pool histograms over\n"
        "$(\\rho^{1/3}, s, \\alpha)$.\n\n"
        "**Critical:** $\\alpha$ enters the subset-selection objective only -- the\n"
        "trained GGA network does NOT consume it. Future MGGA extension is step-8+.\n\n"
        "Reference: Dick & Fernandez-Serra, *Phys. Rev. B* **104**, L161109 (2021), SI II.\n"
    ))
    cells.append(_code(
        "from xcquinox.alec import subset_selection as ss\n"
        "from xcquinox.alec import dick_pool\n"
        "from xcquinox.alec import losses\n"
        "import numpy as np\n"
        "from pathlib import Path\n\n"
        "REPO = Path('/home/awills/Documents/Research/xcquinox')\n"
        "STEP7_ROOT = REPO / 'notebooks' / 'checkpoints_step7'\n"
        "DESCRIPTOR_CACHE = STEP7_ROOT / 'subset_descriptors'\n"
        "REF_HIST_CACHE = STEP7_ROOT / 'dick_pool_full_hist'\n"
        "DESCRIPTOR_CACHE.mkdir(parents=True, exist_ok=True)\n"
        "REF_HIST_CACHE.mkdir(parents=True, exist_ok=True)\n\n"
        "pool = dick_pool.build_dick_pool()\n"
        "print(f'Dick 2021 SI II training pool: {pool[\"n_total\"]} entries')\n"
        "print(f'  AE molecules: {len(pool[\"ae_molecules\"])}')\n"
        "print(f'  BH76 reactions: {len(pool[\"bh76_reactions\"])}')\n"
        "print(f'  IP13 pairs: {len(pool[\"ip13_pairs\"])}')\n"
        "print(f'  Atom refs: {len(pool[\"atom_refs\"])}')\n"
    ))
    cells.append(_md(
        "## 1. Descriptor Extraction (cached)\n\n"
        "For each unique species in the candidate pool, run a single PBE SCF\n"
        "at def2-svp / grid_level=1 (matching step-5/6 conventions) and extract\n"
        "$(\\rho^{1/3}, s, \\alpha)$ on the molecular grid. Cached as\n"
        "`subset_descriptors/<idx>_<species>.npz`.\n"
    ))
    cells.append(_code(
        "ae_descriptors = []\n"
        "for idx, at in enumerate(pool['ae_molecules']):\n"
        "    arrs = ss.extract_descriptors(at, idx=idx, cache_dir=DESCRIPTOR_CACHE)\n"
        "    ae_descriptors.append(arrs)\n"
        "    name = at.info.get('dick_hill', at.get_chemical_formula())\n"
        "    print(f'  {idx:2d} {name:8s} ngrid={arrs[\"rho_third\"].size}')\n"
        "print(f'Total AE descriptors cached: {len(ae_descriptors)}')\n"
    ))
    cells.append(_md(
        "## 2. Reference-Histogram Builder\n\n"
        "Concatenate descriptors across the full 21-AE pool and build 3 200-bin\n"
        "log10 density-normalized histograms over $(\\rho^{1/3}, s, \\alpha)$.\n"
        "Same edges are reused for every candidate-subset histogram.\n"
    ))
    cells.append(_code(
        "import numpy as np\n"
        "h_ref, edges = ss.build_reference_histograms(ae_descriptors)\n"
        "ref_path = REF_HIST_CACHE / 'reference.npz'\n"
        "np.savez(ref_path,\n"
        "         h_ref_rho=h_ref['rho_third'], e_rho=edges['rho_third'],\n"
        "         h_ref_s=h_ref['s'],         e_s=edges['s'],\n"
        "         h_ref_alpha=h_ref['alpha'], e_alpha=edges['alpha'])\n"
        "print(f'Wrote reference histograms to {ref_path}')\n"
        "for k in ('rho_third', 's', 'alpha'):\n"
        "    print(f'  {k:10s} histogram: shape={h_ref[k].shape}, sum={h_ref[k].sum():.4f}')\n"
    ))
    cells.append(_md(
        "## 3. Subset Generation Sweep\n\n"
        "For each $(r, \\text{metric}, \\text{aug})$, call `select_subset`,\n"
        "compute the atom-set per spec §5c, augment with HBPT if requested,\n"
        "and write a `subset.traj` to the per-spec checkpoint directory.\n"
        "Selection pool size = 21 AE molecules; auxiliaries fixed across all subsets.\n"
    ))
    cells.append(_code(
        "from ase.io import write as ase_write\n"
        "from ase import Atoms\n"
        "import json\n\n"
        f"SUBSET_SIZES = {SUBSET_SIZES!r}\n"
        f"METRICS = {METRICS!r}\n"
        f"AUGMENTATIONS = {AUGMENTATIONS!r}\n"
        f"SOLVERS = {SOLVERS!r}\n"
        f"ARCH_NAME = {ARCH_NAME!r}\n"
        f"LOSS_NAME = {LOSS_NAME!r}\n\n"
        "subset_index_log = {}  # (metric, r, aug) -> chosen_indices + atom_set\n"
        "for metric in METRICS:\n"
        "    for r in SUBSET_SIZES:\n"
        "        chosen, val = ss.select_subset(\n"
        "            ae_descriptors, edges, h_ref, r=r, metric=metric)\n"
        "        chosen_atoms = [pool['ae_molecules'][i] for i in chosen]\n"
        "        atom_syms = ss.compute_atom_set(chosen_atoms)\n"
        "        # Match atom_refs in pool by chemical symbol; build new Atoms\n"
        "        # for any element NOT in pool['atom_refs']\n"
        "        pool_ref_syms = {a.get_chemical_formula() for a in pool['atom_refs']}\n"
        "        atom_refs_subset = [a for a in pool['atom_refs']\n"
        "                            if a.get_chemical_formula() in atom_syms]\n"
        "        for sym in atom_syms - pool_ref_syms:\n"
        "            atom_refs_subset.append(Atoms(sym, positions=[(0,0,0)]))\n"
        "        for aug in AUGMENTATIONS:\n"
        "            traj_atoms = ss.augment_with_hbpt(\n"
        "                chosen_atoms, atom_refs_subset, with_hbpt=aug)\n"
        "            tag = f'bin{r:02d}{\"w\" if aug else \"\"}'\n"
        "            for solver in SOLVERS:\n"
        "                spec_dir = (STEP7_ROOT / metric / tag /\n"
        "                            f'{ARCH_NAME}/{LOSS_NAME}/{solver}')\n"
        "                spec_dir.mkdir(parents=True, exist_ok=True)\n"
        "                ase_write(str(spec_dir / 'subset.traj'), traj_atoms)\n"
        "            subset_index_log[(metric, r, aug)] = {\n"
        "                'chosen_indices': list(chosen),\n"
        "                'metric_value': float(val),\n"
        "                'atom_set': sorted(atom_syms),\n"
        "                'tag': tag,\n"
        "            }\n"
        "ledger_path = STEP7_ROOT / 'subset_index_log.json'\n"
        "ledger_path.write_text(json.dumps(\n"
        "    {f'{k[0]}/{k[1]}/{k[2]}': v for k, v in subset_index_log.items()},\n"
        "    indent=2))\n"
        "n_specs = len(subset_index_log) * len(SOLVERS)\n"
        "print(f'Wrote {len(subset_index_log)} (metric, r, aug) entries to {ledger_path}')\n"
        "print(f'Total subset.traj files written: {n_specs}')\n"
        "assert n_specs == 88, f'Expected 88 specs, got {n_specs}'\n"
    ))
    cells.append(_md(
        "## 4. Smoke Test (r=2, l2, no-w, oneshot)\n\n"
        "Verify the wiring with a single training spec before launching the\n"
        "full 88-run grid. Reads the generated subset.traj and confirms the\n"
        "step-6 integration pretrain checkpoint is loadable.\n"
    ))
    cells.append(_code(
        "smoke_metric, smoke_r, smoke_aug, smoke_solver = 'l2', 2, False, 'oneshot'\n"
        "tag = f'bin{smoke_r:02d}{\"w\" if smoke_aug else \"\"}'\n"
        "smoke_spec_dir = (STEP7_ROOT / smoke_metric / tag /\n"
        "                  f'{ARCH_NAME}/{LOSS_NAME}/{smoke_solver}')\n"
        "subset_path = smoke_spec_dir / 'subset.traj'\n"
        "from ase.io import read as ase_read\n"
        "smoke_traj = ase_read(str(subset_path), ':')\n"
        "print(f'Smoke spec: {smoke_metric}/{tag}/{smoke_solver}')\n"
        "print(f'  subset.traj entries: {len(smoke_traj)}')\n"
        "for i, at in enumerate(smoke_traj):\n"
        "    name = at.info.get('name', at.info.get('dick_hill', at.get_chemical_formula()))\n"
        "    print(f'    {i:2d} {at.get_chemical_formula():10s} ({name})')\n\n"
        "# Verify the step-6 integration pretrain checkpoint files exist:\n"
        "smoke_pretrain = (REPO / 'notebooks' / 'checkpoints_step6' /\n"
        "                  'integration' / 'pretrain' / ARCH_NAME)\n"
        "for fname in ('xnet.eqx', 'cnet.eqx'):\n"
        "    fp = smoke_pretrain / fname\n"
        "    assert fp.exists(), f'Missing pretrain checkpoint {fp}'\n"
        "    print(f'  pretrain checkpoint OK: {fp.name}')\n"
        "print('Smoke wiring verified.')\n"
    ))
    cells.append(_md(
        "## 5. Full Training Grid (88 runs)\n\n"
        "$11~\\text{sizes} \\times 2~\\text{metrics} \\times 2~\\text{solvers}\n"
        " \\times 2~\\text{augmentations} = 88$ training runs. Each loads the\n"
        "step-6 integration pretrain checkpoint and trains for $100$ task-loss\n"
        "steps with `L5_gradnorm_vxc_step7` (5 task channels: AE+BH76+IP13+vxc+ρ).\n\n"
        "Skip-if-done: if `eval_df.csv` already exists for a spec, skip it.\n"
    ))
    cells.append(_code(
        "# IMPORTANT: this cell is the driver placeholder. The actual training\n"
        "# call signature must adapt to the alec training entry point present in\n"
        "# xcquinox/alec/train.py at notebook-execution time. The pattern below\n"
        "# matches the step-6 group-3 invocation idiom in _build_step6_notebook.py;\n"
        "# adjust as needed when the real alec runner is wired.\n"
        "from xcquinox.alec import train as alec_train\n"
        "from xcquinox.alec.config import ARCHITECTURE_REGISTRY\n\n"
        "arch_cfg = ARCHITECTURE_REGISTRY[ARCH_NAME]\n"
        "pretrain_dir = REPO / 'notebooks' / 'checkpoints_step6' / 'integration' / 'pretrain' / ARCH_NAME\n"
        "n_done, n_failed, n_skipped = 0, 0, 0\n"
        "for metric in METRICS:\n"
        "    for r in SUBSET_SIZES:\n"
        "        for aug in AUGMENTATIONS:\n"
        "            tag = f'bin{r:02d}{\"w\" if aug else \"\"}'\n"
        "            for solver in SOLVERS:\n"
        "                spec_dir = (STEP7_ROOT / metric / tag /\n"
        "                            f'{ARCH_NAME}/{LOSS_NAME}/{solver}')\n"
        "                eval_csv = spec_dir / 'eval_df.csv'\n"
        "                if eval_csv.exists():\n"
        "                    n_skipped += 1\n"
        "                    continue\n"
        "                # The actual call into alec_train. Adapt signature to\n"
        "                # the version of alec/train.py present at runtime.\n"
        "                try:\n"
        "                    print(f'Training {metric}/{tag}/{solver} ...')\n"
        "                    # Placeholder for the real call:\n"
        "                    # alec_train.run_step7_spec(\n"
        "                    #     arch_cfg=arch_cfg,\n"
        "                    #     loss_name=LOSS_NAME,\n"
        "                    #     loss_kwargs={\n"
        "                    #         'molecules': pool['ae_molecules'],\n"
        "                    #         'bh76_reactions': pool['bh76_reactions'],\n"
        "                    #         'ip13_pairs': pool['ip13_pairs'],\n"
        "                    #     },\n"
        "                    #     solver=solver,\n"
        "                    #     train_traj=spec_dir / 'subset.traj',\n"
        "                    #     pretrain_xnet=pretrain_dir / 'xnet.eqx',\n"
        "                    #     pretrain_cnet=pretrain_dir / 'cnet.eqx',\n"
        "                    #     n_steps=TRAIN_N_STEPS,\n"
        "                    #     lr_start=LR_START, lr_end=LR_END,\n"
        "                    #     lr_decay_start=LR_DECAY_START,\n"
        "                    #     grad_clip=GRAD_CLIP,\n"
        "                    #     out_dir=spec_dir,\n"
        "                    # )\n"
        "                    raise NotImplementedError(\n"
        "                        'Wire to alec/train.py entry point at runtime.'\n"
        "                    )\n"
        "                    n_done += 1\n"
        "                except NotImplementedError as e:\n"
        "                    print(f'  [pending] {metric}/{tag}/{solver}: {e}')\n"
        "                    n_failed += 1\n"
        "                except Exception as e:\n"
        "                    print(f'  [fail] {metric}/{tag}/{solver}: {e}')\n"
        "                    n_failed += 1\n"
        "print(f'Done: {n_done}; Failed/pending: {n_failed}; Skipped: {n_skipped}')\n"
    ))
    cells.append(_md(
        "## 6. Post-Processing Analysis\n\n"
        "Generates 6 figures + headline.json from the 88 eval_df.csv files.\n"
    ))
    cells.append(_code(
        "import subprocess\n"
        "result = subprocess.run([\n"
        "    'python',\n"
        "    str(REPO / 'reports_local' / 'step7_subset_selection' / 'scripts' / 'run_post_processing.py'),\n"
        "], capture_output=True, text=True)\n"
        "print(result.stdout)\n"
        "if result.returncode != 0:\n"
        "    print('STDERR:', result.stderr)\n"
        "    raise RuntimeError(f'post-processing failed: exit {result.returncode}')\n"
    ))
    return cells


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = build_cells()
    NOTEBOOK_OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {NOTEBOOK_OUT}")


if __name__ == "__main__":
    main()

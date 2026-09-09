"""Build the merged family VIEW of the listed campaign runs for the figure suite.

The campaign runs as separate cluster runs (the size group, the family groups, the
meta-GGA group, the 25-cycle and dpyscf-parity arms), each with its own run directory,
while the figures show one architecture axis with every finished cell on it. This script
composes exactly that: ONE view directory of renumbered symlinks to the listed runs' spec
directories, a composed ``manifest.json`` whose cells carry the run's own cell plus the
entry's protocol tag, the certified pretrain directories of the runs that carry the
canonical pre-training, and the first run's ``resolved_config.yaml``. No data is copied
or modified, and every figure function reads the view as it reads a single run.

The runs are LISTED, not discovered: ``family_runs.yaml`` names each run (its category
under the results root's domain, ``latest`` or a ``run_*`` name), the registry keys to
take, the protocol tag shown on its cells' names, and whether its pre-training directory
is the canonical one. A new run enters the figures by one entry; a listed run that is
not pulled locally refuses the build, so the view never silently drops a run.

The guards of ``merge_v4_arms`` apply to every entry: seed provenance, PASS fidelity
certificates for every registry architecture with a cell, no workflow-verification
slice, no duplicate cell -- where the cell key carries the protocol tag, so an arm that
trains the same architecture under another protocol sits beside the untagged cell
instead of being refused as a double count. A later entry whose production identity
(basis, density fitting, grid level, parent anchor) differs from the first is refused,
tag or no tag: the view keeps one configuration, and a tag names a training protocol.

The view is written as ``<results_root>/<domain>/<view>/runs/run_<stamp>``, the stamp
being the newest run's that contributes a spec directory (so the suite's date-gated
provenance reads a campaign date; a family whose contributing runs straddle one of those
dates is refused), staged in a ``.building`` sibling and swapped in only once every entry
passed. Only directories carrying the view's own marker (``MERGED_RUNS.txt``) are ever
displaced or removed.

Usage:
    python notebooks/analysis/merge_family_runs.py [--family FILE] [--results-root DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
_MV_PATH = _HERE / "merge_v4_arms.py"
_mv_spec = importlib.util.spec_from_file_location("merge_v4_arms", _MV_PATH)
mv = importlib.util.module_from_spec(_mv_spec)  # type: ignore[arg-type]
sys.modules.setdefault("merge_v4_arms", mv)
_mv_spec.loader.exec_module(mv)  # type: ignore[union-attr]

DEFAULT_FAMILY = _HERE / "family_runs.yaml"
DEFAULT_ROOT = Path.home() / "Documents/Research/xcquinox-results/runs"
MARKER = "MERGED_RUNS.txt"
_RUN_NAME = re.compile(r"^run_\d{8}T\d{6}Z$")
_ENTRY_KEYS = ("category", "run", "archs", "protocol", "pretrain")
_IDENTITY_KEYS = ("basis:", "density_fit:", "grid_level:", "parent_anchor:",
                  "subset_ledger_path:")
#: the held-out channels the figure suite renders; a spec counts as evaluated
#: when it holds one of them (the cold-start channel is drawn inside figures,
#: never as a set of its own)
_EVAL_CHANNELS = ("eval_holdout", "eval_holdout_val_best", "eval_holdout_converged",
                  "eval_holdout_converged_val_best")
_SPEC_DIR = re.compile(r"^spec_(\d{4})$")


@dataclass(frozen=True)
class RunEntry:
    category: str
    run: str = "latest"
    archs: Optional[Tuple[str, ...]] = None
    protocol: Optional[str] = None
    pretrain: bool = True


@dataclass(frozen=True)
class FamilySpec:
    view: str
    domain: str
    entries: Tuple[RunEntry, ...] = field(default_factory=tuple)


def load_family_runs(path) -> FamilySpec:
    """Parse the run list; every defect names the entry it sits in."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the family list is a mapping with view, domain, runs")
    view, domain = raw.get("view"), raw.get("domain")
    if not isinstance(view, str) or not view or not isinstance(domain, str) or not domain:
        raise ValueError(f"{path}: view and domain are required non-empty strings")
    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{path}: runs is a non-empty list of entries")
    entries: List[RunEntry] = []
    seen: set = set()
    for i, e in enumerate(runs):
        where = f"{path}: runs[{i}]"
        if not isinstance(e, dict):
            raise ValueError(f"{where}: an entry is a mapping")
        unknown = sorted(set(e) - set(_ENTRY_KEYS))
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {unknown}; the keys are "
                             f"{list(_ENTRY_KEYS)}")
        category = e.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{where}: category is required")
        if category in seen:
            raise ValueError(f"{where} ({category}): the category is listed twice; a "
                             "view takes one run per category")
        seen.add(category)
        run = e.get("run", "latest")
        if not isinstance(run, str) or (run != "latest" and not _RUN_NAME.match(run)):
            raise ValueError(f"{where} ({category}): run is 'latest' or a "
                             f"run_YYYYMMDDTHHMMSSZ name, got {run!r}")
        archs = e.get("archs")
        if archs is not None:
            if (not isinstance(archs, list) or not archs
                    or not all(isinstance(a, str) and a for a in archs)):
                raise ValueError(f"{where} ({category}): archs is a non-empty list of "
                                 "registry keys")
            archs = tuple(archs)
        protocol = e.get("protocol")
        if protocol is not None and (not isinstance(protocol, str)
                                     or not protocol.strip()
                                     or "[" in protocol or "]" in protocol):
            raise ValueError(f"{where} ({category}): protocol is a non-empty string "
                             f"without brackets, got {protocol!r}")
        pretrain = e.get("pretrain", True)
        if not isinstance(pretrain, bool):
            raise ValueError(f"{where} ({category}): pretrain is a boolean")
        entries.append(RunEntry(category=category, run=run, archs=archs,
                                protocol=protocol.strip() if protocol else None,
                                pretrain=pretrain))
    return FamilySpec(view=view, domain=domain, entries=tuple(entries))


def resolve_run(results_root, domain: str, entry: RunEntry) -> Optional[Path]:
    """The entry's run directory, or None when it is not pulled locally."""
    base = Path(results_root) / domain / entry.category
    if entry.run == "latest":
        return mv.newest_run(base)
    run = base / "runs" / entry.run
    return run if run.is_dir() else None


def resolve_all(spec: FamilySpec, results_root) -> Dict[str, Path]:
    """``{category: run_dir}`` for every entry; every absent run is printed and one
    refusal names them all."""
    found: Dict[str, Path] = {}
    absent: List[str] = []
    for entry in spec.entries:
        run = resolve_run(results_root, spec.domain, entry)
        if run is None:
            where = Path(results_root) / spec.domain / entry.category / "runs"
            print(f"[merge] listed run {entry.category} ({entry.run}) is not pulled "
                  f"under {where}")
            absent.append(entry.category)
        else:
            found[entry.category] = run
    if absent:
        raise SystemExit(
            f"[merge] REFUSING: listed run(s) not pulled locally: {absent}; a view "
            "that silently dropped a listed run would misreport the campaign")
    return found


def _spec_index(sd: Path) -> int:
    return int(sd.name.split("_", 1)[1])


def _spec_dirs_of(run: Path) -> List[Path]:
    """The run's ``checkpoints/spec_NNNN`` directories, sorted; a stray entry under
    ``checkpoints/`` (a file, a partial directory) is refused by name, since it could
    neither be labelled nor linked."""
    ck = run / "checkpoints"
    if not ck.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(ck.iterdir()):
        if p.is_dir() and _SPEC_DIR.match(p.name):
            out.append(p)
        elif p.name.startswith("spec_"):
            raise SystemExit(
                f"[merge] REFUSING {run}: {p} is not a spec_NNNN directory; the "
                "checkpoints tree holds spec directories only")
    return out


def _has_specs(run: Path) -> bool:
    return bool(_spec_dirs_of(run))


def _evaluated(sd: Path) -> bool:
    """Whether the spec directory holds a held-out channel the suite renders."""
    return any((sd / ch / "per_reaction.json").is_file()
               or (sd / ch / "per_molecule.json").is_file() for ch in _EVAL_CHANNELS)


def _read_manifest(run: Path) -> dict:
    try:
        return json.loads((run / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _identity(cfg: Path):
    """The identity lines (basis, density_fit, grid_level, parent_anchor) of a
    resolved configuration, by stripped-line prefix, or None when unreadable."""
    try:
        lines = cfg.read_text().splitlines()
    except OSError:
        return None
    return tuple(next((ln.strip() for ln in lines if ln.strip().startswith(k)), None)
                 for k in _IDENTITY_KEYS)


def _era_guard(stamps: List[str]) -> None:
    """Refuse contributing runs that straddle a date-gated provenance cutoff of the
    figure suite (the V_xc correction, the fidelity gate): the view carries one stamp
    and the suite discloses one era for the whole set."""
    if len(stamps) < 2:
        return
    spec = importlib.util.spec_from_file_location(
        "make_ablation_arch_figure", _HERE / "make_ablation_arch_figure.py")
    fig = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules.setdefault("make_ablation_arch_figure", fig)
    spec.loader.exec_module(fig)  # type: ignore[union-attr]
    for name, pred in (("V_xc correction", fig._run_predates_vxc_fix),
                       ("fidelity gate", fig._run_predates_fidelity_gate)):
        sides = {s: pred(s) for s in stamps}
        if len(set(sides.values())) > 1:
            raise SystemExit(
                f"[merge] REFUSING: the contributing runs straddle the {name} cutoff "
                f"({sides}); the view carries one stamp and would disclose one era for "
                "cells of both")


def view_path(spec: FamilySpec, results_root, runs: Optional[Dict[str, Path]] = None
              ) -> Path:
    """``<results_root>/<domain>/<view>/runs/run_<stamp>``, the stamp being the newest
    listed run's that holds a spec directory."""
    runs = runs if runs is not None else resolve_all(spec, results_root)
    stamps = [r.name for r in runs.values() if _has_specs(r)]
    stamp = max(stamps) if stamps else "run_00000000T000000Z"
    return Path(results_root) / spec.domain / spec.view / "runs" / stamp


def _build_view_into(spec: FamilySpec, results_root, view_dir: Path,
                     runs: Dict[str, Path]) -> dict:
    """Populate ``view_dir``; returns {category: (run_name, n_specs, n_evaluated)}."""
    from xcquinox.alec.eval_holdout import assert_channel_not_sliced

    ck_out = view_dir / "checkpoints"
    ck_out.mkdir(parents=True)
    report: dict = {}
    merged_specs: List[dict] = []
    seen_cells: Dict[tuple, str] = {}
    fidelity_by_run: dict = {}
    contributing: List[str] = []
    slot_owner: Dict[str, str] = {}
    slot_from: Dict[str, Dict[str, str]] = {}
    model_block = None
    idx = 0
    for entry in spec.entries:
        run = runs[entry.category]
        entries = mv._arm_manifest_entries(run)
        manifest = _read_manifest(run)
        if model_block is None and isinstance(manifest.get("model"), dict):
            model_block = manifest["model"]
        spec_dirs = _spec_dirs_of(run)
        if entries and spec_dirs:
            pass
        elif spec_dirs:
            raise SystemExit(
                f"[merge] REFUSING {entry.category} {run.name}: {len(spec_dirs)} spec "
                "dir(s) on disk but manifest.json yields no usable entries -- their "
                "architectures are unknown, so no gate can be applied to them")
        elif not entries:
            print(f"[merge] note: {entry.category} {run.name} has no manifest and no "
                  "spec directory yet; it enters the figures as its cells land")
        if entry.archs is not None and entries:
            named = {(e.get("cell") or {}).get("arch") for e in entries.values()}
            missing = [a for a in entry.archs if a not in named]
            if missing:
                raise SystemExit(
                    f"[merge] REFUSING {entry.category} {run.name}: archs {missing} are "
                    f"not architectures of its manifest ({sorted(a for a in named if a)}); "
                    "an unmatched name would drop the entry's cells silently")
            keep = set(entry.archs)
            entries = {i: e for i, e in entries.items()
                       if (e.get("cell") or {}).get("arch") in keep}
            n_before = len(spec_dirs)
            spec_dirs = [sd for sd in spec_dirs if _spec_index(sd) in entries]
            dropped = n_before - len(spec_dirs) - sum(
                1 for i in mv._arm_manifest_entries(run) if i not in entries
                and (run / "checkpoints" / f"spec_{i:04d}").is_dir())
            if dropped:
                print(f"[merge] WARNING: {entry.category} {run.name}: {dropped} unlabeled "
                      "spec dir(s) skipped under the archs restriction")
        unlabeled = [_spec_index(sd) for sd in spec_dirs if _spec_index(sd) not in entries]
        if unlabeled:
            print(f"[merge] WARNING: {entry.category} {run.name} manifest lacks entries "
                  f"for {len(unlabeled)} on-disk spec dir(s) -- they merge without labels")
        arch_specs: dict = {}
        for entry_idx, rec_ in sorted(entries.items()):
            arch_specs.setdefault((rec_.get("cell") or {}).get("arch"), []).append(entry_idx)
        # the entry's own architectures are gated whether or not its pretrain
        # directories are carried; a run with no cells has nothing to gate
        mv._validate_arm_seed_policy(run, arch_specs)
        cert_status = {
            a: st for a, (st, _reason, _path, _payload)
            in mv._validate_arm_fidelity_certificates(
                run, arch_specs, arm=entry.category).items()}
        by_run_record = dict(cert_status)
        if entry.pretrain and cert_status:
            mv._carry_arm_certificates(run, view_dir, cert_status, entry.category)
            for a in cert_status:
                slot_owner.setdefault(a, entry.category)
        elif cert_status:
            # not carried: every gated architecture must already have a slot,
            # since the figures read the certificate by the view's slot
            without = [a for a in sorted(cert_status) if a not in slot_owner]
            if without:
                raise SystemExit(
                    f"[merge] REFUSING {entry.category} {run.name}: pretrain is false "
                    f"but {without} have no carried pretrain slot in the view; the "
                    "figures would read those architectures as uncertified")
            slot_from[entry.category] = {a: slot_owner[a] for a in sorted(cert_status)}
        fidelity_by_run[entry.category] = by_run_record
        run_id = _identity(run / "resolved_config.yaml")
        view_cfg = view_dir / "resolved_config.yaml"
        view_id = _identity(view_cfg) if view_cfg.is_file() else None
        if view_id is not None and run_id is not None and run_id != view_id:
            diff = ", ".join(f"{a} vs {b}" for a, b in zip(view_id, run_id) if a != b)
            raise SystemExit(
                f"[merge] REFUSING {entry.category} {run.name}: production identity "
                f"differs from the view's ({diff}); the view keeps one configuration "
                "(basis label, references, anchor), and a protocol tag names a training "
                "protocol, not an identity")
        for sd in spec_dirs:
            for chan in sorted(p.name for p in sd.glob("eval_holdout*") if p.is_dir()):
                assert_channel_not_sliced(sd, chan)
            orig_idx = _spec_index(sd)
            entry_rec = entries.get(orig_idx, {})
            cell = dict(entry_rec.get("cell") or {})
            if entry.protocol is not None:
                cell["protocol"] = entry.protocol
            key = (cell.get("arch"), entry.protocol, cell.get("subset_size"))
            if key[0] is not None and key[2] is not None:
                owner = seen_cells.get(key)
                if owner is not None:
                    where = (f"twice from {owner}" if owner == entry.category
                             else f"from both {owner} and {entry.category}")
                    raise SystemExit(
                        f"[merge] REFUSING: cell {key} arrives {where} -- a duplicate "
                        "cell is a double count, never a merge (a run trained under "
                        "another protocol needs a protocol tag on its entry)")
                seen_cells[key] = entry.category
            (ck_out / f"spec_{idx:04d}").symlink_to(sd.resolve())
            arch_name = cell.get("arch")
            rec = {"index": idx, "cell": cell,
                   "category": entry.category, "run": run.name, "source_index": orig_idx,
                   "fidelity_status": ("UNLABELED" if not arch_name
                                       else cert_status.get(arch_name, "NOT_IN_REGISTRY"))}
            for k in ("spec_file", "sha256"):
                if entry_rec.get(k) is not None:
                    rec[k] = entry_rec[k]
            merged_specs.append(rec)
            idx += 1
        if spec_dirs:
            contributing.append(entry.category)
        n_eval = sum(1 for sd in spec_dirs if _evaluated(sd))
        report[entry.category] = (run.name, len(spec_dirs), n_eval)
        cert_note = (", ".join(f"{a}={s}" for a, s in by_run_record.items())
                     if by_run_record else "no cells yet")
        with open(view_dir / MARKER, "a") as f:
            f.write(f"{entry.category}\t{run.name}\t{len(spec_dirs)} specs\t{n_eval} "
                    f"evaluated\tprotocol: {entry.protocol or '-'}\tpretrain: "
                    f"{'carried' if entry.pretrain else 'slot carried from ' + ', '.join(sorted(set(slot_from.get(entry.category, {}).values())) or ['-'])}"
                    f"\tfidelity: {cert_note}\n")
        for src in [run / "resolved_config.yaml", *sorted(run.glob("scan_pool_*.json"))]:
            dst = view_dir / src.name
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
    out = {"n_specs": idx, "specs": merged_specs,
           "merged_from": contributing,
           "family_runs": [{"category": e.category, "run": e.run,
                            "archs": list(e.archs) if e.archs else None,
                            "protocol": e.protocol, "pretrain": e.pretrain}
                           for e in spec.entries],
           "resolved": {c: r.name for c, r in runs.items()},
           "fidelity": {
               "policy": ("record layer: every registry architecture with a cell "
                          "carries a PASS pretraining-fidelity certificate in its own "
                          "run; a slot carried from an earlier entry is recorded as "
                          "such; an enforced=false waiver is refused"),
               "by_run": fidelity_by_run,
               # the slot an entry's architectures are read against when its
               # own pretrain directories are not carried
               "slot_from": slot_from,
               # counted, not asserted: the gate refuses every non-PASS, so a
               # waived status can only appear if the gate changes
               "n_waived": sum(1 for statuses in fidelity_by_run.values()
                               for s in statuses.values()
                               if str(s).lower().startswith("waived"))}}
    if model_block is not None:
        out["model"] = model_block
    (view_dir / "manifest.json").write_text(json.dumps(out, indent=1))
    return report


def build_view(spec: FamilySpec, results_root) -> dict:
    """(Re)build the view at :func:`view_path`; returns the report
    ``{category: (run_name, n_specs, n_evaluated)}``.

    Staged in ``<name>.building`` and swapped in once every entry passed; only a
    directory carrying the view marker is displaced or removed, so a directory that is
    not a view of this script is never touched."""
    results_root = Path(results_root)
    runs = resolve_all(spec, results_root)
    _era_guard([r.name for r in runs.values() if _has_specs(r)])
    out_dir = view_path(spec, results_root, runs)
    if (out_dir.exists() or out_dir.is_symlink()) and not (out_dir / MARKER).is_file():
        raise SystemExit(
            f"[merge] REFUSING: {out_dir} exists and carries no {MARKER}; it is not a "
            "view of this script and is left alone")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.parent / f"{out_dir.name}.building"
    mv._remove_path(staging)
    try:
        report = _build_view_into(spec, results_root, staging, runs)
    except BaseException:
        try:
            mv._remove_path(staging)
        except OSError:
            pass
        raise
    previous = None
    if out_dir.exists() or out_dir.is_symlink():
        previous = out_dir.parent / f"{out_dir.name}.previous"
        mv._remove_path(previous)
        out_dir.rename(previous)
    staging.rename(out_dir)
    if previous is not None:
        mv._remove_path(previous)
    # the view is derived: one marked directory at a time under <view>/runs/
    for other in sorted(out_dir.parent.glob("run_*")):
        if other != out_dir and (other / MARKER).is_file():
            mv._remove_path(other)
    return report


def evaluated_cells(view_dir) -> Dict[tuple, int]:
    """``{(category, arch, protocol, subset_size): view index}`` over the view's
    evaluated specs, ``arch`` being the registry key the composed manifest holds."""
    view_dir = Path(view_dir)
    out: Dict[tuple, int] = {}
    for rec in _read_manifest(view_dir).get("specs", []):
        idx = rec.get("index")
        sd = view_dir / "checkpoints" / f"spec_{int(idx):04d}"
        if not sd.is_dir() or not _evaluated(sd):
            continue
        cell = rec.get("cell") or {}
        out[(rec.get("category"), cell.get("arch"), cell.get("protocol"),
             cell.get("subset_size"))] = int(idx)
    return out


def assert_view_complete(spec: FamilySpec, results_root, view_dir) -> None:
    """Refuse (SystemExit) a view lacking an evaluated cell of a listed run: a run whose
    cells do not reach the figures fails here, never silently. A source run that grew
    since the view was built fails the same way: the view is stale and is rebuilt."""
    runs = resolve_all(spec, results_root)
    have = evaluated_cells(view_dir)
    missing: List[tuple] = []
    for entry in spec.entries:
        run = runs[entry.category]
        for i, rec in sorted(mv._arm_manifest_entries(run).items()):
            cell = rec.get("cell") or {}
            if entry.archs is not None and cell.get("arch") not in entry.archs:
                continue
            sd = run / "checkpoints" / f"spec_{i:04d}"
            if not sd.is_dir() or not _evaluated(sd):
                continue
            key = (entry.category, cell.get("arch"), entry.protocol, cell.get("subset_size"))
            if key not in have:
                missing.append(key)
    if missing:
        raise SystemExit(
            f"[merge] the view {view_dir} lacks evaluated cell(s) of listed runs "
            f"(category, arch, protocol, subset_size): {missing}; the view is stale or "
            "damaged -- rebuild it")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--family", default=str(DEFAULT_FAMILY))
    p.add_argument("--results-root", default=str(DEFAULT_ROOT))
    args = p.parse_args(argv)
    spec = load_family_runs(args.family)
    root = Path(args.results_root)
    report = build_view(spec, root)
    view_dir = view_path(spec, root)
    assert_view_complete(spec, root, view_dir)
    total = 0
    for entry in spec.entries:
        run, n, n_eval = report[entry.category]
        print(f"[merge] {entry.category:<40} {run:<24} {n:>3} specs {n_eval:>3} evaluated"
              f"  protocol: {entry.protocol or '-'}")
        total += n_eval
    print(f"[merge] view: {view_dir}  ({total} evaluated cells)")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())

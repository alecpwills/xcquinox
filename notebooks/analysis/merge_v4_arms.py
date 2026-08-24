"""Build a merged VIEW of the v4 campaign arms for cross-arm figures.

The campaign runs as separate arms (the GGA/rung-3.5 v4 arm plus the two
SCAN-seeded v5 mgga arms; the PBE-seeded v4 mgga arms are retired and
excluded), each with its own run dir.
The figure collectors scan ``<run_dir>/checkpoints/spec_*``, so a merged
9-arch figure needs one directory whose spec dirs span every arm. This
script builds exactly that: a view directory of RENUMBERED SYMLINKS to the
arms' spec dirs -- no data is copied or modified, and every existing figure
function works on the view unchanged. Arms whose run dirs do not exist yet
are skipped, so the view grows as the campaign lands.

The view is rebuilt from scratch on every invocation (idempotent); its name
carries no ``run_YYYYMMDDT`` stamp, so the V_xc-provenance figure layer
conservatively draws no pre-correction marks on it -- correct, since every
arm postdates the correction.

Usage:
    python notebooks/analysis/merge_v4_arms.py [--results-root DIR]
                                               [--out DIR]

Default results root: ~/Documents/Research/xcquinox-results/runs/dfs_step7
(the pull target of pull_and_plot_v4.sh). The newest run under each arm's
``runs/`` directory is used.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

# v5 era (2026-08-14): the retired v4 mgga arms (PBE-seeded, cancelled
# mid-array) are EXCLUDED; the roster is the still-valid GGA/rung-3.5 arm
# plus the two SCAN-seeded v5 arms. Per-arch seed provenance is VALIDATED
# against the rung-baseline policy, and every registry architecture must
# carry a PASS pretraining-fidelity certificate, before an arm enters the
# view.
ARM_BASES = ("dfs6311_grid3_v4gga", "dfs6311_grid3_v5",
             "dfs6311_grid3_v5mgga2")
DEFAULT_ROOT = Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7"


def newest_run(base_dir: Path) -> Path | None:
    """The lexically newest ``run_*`` under ``<base>/runs`` (timestamps sort)."""
    runs = base_dir / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(d for d in runs.iterdir()
                        if d.is_dir() and d.name.startswith("run_"))
    return candidates[-1] if candidates else None


_IDENTITY_KEYS = ("basis:", "density_fit:", "grid_level:")


def _config_identity(cfg: Path):
    """The production-identity lines of a resolved_config.yaml (basis /
    density_fit / grid_level), or None when the file is absent/unreadable."""
    try:
        lines = cfg.read_text().splitlines()
    except OSError:
        return None
    return tuple(next((ln.strip() for ln in lines
                       if ln.strip().startswith(k)), None)
                 for k in _IDENTITY_KEYS)


def _arm_manifest_entries(run: Path) -> dict:
    """{original_index: full manifest entry} from the arm's manifest.json
    (empty if absent/unreadable). Full entries so the merged manifest can
    carry the spec_file/sha256 provenance fields through."""
    mpath = run / "manifest.json"
    if not mpath.is_file():
        return {}
    try:
        manifest = json.loads(mpath.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {e["index"]: e
            for e in manifest.get("specs", []) if isinstance(e.get("index"), int)}


def _validate_arm_seed_policy(run: Path, arch_names) -> None:
    """Refuse an arm whose resolved seed diverges from the rung-baseline
    policy for any REGISTRY arch it carries.

    The grouped figures must only ever assemble correctly seeded data: a
    PBE-seeded mgga arm (the retired v4 protocol) resolves seed 'pbe'
    where the policy demands 'scan' and is rejected here, not silently
    merged. Archs not in the registry (test fixtures, legacy names) have
    no policy expectation and are skipped; a registry arch WITHOUT a
    loadable resolved_config.yaml is unverifiable and also refused.
    """
    from xcquinox.alec.rungs import seed_xc_for_arch
    registry_archs = {}
    for a in arch_names:
        if not a:
            continue
        try:
            registry_archs[a] = seed_xc_for_arch(a)
        except KeyError:
            continue
    if not registry_archs:
        return
    try:
        from xcquinox.alec.cluster.grid_config import load_grid_config
        from xcquinox.alec.cluster.spec_builder import resolve_seed_xc
        cfg = load_grid_config(str(run / "resolved_config.yaml"))
    except Exception as exc:  # noqa: BLE001 -- unverifiable = refused
        raise SystemExit(
            f"[merge] REFUSING {run}: cannot verify seed provenance for "
            f"registry archs {sorted(registry_archs)} "
            f"({type(exc).__name__}: {exc})")
    for arch, expected in sorted(registry_archs.items()):
        got = resolve_seed_xc(cfg.inputs, arch)
        if got != expected:
            raise SystemExit(
                f"[merge] REFUSING {run}: arch {arch} resolves seed "
                f"{got!r} but the rung-baseline policy demands "
                f"{expected!r} -- a mis-seeded arm cannot enter the "
                "grouped figures")


def _certificate_status_label(status: str, payload) -> str:
    """The status a refusal should print, naming a waiver as one.

    A FAIL that records ``enforced: false`` was released on its own node by
    ``fidelity.gate_certificate`` -- the workflow-verification matrix, whose
    short pretraining cannot meet the tolerance. It is refused here like any
    other FAIL, but a reader who sees only "FAIL" cannot tell a run that was
    never meant to certify from an architecture whose physics did not. The
    same four labels are printed by
    ``make_ablation_arch_figure._certificate_status_label``, so one run
    carries one vocabulary across both record layers.
    """
    if (status == "FAIL" and isinstance(payload, dict)
            and payload.get("enforced") is False):
        return "waived FAIL"
    return status


def _arm_certificate_statuses(run: Path, arch_names) -> dict:
    """``{arch: (status, reason, certificate_path, payload)}`` for the arm's
    REGISTRY architectures, sorted by name.

    Architectures the registry does not know (test fixtures, legacy display
    names) carry no certificate expectation and are absent from the mapping,
    matching :func:`_validate_arm_seed_policy`. ``arch_names`` may be any
    iterable of names, including a ``{arch: [spec index, ...]}`` mapping,
    which iterates over its architectures.

    The classification and the document travel together, from ONE parse: a
    caller that classifies through one read and re-opens the file for the
    records it re-checks judges one document and reports another, and a
    certificate rewritten between the two opens then refuses -- or admits --
    an arm over a file that never existed as a whole. ``payload`` is the
    parsed object, or ``None`` when the file is absent or is not one.
    """
    from xcquinox.alec.cluster.fidelity import (certificate_path,
                                                read_certificate_status_in)
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    from xcquinox.alec.config import get_architecture
    statuses = {}
    for arch in arch_names:
        if not arch or arch in statuses:
            continue
        try:
            get_architecture(arch)
        except KeyError:
            continue
        status, reason, payload = read_certificate_status_in(
            pretrain_checkpoint_dir(str(run), arch))
        statuses[arch] = (status, reason, certificate_path(str(run), arch),
                          payload)
    return dict(sorted(statuses.items()))


def _validate_arm_fidelity_certificates(run: Path, arch_names,
                                        arm: str | None = None) -> dict:
    """Refuse an arm whose REGISTRY architectures lack a PASS certificate.

    The per-architecture physics certificate
    (``xcquinox.alec.cluster.fidelity``) is the only machine-checked statement
    that an architecture's pretrained networks reproduce their parent
    functional. Without it the arm's held-out numbers cannot be read against
    the parent baselines the grouped figures draw, so an uncertified, failed
    or unreadable arm is refused here rather than silently merged. Runs that
    predate the certificate hold none, and are refused on the same rule: the
    absence of a measurement is not a measurement that passed.

    This is a RECORD layer: it requires PASS from the certificate and
    ignores the certificate's ``enforced`` field, which releases the ON-NODE
    gates of a workflow-verification run only (``fidelity.gate_certificate``).
    No waiver is accepted here, so no status other than PASS can reach a built
    view.

    Four further records on a PASS certificate are re-checked against the
    arm on disk, because ``validate_run`` -- which imposes them on the
    cluster -- is not part of the pull pipeline, so on pulled data this is
    the only place they are read:

    * ``arch``: the certificate is located by DIRECTORY, so a file copied
      from another architecture's pretrain dir would otherwise certify this
      one.
    * ``parent``: the functional an architecture must reproduce follows its
      RUNG (``fidelity.resolve_parent``); a certificate measured against the
      other one bounds nothing here.
    * ``checkpoint.xnet_sha256`` / ``cnet_sha256``: the digests of the two
      files the verdict refers to, recomputed with the writer's own
      ``materialize._sha256_file``. A checkpoint rewritten or re-pretrained
      after certification is not the one that was measured.
    * ``identity``: the fields ``fidelity.run_identity`` defines (basis, grid
      level, the Coulomb backend, the orientation-lock strength), compared
      against the arm's ``resolved_config.yaml``. A PASS measured at another
      SCF identity does not describe the SCF the arm's held-out numbers come
      from.

    Each comparison is made over the keys the certificate ITSELF records: the
    writer emits all four (and all five identity fields), so a real
    certificate is fully compared, while a record that is absent states
    nothing to contradict and is not read as agreement. ``validate_run`` is
    stricter on its own run -- it treats an absent identity key as a mismatch,
    since there the config is the authority on what must have been measured.
    An arm whose config cannot be loaded is already refused above for every
    registry architecture by :func:`_validate_arm_seed_policy`.

    ``arch_names`` may be a ``{arch: [spec index, ...]}`` mapping, in which
    case a refusal names the spec directories the architecture owns.

    Every check reads the certificate the classification came from -- one
    parse per architecture -- and the statuses validated are RETURNED, so the
    view records the statuses this guard acted on rather than re-reading the
    files to record them.
    """
    from xcquinox.alec.cluster.fidelity import (VERDICT_PASS, resolve_parent,
                                                run_identity)
    from xcquinox.alec.cluster.grid_config import (load_grid_config,
                                                   pretrain_checkpoint_dir)
    from xcquinox.alec.cluster.materialize import _sha256_file
    statuses = _arm_certificate_statuses(run, arch_names)
    if not statuses:
        return statuses
    label = f"{arm or run.parent.parent.name} {run.name}"
    spec_indices = arch_names if isinstance(arch_names, Mapping) else {}
    for arch, (status, reason, path, payload) in statuses.items():
        if status == VERDICT_PASS:
            continue
        owned = sorted(spec_indices.get(arch) or [])
        where = (" (" + ", ".join(f"spec_{i:04d}" for i in owned) + ")"
                 if owned else "")
        raise SystemExit(
            f"[merge] REFUSING {label}: arch {arch}{where} has no PASS "
            f"pretraining-fidelity certificate -- "
            f"{_certificate_status_label(status, payload)} at {path} "
            f"({reason}) -- an uncertified arm cannot enter the grouped "
            "figures")
    try:
        expected_identity = run_identity(
            load_grid_config(str(run / "resolved_config.yaml")))
    except Exception:  # noqa: BLE001 -- refused above where it matters
        expected_identity = None

    def refuse(arch, path, detail):
        """Refuse the arm over one architecture's record; never returns."""
        raise SystemExit(
            f"[merge] REFUSING {label}: the pretraining-fidelity certificate "
            f"for arch {arch} at {path} {detail}")

    for arch, (_status, _reason, path, read_payload) in statuses.items():
        pretrain_dir = pretrain_checkpoint_dir(str(run), arch)
        # The document the PASS above was read from, not a fresh open of the
        # same path: the verdict acted on and the records re-checked have to
        # describe one file.
        payload = read_payload or {}
        named = payload.get("arch")
        if named is not None and named != arch:
            refuse(arch, path,
                   f"names arch {named!r} -- it does not certify this "
                   "architecture; the certificate is located by directory, "
                   "so a file copied from another arch's pretrain dir would "
                   "otherwise pass as this one's")
        recorded_parent = payload.get("parent")
        if recorded_parent is not None:
            wanted_parent = resolve_parent(arch)
            if recorded_parent != wanted_parent:
                refuse(arch, path,
                       f"records parent {recorded_parent!r}, but this "
                       f"architecture's rung is pretrained against "
                       f"{wanted_parent!r} -- a certificate measured against "
                       "the other functional bounds nothing about the "
                       "distance this one had to close")
        checkpoint = payload.get("checkpoint")
        if isinstance(checkpoint, dict):
            for field, fname in (("xnet_sha256", "xnet.eqx"),
                                 ("cnet_sha256", "cnet.eqx")):
                want = checkpoint.get(field)
                if not isinstance(want, str) or not want:
                    continue
                fpath = os.path.join(pretrain_dir, fname)
                try:
                    got = _sha256_file(fpath)
                except OSError as exc:
                    refuse(arch, path,
                           f"records {field} for {fname}, which cannot be "
                           f"read from the pretrain directory "
                           f"({type(exc).__name__}: {exc}) -- the verdict "
                           "refers to a file that is not there")
                else:
                    if got != want:
                        refuse(arch, path,
                               f"records {field} {want} for {fname}, but "
                               f"the file on disk hashes to {got} -- the "
                               "checkpoint was rewritten or re-pretrained "
                               "after certification, so the verdict does not "
                               "refer to the networks the train stage loads")
        recorded = payload.get("identity")
        if expected_identity is None or not isinstance(recorded, dict):
            continue
        differing = {k: (v, expected_identity[k]) for k, v in recorded.items()
                     if k in expected_identity and v != expected_identity[k]}
        if differing:
            shown = ", ".join(f"{k}: certificate {c!r} vs run {r!r}"
                              for k, (c, r) in sorted(differing.items()))
            refuse(arch, path,
                   f"was measured at a different run identity than the arm "
                   f"itself ({shown}) -- its energies do not describe this "
                   "arm's SCF")
    return statuses


def _remove_path(path: Path) -> None:
    """Remove a directory tree, or just the LINK when ``path`` is a symlink.

    ``shutil.rmtree`` refuses a symbolic link outright, and the view path can
    be one (a view parked on another filesystem and reached through a link).
    Unlinking removes the link and never touches what it points at, which is
    the right outcome for both: the staged tree is ours to delete, and the
    displaced view is the user's to keep. A path that is neither is left
    alone, so the call is safe to make unconditionally.
    """
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _carry_arm_certificates(run: Path, view_dir: Path, arch_names,
                            arm: str) -> None:
    """Link the arm's GATED pretrain directories into ``<view_dir>/pretrain``.

    The merged directory runs no pretrain stage, so the arms' per-arch
    certificates travel with the merge: the figure layer resolves them through
    the same ``<run_dir>/pretrain/<arch>`` layout it uses for a single-arm run.

    Only the architectures the arm's manifest names AND the gate cleared are
    carried. ``<run>/pretrain`` can also hold directories no cell of this run
    references -- an architecture from an earlier submission, one whose specs
    were dropped, one pretrained before the sweep was cut down -- and those
    were never gated: linking one would put an unchecked, possibly FAILED
    certificate in the view under a name the figure layer reads, and would
    take the slot of the arm that actually ran that architecture. Each
    directory's own certificate is re-read here as the link-time precondition,
    so what the view exposes is verified at the point of exposure and not only
    inferred from the gate above.

    The view has ONE slot per architecture name. When a second arm brings the
    same name, the two certificates are compared: differing verdicts or
    differing recorded identities are refused, since neither record can stand
    for the other. Agreeing certificates still describe SEPARATELY pretrained
    networks, so the first arm's is kept and the collision is reported --
    the numbers and checkpoint digests the figure layer reads are that arm's.
    """
    from xcquinox.alec.cluster.fidelity import (VERDICT_PASS,
                                                read_certificate_status_in)
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    if not arch_names:
        return
    pt_out = view_dir / "pretrain"
    pt_out.mkdir(exist_ok=True)
    for arch in sorted(arch_names):
        src = Path(pretrain_checkpoint_dir(str(run), arch))
        # One parse per certificate here too: the precondition that releases
        # the link and the identity compared below are the same document.
        status, reason, payload = read_certificate_status_in(str(src))
        if status != VERDICT_PASS:
            raise SystemExit(
                f"[merge] REFUSING {arm} {run.name}: the pretrain directory "
                f"{src} does not hold a PASS pretraining-fidelity "
                f"certificate ({status}: {reason}) -- only a certified "
                "directory is carried into the view")
        dst = pt_out / arch
        # is_symlink() as well as exists(): a link whose target went away
        # mid-build reports False from exists() alone, and symlink_to would
        # then raise on it.
        if not (dst.exists() or dst.is_symlink()):
            dst.symlink_to(src.resolve())
            continue
        incumbent = dst.resolve()
        if incumbent == src.resolve():
            continue
        inc_status, inc_reason, inc_payload = read_certificate_status_in(
            str(incumbent))
        if inc_status != status:
            raise SystemExit(
                f"[merge] REFUSING {arm} {run.name}: arch {arch} is carried "
                f"by more than one arm and the two certificates disagree "
                f"({incumbent}: {inc_status}; {src}: {status} -- "
                f"{inc_reason}) -- the view has one pretrain slot per arch, "
                "so one arm's specs would be read against the other's record")
        inc_identity = (inc_payload or {}).get("identity")
        new_identity = (payload or {}).get("identity")
        if (isinstance(inc_identity, dict) and isinstance(new_identity, dict)
                and inc_identity != new_identity):
            keys = sorted(set(inc_identity) | set(new_identity))
            shown = ", ".join(
                f"{k}: {inc_identity.get(k)!r} vs {new_identity.get(k)!r}"
                for k in keys if inc_identity.get(k) != new_identity.get(k))
            raise SystemExit(
                f"[merge] REFUSING {arm} {run.name}: arch {arch} is certified "
                f"in more than one arm at DIFFERENT run identities "
                f"({incumbent} vs {src}: {shown}) -- neither certificate "
                "stands for the other, and the view has one pretrain slot "
                "per arch")
        print(f"[merge] WARNING: arch {arch} is certified in more than one "
              f"arm; the view links {incumbent} and not {src} -- the two "
              "verdicts agree, but the certificate numbers and checkpoint "
              "digests the figure layer reads are the first arm's")


def build_view(results_root: Path, out_dir: Path) -> dict:
    """(Re)build the merged view; returns {arm_base: (run_name, n_specs)}.

    The view is rebuilt from scratch on every invocation, and every guard --
    seed provenance, the fidelity certificates, the sliced-channel predicate,
    the duplicate-cell rule -- can refuse part way through an arm. The rebuild
    is therefore STAGED in a sibling ``<name>.building`` directory and swapped
    in only once every arm has passed: a refusal leaves the view already on
    disk exactly as it was, instead of destroying the last good one (the live
    merged view carries the whole campaign's spec links, and the figure suite
    reads it). The swap moves the old view aside before renaming the new one
    into place, so an interruption leaves ``<name>.previous`` to recover from
    rather than nothing at all.

    The staged directory is removed on refusal, so no half-populated view is
    left for a later figure run to read as complete.

    A view path that is itself a SYMLINK is displaced by the swap like any
    other: the link is moved aside and then unlinked (never followed into a
    tree deletion), so the directory it pointed at survives untouched and the
    path becomes the real view directory.
    """
    out_dir = Path(out_dir)
    staging = out_dir.parent / f"{out_dir.name}.building"
    _remove_path(staging)
    try:
        report = _build_view_into(results_root, staging)
    except BaseException:
        # SystemExit included: a refusal must not leave the staged tree.
        try:
            _remove_path(staging)
        except OSError:
            pass
        raise
    previous = None
    if out_dir.exists() or out_dir.is_symlink():
        previous = out_dir.parent / f"{out_dir.name}.previous"
        _remove_path(previous)
        out_dir.rename(previous)
    staging.rename(out_dir)
    if previous is not None:
        _remove_path(previous)
    return report


def _build_view_into(results_root: Path, view_dir: Path) -> dict:
    """Populate ``view_dir``; returns {arm_base: (run_name, n_specs)}.

    Alongside the renumbered spec symlinks a merged ``manifest.json`` is
    composed from the arms' own manifests -- the figure collectors join
    rows against it for the arch/subset labels, so without it every row
    would carry ``arch=None`` and the merged figures would be empty.

    Every guard here raises before :func:`build_view` swaps the directory
    into place, so a refused rebuild leaves the view already on disk
    untouched.
    """
    # Imported here, not at module scope: this script is filesystem work and
    # runs without the training package otherwise.
    from xcquinox.alec.eval_holdout import assert_channel_not_sliced

    ck_out = view_dir / "checkpoints"
    ck_out.mkdir(parents=True)

    report: dict = {}
    merged_specs = []
    seen_cells: dict = {}
    fidelity_by_arm: dict = {}
    idx = 0
    for base in ARM_BASES:
        run = newest_run(results_root / base)
        if run is None:
            report[base] = (None, 0)
            continue
        entries = _arm_manifest_entries(run)
        spec_dirs = sorted((run / "checkpoints").glob("spec_*")) \
            if (run / "checkpoints").is_dir() else []
        unlabeled = [int(sd.name.split("_", 1)[1]) for sd in spec_dirs
                     if int(sd.name.split("_", 1)[1]) not in entries]
        # A spec dir with no manifest entry merges with no arch/subset
        # labels, no duplicate-cell key, and no seed or certificate
        # validation. With NO entries at all that is the whole arm: its
        # architectures are unknown, so neither gate can be applied to a
        # single spec, while the view's own record would report the arm as
        # fully covered (an empty by_arm map beside a policy line asserting
        # universal PASS coverage). A pull interrupted mid-rsync reaches
        # exactly this state, so it is refused. An arm that has not
        # materialized anything yet -- no manifest AND no spec dirs, the
        # state of a freshly submitted arm -- has nothing to gate and gets a
        # low-key note.
        if not entries:
            if spec_dirs:
                raise SystemExit(
                    f"[merge] REFUSING {base} {run.name}: "
                    f"{len(spec_dirs)} spec dir(s) on disk but manifest.json "
                    "yields no usable entries (missing, unreadable, or "
                    "empty) -- their architectures are unknown, so neither "
                    "the seed-provenance nor the pretraining-fidelity gate "
                    "can be applied to them, and an ungated arm cannot enter "
                    "the grouped figures")
            print(f"[merge] note: {base} {run.name} has no manifest yet "
                  "-- seed-provenance validation deferred until the arm "
                  "materializes")
        elif unlabeled:
            shown = ", ".join(str(i) for i in unlabeled[:8])
            more = ("" if len(unlabeled) <= 8
                    else f", +{len(unlabeled) - 8} more")
            print(f"[merge] WARNING: {base} {run.name} manifest lacks "
                  f"entries for {len(unlabeled)} on-disk spec dir(s) "
                  f"(indices {shown}{more}) -- they merge without arch/"
                  "subset labels, duplicate-cell protection, or seed "
                  "validation")
        # {arch: [spec index, ...]} rather than a bare set of names, so a
        # refusal can name the spec dirs the architecture owns.
        arch_specs: dict = {}
        for entry_idx, entry_rec in sorted(entries.items()):
            arch_specs.setdefault(
                (entry_rec.get("cell") or {}).get("arch"), []).append(entry_idx)
        _validate_arm_seed_policy(run, arch_specs)
        cert_status = {
            a: st for a, (st, _reason, _path, _payload)
            in _validate_arm_fidelity_certificates(
                run, arch_specs, arm=base).items()}
        fidelity_by_arm[base] = cert_status
        _carry_arm_certificates(run, view_dir, cert_status, base)
        for sd in spec_dirs:
            # A workflow-verification slice covers a handful of species, not
            # the held-out pool; merged into the view it would average into a
            # cell as though it were a full-pool eval. The WHOLE spec dir is
            # symlinked below, so every held-out channel it carries enters
            # the view -- the channel set is read off disk rather than
            # fixed here, so a channel added later is covered by
            # construction. Refused before anything is counted or linked
            # (see eval_holdout, spec 3.4).
            for chan in sorted(p.name for p in sd.glob("eval_holdout*")
                               if p.is_dir()):
                assert_channel_not_sliced(sd, chan)
            orig_idx = int(sd.name.split("_", 1)[1])
            entry = entries.get(orig_idx, {})
            cell = entry.get("cell") or {}
            cell_key = (cell.get("arch"), cell.get("subset_size"))
            if all(v is not None for v in cell_key):
                owner = seen_cells.get(cell_key)
                if owner is not None and owner != base:
                    raise SystemExit(
                        f"[merge] REFUSING: cell {cell_key} arrives from "
                        f"both {owner} and {base} -- a duplicate cell is a "
                        "double-count, never a merge")
                seen_cells[cell_key] = base
            (ck_out / f"spec_{idx:04d}").symlink_to(sd.resolve())
            # The view's own record of what each spec was admitted under.
            # UNLABELED: no manifest entry, so no architecture to certify;
            # NOT_IN_REGISTRY: an arch the registry does not know, which
            # carries no certificate expectation (see
            # _arm_certificate_statuses). Neither is a PASS.
            arch_name = cell.get("arch")
            rec = {"index": idx, "cell": cell,
                   "arm": base, "arm_run": run.name, "arm_index": orig_idx,
                   "fidelity_status": (
                       "UNLABELED" if not arch_name
                       else cert_status.get(arch_name, "NOT_IN_REGISTRY"))}
            # Carry the integrity provenance through: spec_file/sha256 are
            # the expected hash record, and arm_run names the source run so
            # a verifier can resolve <arm>/runs/<arm_run>/specs/<spec_file>
            # without guessing the newest run (which moves as pulls land).
            for k in ("spec_file", "sha256"):
                if entry.get(k) is not None:
                    rec[k] = entry[k]
            merged_specs.append(rec)
            idx += 1
        report[base] = (run.name, len(spec_dirs))
        # Keep one provenance breadcrumb per arm; the eval count separates
        # finished cells from empty/mid-training spec dirs.
        n_eval = sum(
            1 for sd in spec_dirs
            if (sd / "eval_holdout" / "per_molecule.json").is_file()
            or (sd / "eval_holdout" / "per_reaction.json").is_file())
        cert_note = (", ".join(f"{a}={s}" for a, s in cert_status.items())
                     if cert_status else "no registry archs")
        with open(view_dir / "MERGED_ARMS.txt", "a") as f:
            f.write(f"{base}\t{run.name}\t{len(spec_dirs)} specs"
                    f"\t{n_eval} eval_holdout\tfidelity: {cert_note}\n")
        # Propagate the run-identity + SCAN-cache files the figure loaders
        # resolve against the run-dir root (the arms share one production
        # identity, so the first copy wins): without resolved_config.yaml the
        # basis label degrades to "unknown" and the SCAN reference lines never
        # draw on the merged figures. The view is wiped on every rebuild, so
        # these must be copied here rather than dropped in by hand. Later arms
        # are checked against the view's identity -- a mismatched arm would
        # make the propagated caches/labels silently wrong for it.
        arm_id = _config_identity(run / "resolved_config.yaml")
        view_id = _config_identity(view_dir / "resolved_config.yaml")
        if view_id is not None and arm_id is not None and arm_id != view_id:
            print(f"[merge] WARNING: {base} {run.name} production identity "
                  f"{arm_id} differs from the view's {view_id} -- the "
                  "propagated SCAN caches/labels may not apply to this arm")
        for src in [run / "resolved_config.yaml",
                    *sorted(run.glob("scan_pool_*.json"))]:
            dst = view_dir / src.name
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
    (view_dir / "manifest.json").write_text(json.dumps(
        {"n_specs": idx, "specs": merged_specs,
         "merged_from": [b for b, (r, _n) in report.items() if r],
         "fidelity": {
             "policy": (
                 "record layer: every registry architecture must carry a "
                 "PASS pretraining-fidelity certificate whose identity "
                 "matches its arm; an enforced=false waiver releases the "
                 "on-node gates only and is refused here"),
             "by_arm": fidelity_by_arm,
             # No status other than PASS can reach a built view, so this is a
             # recorded zero rather than an unstated assumption.
             "n_waived": 0}}, indent=1))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-root", default=str(DEFAULT_ROOT))
    p.add_argument("--out", default=None,
                   help="view dir (default: <results-root>/merged_v4_arms)")
    args = p.parse_args(argv)
    root = Path(args.results_root)
    out = Path(args.out) if args.out else root / "merged_v4_arms"
    report = build_view(root, out)
    total = 0
    for base, (run, n) in report.items():
        print(f"[merge] {base:<28} {run or '(not pulled yet)':<28} {n} specs")
        total += n
    print(f"[merge] view: {out}  ({total} specs)")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())

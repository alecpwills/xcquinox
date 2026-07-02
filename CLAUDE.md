# xcquinox -- repo conventions

## Improvement-history log (REQUIRED)

Every change that affects **methodology or results** must be recorded in
[`xcquinox/alec/HISTORY.md`](xcquinox/alec/HISTORY.md) in the same change that makes it.

This applies to: new features; physics / loss / descriptor / training-scheme changes; data,
benchmark, or reference-value changes; basis / grid / density-fitting changes; and any bug fix
that changes results. Each entry records:

- the date and commit short-hash,
- WHAT changed, and
- **Why:** the rationale.

The point is that the *reasoning* survives -- for the paper's methods/development narrative and
for future work -- not just the diff. Keep the commit-message body and the HISTORY.md entry
consistent (both should carry the "why"). Pure scaffolding / formatting / test-only commits may be
grouped into a single line or omitted.

`xcquinox/alec/HISTORY.md` is the canonical source for the paper's development history.

## Cluster (SeaWulf) sync -- REQUIRED command form

When handing the user a command to push code to the cluster, **use the real target -- never
guess a host/alias**. The destination is fixed:

- **host:** the `$swpath` env var (= `awills@login.seawulf.stonybrook.edu`, set in `~/.bashrc`).
  Write `"$swpath"` literally in the command so it resolves from the user's shell.
- **cluster repo root:** `/gpfs/projects/FernandezGroup/Alec/xcquinox` (= `$GROUP/Alec/xcquinox`,
  also reachable via the `~/xcquinox` symlink). The package tree mirrors local, so a file at
  `xcquinox/alec/<...>` locally goes to `.../Alec/xcquinox/xcquinox/alec/<...>`.

Canonical form (run from the local repo root, syncing only the changed files):

```bash
rsync -av <changed paths, repo-relative> \
      "$swpath":/gpfs/projects/FernandezGroup/Alec/xcquinox/<same dir, repo-relative>/
```

Example -- two cluster-module files:

```bash
rsync -av xcquinox/alec/cluster/grid_config.py xcquinox/alec/cluster/spec_builder.py \
      "$swpath":/gpfs/projects/FernandezGroup/Alec/xcquinox/xcquinox/alec/cluster/
```

I never run the rsync/ssh myself -- I hand the user the exact command. That is precisely why the
host and path must be correct, not guessed. See `hpcjobs/SEAWULF_RUNBOOK.md` for the full path table.

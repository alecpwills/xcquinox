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

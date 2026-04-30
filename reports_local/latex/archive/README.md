# Archive — superseded by 2026-04-30 LaTeX restructure

These files were superseded by the restructure committed on 2026-04-30 (commits
`8767604f` through `e411e5547`). They are kept here as historical reference only;
do NOT include them in any build.

## archive/old_supplements/
The four standalone supplement files that were absorbed into the master
`../supplement.tex` (39 pp) on 2026-04-30. Cross-document references to figures
and tables originally distributed across these four files now resolve via
`xr-hyper` to the master supplement.

- `step5_supplement.tex` → content now in master `supplement.tex` §S1 + §S2
- `step6_unweighted_supplement.tex` → content now in master §S3 (paired with integration)
- `step6_integration_supplement.tex` → content now in master §S3 (paired with unweighted)
- `step6_comparison_supplement.tex` → content now in master §S4

## archive/old_builds/
Old build output directories for the pre-restructure manuscript family.
Regenerable from the archived sources if ever needed; otherwise safe to delete.

- `build/` — old main.pdf (pre-restructure, 9 pp Results-section-only main)
- `build_s5/` — old step5_supplement.pdf
- `build_s6c/` — old step6_comparison_supplement.pdf
- `build_s6i/` — old step6_integration_supplement.pdf
- `build_s6u/` — old step6_unweighted_supplement.pdf

## Current pipeline (use these, not the archive)

Active files in `reports_local/latex/`:
- `main.tex` (12 pp; lean Intro / Theory / Results A,B,C / Conclusions / Limitations)
- `supplement.tex` (39 pp; master supplement absorbing all 4 old supplements)
- `references.bib` (shared bibliography)

Active build outputs:
- `build_supp/supplement.pdf` (39 pp)
- `build_main/main.pdf` (12 pp; depends on `supplement.aux` for xr-hyper cross-refs)

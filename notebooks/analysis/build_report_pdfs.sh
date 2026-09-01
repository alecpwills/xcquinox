#!/usr/bin/env bash
# Build the paper-precursor report PDFs from their markdown sources.
#
# The two-stage form (pandoc -s to .tex, then two explicit pdflatex passes) is
# deliberate: pandoc's internal pdflatex invocation silently dropped whole
# paragraphs from REPORT_pretraining_evolution.md wherever an unbreakable
# figure+caption block overfilled a page (Overfull \vbox), losing ~470 words
# with a zero exit status. The explicit passes render the identical source
# completely (verified word-for-word against a tall-page control render), and
# leave a .log that can be checked for Overfull \vbox lines, which is the
# failure signature. report_pdf_header.tex caps figure heights so float pages
# break correctly, and sets longtable to \scriptsize for the wide result
# tables.
#
# Usage: ./build_report_pdfs.sh [REPORT.md ...]
# With no arguments, builds both reports. PDFs land beside the sources.

set -uo pipefail
cd "$(dirname "$0")"

# pandoc >= 3 is required: the 2.x LaTeX template emits unclamped floats whose
# pages can overfill and clip (a figure and its caption were lost from page 24
# of the pretraining report under pandoc 2.5); the 3.x template bounds every
# image with \pandocbounded. Override with PANDOC=/path/to/pandoc.
PANDOC="${PANDOC:-}"
if [ -z "$PANDOC" ]; then
    for cand in pandoc "$HOME/anaconda3/envs/cosmopoesis/bin/pandoc"; do
        ver=$("$cand" --version 2>/dev/null | awk 'NR==1{print $2}')
        case "$ver" in
            3.*|[4-9].*) PANDOC="$cand"; break ;;
        esac
    done
fi
if [ -z "$PANDOC" ]; then
    echo "FAIL: no pandoc >= 3 found (system pandoc 2.x clips overfull float pages); set PANDOC=" >&2
    exit 1
fi

REPORTS=("$@")
if [ ${#REPORTS[@]} -eq 0 ]; then
    REPORTS=(REPORT_pretraining_evolution.md REPORT_problem_species.md)
fi

BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT

fail=0
for md in "${REPORTS[@]}"; do
    base="${md%.md}"
    "$PANDOC" "$md" -o "$BUILD/$base.tex" -s \
        --pdf-engine=pdflatex \
        -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true \
        --include-in-header=report_pdf_header.tex
    ok=1
    for pass in 1 2; do
        pdflatex -interaction=nonstopmode -output-directory="$BUILD" \
            "$BUILD/$base.tex" > "$BUILD/$base.pass$pass.out" 2>&1 || ok=0
    done
    log="$BUILD/$base.log"
    vbox=$(grep -c "Overfull \\\\vbox" "$log" || true)
    toolarge=$(grep -c "Float too large" "$log" || true)
    if [ "$ok" -ne 1 ] || [ ! -s "$BUILD/$base.pdf" ]; then
        echo "FAIL $md: pdflatex did not produce a PDF (see $BUILD/$base.pass2.out)" >&2
        fail=1
        trap - EXIT
        continue
    fi
    if [ "$vbox" != "0" ] || [ "$toolarge" != "0" ]; then
        echo "FAIL $md: $vbox Overfull-vbox / $toolarge float-too-large lines -- content may be dropped (log kept at $log)" >&2
        fail=1
        trap - EXIT
        continue
    fi
    cp "$BUILD/$base.pdf" "$base.pdf"
    pages=$(pdfinfo "$base.pdf" | awk '/^Pages:/{print $2}')
    echo "OK   $base.pdf ($pages pages, 0 overfull vboxes)"
done
exit $fail

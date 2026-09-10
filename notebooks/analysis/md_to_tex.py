"""The report's markdown-to-LaTeX translator.

Turns one markdown file (the assembled v7 report or its summary) into a ``.tex`` file beside
it and, with ``--pdf``, compiles it with xelatex. The markdown dialect is the one the report
parts use: ``# title`` (dropped; the title is set on the command line), ``## N. title`` and
``## title`` sections, ``### N.M title`` subsections, paragraphs wrapped at will, ``$...$``
inline math (which may wrap across lines), ``$$`` display blocks, ``* `` lists whose items may
carry an indented ``$$`` block, ``|`` tables with a ``|---|`` separator row, ``![alt](path)``
figures, ``**bold**`` and backtick code spans.

Headings keep the markdown's own numbers (starred LaTeX sections), so that the summary and the
full report cite the same section numbers; a numbered heading out of order is an error.

Tables: a longtable in ``small`` (up to 5 columns) or ``footnotesize``; the columns are ``l``
except the long text columns (widest cell over ``LONG`` characters), which are wrapped ``p{}``
columns sharing what the short columns leave of the line. The table goes on a landscape page
when the short columns' natural width (``CHAR[size]`` per character, ``PAD`` per column, in
fractions of the portrait line, calibrated on the rendered tables) plus ``MIN_LONG`` for each
wrapped column exceeds ``LINE`` of the portrait line, and is set in ``scriptsize`` when that
still exceeds the landscape line (``LANDSCAPE_LINE`` portrait lines). Inside ``pdflscape``'s
landscape page ``\\linewidth`` is already the landscape line, so the wrapped widths are emitted
divided by ``LANDSCAPE_LINE`` there. A short bold paragraph directly before a landscape table
is its caption and moves inside the landscape block. The separator row is measured with the
data (a three-character floor); column alignment markers are not supported; the measure counts
markdown characters (a code or math cell is measured without its delimiters).

Figures: a run of consecutive figures is emitted in markdown order; the ones at least ``WIDE``
times wider than tall are set on landscape pages at the landscape line width (their natural
height then fits one per page, or two per page from an aspect ratio of about 3.1; the
``0.9\\textheight`` cap is a guard that never binds for these aspect ratios), the others in
portrait at the line width, capped at ``0.92\\textheight``.

Usage::

    python notebooks/analysis/md_to_tex.py notebooks/analysis/REPORT_v7_2026-09-09.md \\
        --title "..." --date 2026-09-09 --pdf
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image

# one character as a fraction of the portrait line width, by font size: the widest cells of
# the rendered tables measure 0.19 cm per character at small (9 pt DejaVu Serif) and 0.16 cm
# at footnotesize on the 17.0 cm line (back-solved from the overfull widths of the first
# build), scriptsize scaled by the point size
CHAR = {"small": 0.0112, "footnotesize": 0.0100, "scriptsize": 0.0088}
PAD = 0.025            # two \tabcolsep, as a fraction of the portrait line
LONG = 28              # a column whose widest cell exceeds this is wrapped
MIN_LONG = 0.30        # the width a wrapped column is given when the line has room for it
MIN_SHARE = 0.08       # below this share a wrapped column is unreadable: the table is refused
LINE = 0.95            # the share of a line a table may fill (the rest is the margin the
                       # width model needs: the calibration is the widest cells, not the mean)
LANDSCAPE_LINE = 1.5   # the landscape line in portrait lines
WIDE = 2.0             # figures at least this wide relative to their height go on landscape pages
CAPTION_MAX = 400      # a bold paragraph longer than this before a table is prose, not a caption

_SPECIAL = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
            "~": r"\textasciitilde{}", "^": r"\^{}", "{": r"\{", "}": r"\}", "$": r"\$"}
# an inline math span opens with a $ not followed by a space and closes with a $ not preceded
# by one, so that two literal dollar amounts in a sentence are not a span
_TOKEN = re.compile(r"(`[^`]*`|\$(?!\s)[^$]*?(?<!\s)\$)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_IMAGE = re.compile(r"!\[[^\]]*\]\((.+)\)")
_COMMENT = re.compile(r"<!--.*?-->")   # a whole-line HTML comment: the table generator's markers
_SECTION = re.compile(r"## (\d+)\. ")
_SUBSECTION = re.compile(r"### (\d+)\.(\d+) ")

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.0cm]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{longtable,booktabs,pdflscape}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setmonofont{DejaVu Sans Mono}[Scale=0.85]
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\graphicspath{{./}}
"""


class MarkdownError(ValueError):
    """Malformed markdown, named by line."""


# ----------------------------------------------------------------------------- inline

def escape_text(s: str) -> str:
    """LaTeX-escape plain text."""
    return "".join(_SPECIAL.get(c, c) for c in s)


def code_span(s: str) -> str:
    """A backtick span: ``\\path`` unless it holds a space or a backslash."""
    if " " in s or "\\" in s:
        return r"\texttt{" + escape_text(s) + "}"
    return r"\path{" + s + "}"


def _inline_run(s: str) -> str:
    out = []
    for tok in _TOKEN.split(s):
        if not tok:
            continue
        if tok.startswith("`"):
            out.append(code_span(tok[1:-1]))
        elif tok.startswith("$"):
            out.append(tok)
        else:
            out.append(escape_text(tok))
    return "".join(out)


def inline(s: str) -> str:
    """A text run: bold resolved first (it may span code or math), then the run tokenized
    into code spans (``\\path`` / ``\\texttt``), inline math (kept verbatim) and escaped text."""
    out = []
    for k, piece in enumerate(_BOLD.split(s)):
        if not piece:
            continue
        out.append(r"\textbf{" + _inline_run(piece) + "}" if k % 2 else _inline_run(piece))
    return "".join(out)


# ----------------------------------------------------------------------------- tables

def _cell_measure(cell: str) -> int:
    return len(cell.replace("`", "").replace("$", ""))


def _split_row(row: str) -> List[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def table_block(rows: Sequence[str], caption: Optional[str] = None, first_line: int = 0) -> str:
    """A markdown table (header, separator, body rows) as a longtable.

    ``caption`` is already-translated LaTeX; it is placed inside the landscape block ahead of
    the table when the table is landscape, and ignored otherwise (the caller keeps it as the
    paragraph before the table). ``first_line`` names the table's first line in errors.
    """
    cells = [_split_row(r) for r in rows]
    if len(cells) < 2:
        raise MarkdownError(f"line {first_line + 1}: a table needs a header and a separator row")
    header, body = cells[0], cells[2:]
    n = len(header)
    for k, r in enumerate(cells[1:], start=1):
        if len(r) != n:
            raise MarkdownError(
                f"line {first_line + k + 1}: table row has {len(r)} cells, the header {n}")
    maxlen = [max(_cell_measure(r[i]) for r in cells) for i in range(n)]
    size = "small" if n <= 5 else "footnotesize"
    long_cols = [i for i in range(n) if maxlen[i] > LONG]
    short = [i for i in range(n) if i not in long_cols]

    def short_width(sz):
        return sum(CHAR[sz] * maxlen[i] + PAD for i in short)

    landscape = short_width(size) + MIN_LONG * len(long_cols) > LINE
    line = LANDSCAPE_LINE if landscape else 1.0
    if landscape and short_width(size) + MIN_LONG * len(long_cols) > LINE * line:
        size = "scriptsize"
    if long_cols:
        avail = LINE * line - short_width(size)
        share = max(MIN_LONG, avail / len(long_cols))
        if share * len(long_cols) > avail:
            # the floor does not fit even at scriptsize: the wrapped columns share what is
            # left, and a table that leaves them unreadable is refused rather than overfull
            share = avail / len(long_cols)
            if share < MIN_SHARE:
                raise MarkdownError(
                    f"line {first_line + 1}: table too wide, {len(long_cols)} wrapped "
                    f"columns would get {share:.2f} of the line each")
        emitted = share / line            # inside landscape, \linewidth is the landscape line
        spec = "".join(f"p{{{emitted:.3f}\\linewidth}}" if i in long_cols else "l"
                       for i in range(n))
    else:
        spec = "l" * n
    lines = []
    if landscape:
        lines.append(r"\begin{landscape}")
        if caption:
            lines += [caption, ""]
    lines.append(f"\\begin{{{size}}}")
    lines.append(f"\\begin{{longtable}}{{{spec}}}")
    lines.append(r"\toprule " + " & ".join(inline(c) for c in header) + r" \\ \midrule \endhead")
    for r in body:
        lines.append(" & ".join(inline(c) for c in r) + r" \\")
    lines.append(r"\bottomrule \end{longtable}")
    lines.append(f"\\end{{{size}}}")
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- figures

def image_aspect(path, figroot, lineno: int = 0) -> float:
    """width / height of a figure; a figure that is not on disk is named with its line."""
    full = Path(figroot) / path
    if not full.is_file():
        raise MarkdownError(f"line {lineno + 1}: figure not found: {full}")
    with Image.open(full) as im:
        w, h = im.size
    return w / h


def figure_block(paths: Sequence[str], figroot, first_line: int = 0) -> str:
    """Consecutive figures in markdown order; each run of wide ones on landscape pages."""
    out: List[str] = []
    run: List[str] = []

    def flush_wide():
        if run:
            out.append(r"\begin{landscape}")
            for f in run:
                out.append("\\begin{center}\\includegraphics[width=\\linewidth,"
                           "height=0.9\\textheight,keepaspectratio]{" + f + "}\\end{center}")
            out.append(r"\end{landscape}")
            run.clear()

    for k, f in enumerate(paths):
        if image_aspect(f, figroot, first_line + k) >= WIDE:
            run.append(f)
        else:
            flush_wide()
            out.append("\\begin{center}\\includegraphics[width=\\linewidth,"
                       "height=0.92\\textheight,keepaspectratio]{" + f + "}\\end{center}")
    flush_wide()
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- blocks

def _list_item(item_lines: Sequence[str], first_line: int) -> str:
    """One list item: its text runs joined before escaping, its ``$$`` blocks as display math."""
    pieces, text, math_buf, in_math = [], [], [], False
    for k, piece in enumerate(item_lines):
        if piece == "$$":
            if in_math:
                pieces.append("\\[\n" + "\n".join(math_buf) + "\n\\]")
                math_buf = []
            elif text:
                pieces.append(inline(" ".join(text)))
                text = []
            in_math = not in_math
        elif in_math:
            math_buf.append(piece)
        else:
            text.append(piece)
    if in_math:
        raise MarkdownError(f"line {first_line + 1}: unclosed $$ block in a list item")
    if text:
        pieces.append(inline(" ".join(text)))
    return r"\item " + " ".join(pieces)


def _heading(ln: str, state: dict, lineno: int) -> str:
    if ln.startswith("## "):
        m = _SECTION.match(ln)
        if m:
            n = int(m.group(1))
            if n <= state["section"]:
                raise MarkdownError(
                    f"line {lineno + 1}: section {n} does not follow section {state['section']}")
            state["section"], state["subsection"] = n, 0
        return f"\\section*{{{inline(ln[3:])}}}"
    m = _SUBSECTION.match(ln)
    if m:
        n, k = int(m.group(1)), int(m.group(2))
        if n != state["section"] or k <= state["subsection"]:
            raise MarkdownError(
                f"line {lineno + 1}: subsection {n}.{k} does not follow "
                f"{state['section']}.{state['subsection']}")
        state["subsection"] = k
    return f"\\subsection*{{{inline(ln[4:])}}}"


def convert(md: str, figroot) -> str:
    """The document body."""
    out: List[str] = []
    lines = md.splitlines()
    state = {"section": 0, "subsection": 0}
    i = 0
    in_list = False
    para: List[str] = []

    def flush():
        if para:
            out.append(inline(" ".join(para)))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if _COMMENT.fullmatch(ln.strip()):
            i += 1                        # a generator marker (<!-- table:NAME -->), not text
            continue
        special = ln.startswith(("* ", "# ", "## ", "### ", "|", "![")) or ln.strip() in ("", "$$")
        if not special:
            para.append(ln.strip())
            i += 1
            continue
        flush()
        if in_list and not ln.startswith("* ") and ln.strip() != "":
            out.append(r"\end{itemize}")     # a heading, table, figure or $$ ends the list
            in_list = False
        if ln.startswith("* "):
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            first = i
            item = [ln[2:].strip()]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                item.append(lines[i].strip())
                i += 1
            out.append(_list_item(item, first))
            continue
        if in_list and ln.strip() == "":
            out.append(r"\end{itemize}")
            in_list = False
        if ln.strip() == "":
            out.append("")
            i += 1
            continue
        if ln.startswith("# "):
            i += 1
            continue
        if ln.startswith("## ") or ln.startswith("### "):
            out.append(_heading(ln, state, i))
            i += 1
            continue
        if ln.strip() == "$$":
            j = i + 1
            while j < len(lines) and lines[j].strip() != "$$":
                j += 1
            if j >= len(lines):
                raise MarkdownError(f"line {i + 1}: unclosed $$ block")
            out.append("\\[\n" + "\n".join(lines[i + 1:j]) + "\n\\]")
            i = j + 1
            continue
        if ln.startswith("!["):
            figs = []
            first = i
            while i < len(lines):
                if lines[i].startswith("!["):
                    m = _IMAGE.fullmatch(lines[i].strip())
                    if not m:
                        raise MarkdownError(f"line {i + 1}: unreadable figure line {lines[i]!r}")
                    figs.append(m.group(1))
                    i += 1
                elif lines[i].strip() == "" and i + 1 < len(lines) and lines[i + 1].startswith("!["):
                    i += 1
                else:
                    break
            out.append(figure_block(figs, figroot, first))
            continue
        # a table
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        k = len(out) - 1
        while k >= 0 and out[k] == "":
            k -= 1
        caption = None
        if k >= 0 and out[k].startswith(r"\textbf{") and len(out[k]) <= CAPTION_MAX:
            caption = out[k]
        block = table_block(lines[i:j], caption, i)
        if caption and block.startswith(r"\begin{landscape}"):
            del out[k:]                   # the caption moved inside the landscape block
        out.append(block)
        i = j
    flush()
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)


#: the figure directories the v7 documents may reference: the merged family sets and the
#: subset-selection set; a per-run set of an earlier refresh fails the build
V7_FIGURE_PREFIXES = ("figures_dfs_step7_v7_family", "figures_dfs_step7_v7_subsets")
V7_DOCUMENT_PREFIXES = ("REPORT_v7", "SUMMARY_v7", "SLIDES_v7")
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)|\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def figure_paths(text: str) -> List[str]:
    """Every image path referenced by markdown ``![](path)`` or LaTeX
    ``\\includegraphics[...]{path}`` lines, in order."""
    return [a or b for a, b in _IMAGE_REF.findall(text)]


def check_figure_paths(text: str, allowed_prefixes=V7_FIGURE_PREFIXES) -> None:
    """Raise :class:`MarkdownError` naming the first image path whose first directory
    does not start with one of ``allowed_prefixes``: a v7 document may reference the
    merged family sets and the subset set only, so a stale per-run path fails the
    build instead of rendering an old figure."""
    for path in figure_paths(text):
        head = path.split("/", 1)[0]
        if not any(head.startswith(p) for p in allowed_prefixes):
            raise MarkdownError(
                f"figure path outside the allowed directories {list(allowed_prefixes)}: "
                f"{path}")


def document(body: str, title: str, date: str) -> str:
    return (PREAMBLE + "\\begin{document}\n"
            + f"\\title{{{inline(title)}}}\n\\author{{}}\\date{{{date}}}\n\\maketitle\n\n"
            + body + "\n\\end{document}\n")


def failure_excerpt(log_text: str, width: int = 1200) -> str:
    """The first error of a xelatex log, or its tail when the log carries no error marker."""
    k = log_text.find("\n! ")
    return log_text[k:k + width] if k >= 0 else log_text[-width:]


def _default_title(md: str, path: Path) -> str:
    for ln in md.splitlines():
        if ln.startswith("# "):
            return ln[2:].strip()
    return path.stem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("markdown", help="the markdown file; the .tex is written beside it")
    ap.add_argument("--title", default=None, help="default: the markdown's # heading")
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    ap.add_argument("--pdf", action="store_true", help="compile with xelatex (twice)")
    args = ap.parse_args(argv)

    src = Path(args.markdown)
    if not src.is_file():
        print(f"markdown file not found: {src}")
        return 1
    md = src.read_text()
    if src.name.startswith(V7_DOCUMENT_PREFIXES):
        # the v7 documents are built on the merged family figure sets alone
        check_figure_paths(md)
    title = args.title if args.title is not None else _default_title(md, src)
    tex = src.with_suffix(".tex")
    tex.write_text(document(convert(md, src.parent), title, args.date))
    print(f"wrote {tex}")
    if not args.pdf:
        return 0
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                           cwd=src.parent, capture_output=True, text=True)
        if r.returncode != 0:
            log = tex.with_suffix(".log")
            text = log.read_text(errors="replace") if log.exists() else r.stdout
            print("xelatex failed:\n" + failure_excerpt(text))
            return 1
    for suffix in (".aux", ".log", ".out"):
        p = tex.with_suffix(suffix)
        if p.exists():
            p.unlink()
    info = subprocess.run(["pdfinfo", str(tex.with_suffix(".pdf"))], capture_output=True, text=True)
    print(f"wrote {tex.with_suffix('.pdf')}",
          *[ln for ln in info.stdout.splitlines() if ln.startswith("Pages")])
    return 0


if __name__ == "__main__":
    sys.exit(main())

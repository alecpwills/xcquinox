"""Tests for ``notebooks/analysis/md_to_tex.py`` (the report's markdown-to-LaTeX translator).

The translator turns one markdown file -- the assembled v7 report -- into a ``.tex`` file and,
on request, compiles it with xelatex. The tests below are the T1-T5 list of the design page
``scratch/density_audit_2026-09-07/page_md_to_tex.md``: the inline tokenizer (escaping, math
kept verbatim, code spans, bold), the block converter (paragraph joining, display math inside a
list item), the table rule (the column widths and the portrait/landscape decision), the figure
grouping (the wide figures on a landscape page with their height caps) and the command-line
entry point (the ``.tex`` beside the markdown, and the PDF with ``--pdf``).

Everything is synthetic and local: the markdown, the tables and the PNGs are built in
``tmp_path``. No repository markdown, no repository figure, no network. The only external
program used is xelatex, and only in the one test that compiles; that test is skipped where
xelatex is absent.

Numeric thresholds come from the module's own constants (``WIDE``, ``LONG``, ``MIN_LONG``,
``CHAR``, ``PAD``, ``LANDSCAPE_LINE``) so that the tests state the rule rather than a
transcription of one run's output. Two numbers are pinned here instead, because the design page
states them without naming them as constants: ``mod.LINE`` (the usable portrait line) and
``_TALL_ASPECT`` (the aspect at which a landscape figure is capped at half a text height).

The script under test is loaded by path INSIDE each test, as the sibling ``test_plot_subset_jsd``
does. The lazy load is deliberate: while the script is absent every test reports its own
ImportError instead of the module collapsing into one collection error, so the RED state names
each requirement separately.

Mutation coverage (the page's list): m1 -> test_inline_keeps_math_verbatim,
test_convert_joins_wrapped_lines_into_one_paragraph; m2 ->
test_convert_joins_wrapped_lines_into_one_paragraph; m3 ->
test_table_block_two_long_columns_is_landscape_with_the_caption_inside; m4 -> that test and
test_convert_moves_a_bold_caption_into_the_landscape_block; m5 ->
test_figure_block_groups_the_wide_figures_into_one_landscape_block; m6 ->
test_main_pdf_compiles_and_removes_the_auxiliary_files.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "md_to_tex.py"
_MODNAME = "md_to_tex"

# The share of the line a table may occupy is the module's LINE (read as mod.LINE below).

# The aspect ratio from which two landscape figures fit one page at the landscape line width
# (their natural height, not a cap, decides it); the fixture straddles it.
_TALL_ASPECT = 3.1

_GRAPHICS = re.compile(r"\\includegraphics\[([^]]*)\]\{([^}]*)\}")


def _load_script():
    """Load the script under test from its path, fresh on every call."""
    if not _SCRIPT.exists():
        raise ImportError(f"{_SCRIPT} does not exist: the translator has not been written")
    spec = importlib.util.spec_from_file_location(_MODNAME, _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build a module spec for {_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODNAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(_MODNAME, None)
        raise
    return mod


# ---------------------------------------------------------------------------
# table fixtures and the page's column rule, expressed from the module constants
# ---------------------------------------------------------------------------

_LONG_A = "the recurring density tail species removed from this leg"
_LONG_B = "the cell-slice PBE anchor, not the pooled union anchor"


def _markdown_table(header, body):
    """A markdown table: the header row, the separator row, the body rows."""
    rows = ["| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|"]
    rows += ["| " + " | ".join(r) + " |" for r in body]
    return rows


def _rule(mod, header, body):
    """The page's table rule, computed from the module's constants.

    Returns ``(size, long_cols, short_width, landscape, share)``: the environment name, the
    indices of the wrapped columns, the natural width of the unwrapped columns as a fraction
    of the portrait line, the landscape verdict and the width one wrapped column receives.
    The measured length of a column is the widest cell over the whole markdown block, the
    separator row included, as the block is what the translator is handed.
    """
    cells = [[c.strip() for c in r.strip().strip("|").split("|")]
             for r in _markdown_table(header, body)]
    n = len(header)
    maxlen = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(n)]
    size = "small" if n <= 5 else "footnotesize"
    long_cols = [i for i in range(n) if maxlen[i] > mod.LONG]
    short_width = sum(mod.CHAR[size] * maxlen[i] + mod.PAD
                      for i in range(n) if i not in long_cols)
    landscape = short_width + mod.MIN_LONG * len(long_cols) > mod.LINE
    line = mod.LANDSCAPE_LINE if landscape else 1.0
    share = (max(mod.MIN_LONG, (mod.LINE * line - short_width) / len(long_cols))
             if long_cols else None)
    return size, long_cols, short_width, landscape, share


def _four_numeric():
    header = ["arch", "subset", "E", "ED"]
    body = [["medium", "12", "8.92", "11.27"], ["medium", "18", "9.41", "12.03"]]
    return header, body


def _twelve_numeric():
    header = ["G", "arch", "r", "nrxn", "E", "Epbe", "nsp", "eps", "epsp", "ED", "EDpbe", "win"]
    body = [["G1", "med", "12", "47", "8.92", "5.67", "53", "0.013", "0.008", "11.3", "12.5",
             "True"]]
    return header, body


def _six_with_one_long():
    header = ["arch", "r", "E", "D", "ED", "note"]
    body = [["medium", "12", "8.92", "0.013", "11.27", _LONG_A]]
    return header, body


def _twelve_with_two_long():
    header = ["G", "arch", "r", "nrxn", "E", "Epbe", "nsp", "eps", "ED", "win", "note",
              "species"]
    body = [["G1", "med", "12", "47", "8.92", "5.67", "53", "0.013", "11.3", "True",
             _LONG_A, _LONG_B]]
    return header, body


# ---------------------------------------------------------------------------
# T1  inline: escaping, math kept verbatim, code spans, bold
# ---------------------------------------------------------------------------

def test_inline_escapes_the_latex_specials_of_plain_text():
    mod = _load_script()
    out = mod.inline("the loss_metric is 50% of A & B, run #1, in {braces}")
    assert r"loss\_metric" in out
    assert r"50\%" in out
    assert r"A \& B" in out
    assert r"\#1" in out
    assert r"\{braces\}" in out
    assert "loss_metric" not in out          # no bare underscore survives
    assert out.count("%") == 1               # the only % is the escaped one
    assert "%" not in out.replace(r"\%", "")


def test_inline_escapes_backslash_tilde_and_caret():
    mod = _load_script()
    out = mod.inline(r"a path C:\tmp with ~home and x^2")
    assert r"\textbackslash" in out
    # the tilde is escaped (\textasciitilde{} or \~{}), never passed through bare
    assert re.search(r"(?<!\\)~", out) is None
    assert r"\^" in out


def test_inline_keeps_math_verbatim():
    """Mutation m1 (escaping applied inside math) is killed here."""
    mod = _load_script()
    out = mod.inline(r"the weight $w_t^2 = 50\%$ and the tail_index")
    assert r"$w_t^2 = 50\%$" in out          # the span is copied through untouched
    assert r"tail\_index" in out             # the surrounding text is still escaped
    assert r"\$" not in out
    assert r"w\_t" not in out


def test_inline_code_spans_use_path_or_texttt_by_the_space_rule():
    mod = _load_script()
    assert mod.inline("see `aux_log.pkl` now") == r"see \path{aux_log.pkl} now"
    assert mod.inline("the `loss_metric: absolute` flag") == \
        r"the \texttt{loss\_metric: absolute} flag"
    backslashed = mod.inline(r"the `a\b` span")
    assert backslashed.startswith(r"the \texttt{a\textbackslash")
    assert backslashed.endswith("b} span")


def test_inline_bold_becomes_textbf():
    mod = _load_script()
    assert mod.inline("**Fig. 1** the left panel") == r"\textbf{Fig. 1} the left panel"
    assert mod.inline("**medium_attn** wins") == r"\textbf{medium\_attn} wins"


# ---------------------------------------------------------------------------
# T2  convert: paragraph joining, display math in a list item, headings
# ---------------------------------------------------------------------------

def test_convert_joins_wrapped_lines_into_one_paragraph(tmp_path):
    """Mutations m1 (escaping inside math) and m2 (line-by-line escaping) are killed here.

    The report wraps its prose, so an inline span opens on one line and closes on the next.
    Escaping the lines separately turns the unterminated ``$`` into ``\\$``, the backslashes
    of the span into ``\\textbackslash{}`` and the braces of its exponent into ``\\{``.
    """
    mod = _load_script()
    md = ("For each descriptor $x \\in\n"
          "\\{\\rho^{1/3}, s, \\alpha\\}$ the reference sample is binned.\n")
    body = mod.convert(md, tmp_path)
    assert r"$x \in \{\rho^{1/3}, s, \alpha\}$" in body
    assert r"\$" not in body
    assert "^{1/3}" in body                  # the braces of the exponent are not escaped
    assert r"\textbackslash" not in body


def test_convert_display_math_inside_a_list_item_and_the_itemize_closes(tmp_path):
    mod = _load_script()
    md = ("* Reaction energies, the tail-weighted sum:\n"
          "  $$\n"
          "  \\mathcal{L}_{\\rm rxn} = \\frac{1}{|T|}\\sum_t w_t^2 r_t^2,\n"
          "  $$\n"
          "  the DFS convergence-tail scoring.\n"
          "* Atom anchors, the residue of the AE channel.\n"
          "\n"
          "The optimizer is AdamW.\n")
    body = mod.convert(md, tmp_path)
    assert r"\begin{itemize}" in body
    assert body.count(r"\item") == 2
    assert r"\mathcal{L}_{\rm rxn} = \frac{1}{|T|}\sum_t w_t^2 r_t^2," in body
    assert r"\$" not in body
    # the display math is inside the item, not a paragraph of its own
    i_begin = body.index(r"\begin{itemize}")
    i_math = body.index(r"\[")
    i_end = body.index(r"\end{itemize}")
    i_tail = body.index("The optimizer is AdamW.")
    assert i_begin < i_math < i_end < i_tail, "the list must close at the blank line"
    assert r"\]" in body


def test_convert_headings_numbered_unnumbered_and_subsection(tmp_path):
    mod = _load_script()
    md = ("# The document title\n"
          "\n"
          "## Scope and inventory\n"
          "\n"
          "## 3. Pre-training (functional cloning)\n"
          "\n"
          "### 3.1 Set, targets and objective\n")
    body = mod.convert(md, tmp_path)
    # the markdown's own numbers are kept (starred sections), so that the summary and the
    # full report cite the same section numbers
    assert r"\section*{Scope and inventory}" in body
    assert r"\section*{3. Pre-training (functional cloning)}" in body
    assert r"\subsection*{3.1 Set, targets and objective}" in body
    assert "The document title" not in body, "the H1 is dropped; the title comes from the CLI"


def test_convert_refuses_headings_out_of_order(tmp_path):
    mod = _load_script()
    with pytest.raises(ValueError, match="line 3"):
        mod.convert("## 3. A\n\n## 2. B\n", tmp_path)
    with pytest.raises(ValueError, match="line 5"):
        mod.convert("## 3. A\n\n### 3.2 x\n\n### 3.1 y\n", tmp_path)
    with pytest.raises(ValueError, match="line 1"):
        mod.convert("### 4.1 y\n", tmp_path)
    # a skipped number is allowed (the summary omits subsections of the full report)
    body = mod.convert("## 3. A\n\n### 3.1 x\n\n### 3.3 y\n\n## 5. B\n", tmp_path)
    assert r"\subsection*{3.3 y}" in body and r"\section*{5. B}" in body


def test_convert_joins_the_lines_of_a_list_item_before_escaping(tmp_path):
    """The report's list items wrap too: a span opened on one continuation line and closed
    on the next must stay one span (the first build split them)."""
    mod = _load_script()
    md = ("* the tail, at $N = 3$ all three, $w_t^2 = 0,\n"
          "  1/16, 1$, absolute (`loss_metric:\n"
          "  absolute`).\n")
    body = mod.convert(md, tmp_path)
    assert r"$w_t^2 = 0, 1/16, 1$" in body
    assert r"\texttt{loss\_metric: absolute}" in body
    assert r"\$" not in body and "`" not in body


def test_inline_bold_spanning_a_code_or_math_span():
    mod = _load_script()
    out = mod.inline("**Iso-orbital indicator (`metagga`)**, SCAN's $\\alpha$")
    assert out.startswith(r"\textbf{Iso-orbital indicator (\path{metagga})}")
    assert "**" not in out
    assert mod.inline("**a $x_1$ b**") == r"\textbf{a $x_1$ b}"


def test_convert_refuses_malformed_input(tmp_path):
    mod = _load_script()
    with pytest.raises(ValueError, match="line 2"):
        mod.convert("text\n$$\nx = 1\n", tmp_path)
    with pytest.raises(ValueError, match="line 3"):
        mod.convert("| a | b |\n|---|---|\n| 1 | 2 | 3 |\n", tmp_path)
    with pytest.raises(ValueError, match="line 1"):
        mod.convert("![broken\n", tmp_path)


def test_convert_keeps_a_long_bold_paragraph_before_a_landscape_table_as_prose(tmp_path):
    """Only a short bold paragraph is a caption; the reading paragraphs that open in bold
    stay where they are."""
    mod = _load_script()
    header, body = _twelve_with_two_long()
    prose = "**Energy.** " + "On the WTMAD-2 the medium architecture is below PBE. " * 12
    assert len(prose) > mod.CAPTION_MAX
    md = prose + "\n\n" + "\n".join(_markdown_table(header, body)) + "\n"
    out = mod.convert(md, tmp_path)
    assert out.index(r"\textbf{Energy.}") < out.index(r"\begin{landscape}")


# ---------------------------------------------------------------------------
# T3  table_block: the sizes, the wrapped columns and the landscape decision
# ---------------------------------------------------------------------------

def test_table_block_four_numeric_columns_is_portrait_small():
    mod = _load_script()
    header, body = _four_numeric()
    size, long_cols, short_width, landscape, _ = _rule(mod, header, body)
    assert (size, long_cols, landscape) == ("small", [], False), \
        f"fixture must be portrait small by the page's rule (short width {short_width:.3f})"
    block = mod.table_block(_markdown_table(header, body))
    assert r"\begin{small}" in block
    assert r"\begin{longtable}{llll}" in block
    assert r"\begin{landscape}" not in block
    assert "p{" not in block
    assert r"\toprule" in block and r"\midrule" in block and r"\bottomrule" in block
    assert r"\endhead" in block, "the header must repeat on every page of the longtable"
    assert "8.92 & 11.27" in block


def test_table_block_twelve_numeric_columns_is_portrait_footnotesize():
    mod = _load_script()
    header, body = _twelve_numeric()
    size, long_cols, short_width, landscape, _ = _rule(mod, header, body)
    assert (size, long_cols, landscape) == ("footnotesize", [], False), \
        f"fixture must be portrait footnotesize (short width {short_width:.3f})"
    block = mod.table_block(_markdown_table(header, body))
    assert r"\begin{footnotesize}" in block
    assert r"\begin{longtable}{" + "l" * 12 + "}" in block
    assert r"\begin{landscape}" not in block
    assert "p{" not in block


def test_table_block_one_long_column_is_portrait_with_one_wrapped_column():
    mod = _load_script()
    header, body = _six_with_one_long()
    size, long_cols, short_width, landscape, share = _rule(mod, header, body)
    assert len(_LONG_A) > mod.LONG, "the note column must exceed the wrapping threshold"
    assert (size, long_cols, landscape) == ("footnotesize", [5], False), \
        f"fixture must be portrait with one wrapped column (short width {short_width:.3f})"
    assert share > mod.MIN_LONG, "the short columns must leave more than the floor"
    block = mod.table_block(_markdown_table(header, body))
    assert r"\begin{footnotesize}" in block
    assert r"\begin{landscape}" not in block
    assert block.count("p{") == 1
    assert "p{0." in block
    expected = f"\\begin{{longtable}}{{lllll" + f"p{{{share:.3f}\\linewidth}}" + "}"
    assert expected in block, f"expected the wrapped column at {share:.3f} of the line"


def test_table_block_two_long_columns_is_landscape_with_the_caption_inside():
    """Mutations m3 (the landscape rule ignoring the long columns' share) and m4 (the caption
    emitted before the landscape block) are killed here.

    Under m3 the decision is taken on the short columns alone, ``short_width`` is below the
    line and this table stays portrait.
    """
    mod = _load_script()
    header, body = _twelve_with_two_long()
    size, long_cols, short_width, landscape, share = _rule(mod, header, body)
    assert (size, long_cols, landscape) == ("footnotesize", [10, 11], True), \
        f"fixture must be landscape (short width {short_width:.3f})"
    assert short_width < mod.LINE, \
        "the short columns alone must fit the portrait line, so only the wrapped columns' " \
        "share can force the landscape page"
    assert short_width + mod.MIN_LONG * len(long_cols) <= mod.LINE * mod.LANDSCAPE_LINE, \
        "the fixture must not reach the size demotion the page leaves unstated"
    caption = r"\textbf{Combined leg} (both pools)."
    block = mod.table_block(_markdown_table(header, body), caption)
    assert block.count(r"\begin{landscape}") == 1
    assert block.count(r"\end{landscape}") == 1
    assert r"\begin{footnotesize}" in block
    assert block.count("p{") == 2
    assert "p{0." in block
    assert share >= mod.MIN_LONG
    # inside pdflscape's landscape page \linewidth is already the landscape line, so the
    # share (a portrait-line fraction) is emitted divided by LANDSCAPE_LINE
    emitted = f"p{{{share / mod.LANDSCAPE_LINE:.3f}\\linewidth}}"
    assert block.count(emitted) == 2, f"expected the wrapped columns at {emitted}"
    assert f"p{{{share:.3f}\\linewidth}}" not in block or abs(share - share / mod.LANDSCAPE_LINE) < 5e-4
    i_ls = block.index(r"\begin{landscape}")
    i_cap = block.index(caption)
    i_tab = block.index(r"\begin{longtable}")
    i_end = block.index(r"\end{landscape}")
    assert i_ls < i_cap < i_tab < i_end, \
        "the caption belongs inside the landscape block, ahead of its table"
    assert block.count(caption) == 1


def test_table_block_portrait_table_does_not_carry_the_caption():
    """A caption handed to a portrait table stays the caller's paragraph: the block itself
    must not repeat it, and must not open a landscape page to hold it."""
    mod = _load_script()
    header, body = _six_with_one_long()
    _, _, _, landscape, _ = _rule(mod, header, body)
    assert landscape is False
    caption = r"\textbf{Table 4.2} the in-sample density fit per cell."
    block = mod.table_block(_markdown_table(header, body), caption)
    assert r"\begin{landscape}" not in block
    assert caption not in block


def test_convert_leaves_a_bold_caption_before_a_portrait_table_as_a_paragraph(tmp_path):
    mod = _load_script()
    header, body = _six_with_one_long()
    assert _rule(mod, header, body)[3] is False
    md = ("**Table 4.2** the in-sample density fit per cell.\n"
          "\n" + "\n".join(_markdown_table(header, body)) + "\n")
    out = mod.convert(md, tmp_path)
    assert r"\begin{landscape}" not in out
    assert out.count(r"\textbf{Table 4.2}") == 1
    assert out.index(r"\textbf{Table 4.2}") < out.index(r"\begin{longtable}")


def test_convert_moves_a_bold_caption_into_the_landscape_block(tmp_path):
    """Mutation m4 is killed here: emitting the caption before the block leaves it outside."""
    mod = _load_script()
    header, body = _twelve_with_two_long()
    assert _rule(mod, header, body)[3] is True
    md = ("**Combined leg** (both pools).\n"
          "\n" + "\n".join(_markdown_table(header, body)) + "\n")
    out = mod.convert(md, tmp_path)
    assert out.count(r"\begin{landscape}") == 1
    assert out.count(r"\textbf{Combined leg}") == 1, \
        "the caption is moved, not copied: one occurrence only"
    assert out.index(r"\begin{landscape}") < out.index(r"\textbf{Combined leg}") \
        < out.index(r"\begin{longtable}")


# ---------------------------------------------------------------------------
# T4  figure_block: the grouping and the height caps
# ---------------------------------------------------------------------------

def _png(path, width, height):
    Image.new("RGB", (int(width), int(height)), (255, 255, 255)).save(path)
    return path.name


def test_figure_block_groups_the_wide_figures_into_one_landscape_block(tmp_path):
    """Mutation m5 (``WIDE`` raised to 3.0) is killed here: the 2.5 figure would go portrait."""
    mod = _load_script()
    narrow = _png(tmp_path / "narrow.png", 700, 500)      # aspect 1.4
    wide = _png(tmp_path / "wide.png", 1000, 400)         # aspect 2.5
    tall = _png(tmp_path / "tall.png", 1200, 300)         # aspect 4.0
    assert 1.4 < mod.WIDE <= 2.5 < _TALL_ASPECT <= 4.0, \
        "the fixture aspects must straddle the module's WIDE and the 3.1 cap threshold"

    block = mod.figure_block([narrow, wide, tall], tmp_path)
    opts = {m.group(2): m.group(1) for m in _GRAPHICS.finditer(block)}
    assert set(opts) == {narrow, wide, tall}, "every figure of the group is emitted once"

    assert block.count(r"\begin{landscape}") == 1
    assert block.count(r"\end{landscape}") == 1
    i_ls, i_le = block.index(r"\begin{landscape}"), block.index(r"\end{landscape}")
    assert block.index(narrow) < i_ls, "the portrait figure stays out of the landscape block"
    assert i_ls < block.index(wide) < i_le
    assert i_ls < block.index(tall) < i_le

    assert r"width=\linewidth" in opts[narrow]
    assert "keepaspectratio" in opts[narrow]
    # one guard cap for every wide figure: at the landscape line width the natural height
    # of these aspect ratios already fits (one per page, two from about 3.1)
    assert r"height=0.9\textheight" in opts[wide]
    assert r"height=0.9\textheight" in opts[tall]
    assert "keepaspectratio" in opts[wide] and "keepaspectratio" in opts[tall]


def test_figure_block_keeps_the_markdown_order_of_a_mixed_group(tmp_path):
    mod = _load_script()
    wide1 = _png(tmp_path / "w1.png", 1000, 400)
    narrow = _png(tmp_path / "n.png", 700, 500)
    wide2 = _png(tmp_path / "w2.png", 1000, 400)
    block = mod.figure_block([wide1, narrow, wide2], tmp_path)
    assert block.count(r"\begin{landscape}") == 2, "two runs of wide figures, in order"
    assert block.index(wide1) < block.index(narrow) < block.index(wide2)


def test_figure_block_all_narrow_opens_no_landscape_page(tmp_path):
    mod = _load_script()
    a = _png(tmp_path / "a.png", 1650, 1140)              # aspect 1.45, the Fx/Fc figures
    b = _png(tmp_path / "b.png", 700, 500)
    block = mod.figure_block([a, b], tmp_path)
    assert r"landscape" not in block
    opts = {m.group(2): m.group(1) for m in _GRAPHICS.finditer(block)}
    assert set(opts) == {a, b}
    for f in (a, b):
        assert r"height=0.92\textheight" in opts[f], "a portrait figure is capped at the page"


def test_figure_block_names_a_missing_figure_by_line(tmp_path):
    mod = _load_script()
    a = _png(tmp_path / "a.png", 700, 500)
    with pytest.raises(ValueError, match="line 4"):
        mod.convert("text\n\n" + f"![]({a})\n" + "![](absent.png)\n", tmp_path)


# ---------------------------------------------------------------------------
# the table fit guard and its calibration
# ---------------------------------------------------------------------------

def _certificate_like():
    """The shape that overflowed the first build: ten columns, four of them long text."""
    header = ["Group", "Architecture", "Parent", "Steps", "Pointwise loss X / C",
              "max atom (mHa)", "mean (kcal/mol)", "max (kcal/mol)", "Species over 1 kcal/mol",
              "Verdict"]
    body = [["G2", "deep_cusp_mgga_3x16", "SCAN", "20000", "3.70e-06 / 5.18e-06", "3.132",
             "2.062", "7.257",
             "AlCl3, C2H2, C3H8, C4H6, CH2, CH4, CO, CO2, F2, H2O, HCN, O2, PH3, SiCH6",
             "FAIL"],
            ["G1", "shallow", "PBE", "20000", "5.41e-08 / 3.29e-06", "0.369", "0.343", "1.421",
             "C3H8, C4H6 and the two species named above in the second certificate round",
             "PASS"]] + [["G1", "medium", "PBE", "20000", "6.40e-08 / 7.58e-07", "0.210",
                          "0.162", "0.519", "none " * 8, "PASS"]] * 2
    return header, body


def _emitted_widths(block):
    return [float(w) for w in re.findall(r"p\{([0-9.]+)\\linewidth\}", block)]


def test_table_block_wrapped_columns_never_exceed_the_line():
    """Mutation: MIN_LONG raised to 0.45 (the floor binding without a fit check) must fail."""
    mod = _load_script()
    header, body = _certificate_like()
    rows = _markdown_table(header, body)
    block = mod.table_block(rows)
    size = re.search(r"\\begin\{(small|footnotesize|scriptsize)\}", block).group(1)
    landscape = r"\begin{landscape}" in block
    line = mod.LANDSCAPE_LINE if landscape else 1.0
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    n = len(header)
    maxlen = [max(len(r[i]) for r in cells) for i in range(n)]
    short = sum(mod.CHAR[size] * maxlen[i] + mod.PAD for i in range(n) if maxlen[i] <= mod.LONG)
    widths = _emitted_widths(block)
    assert widths, "the long text columns are wrapped"
    assert short / line + sum(widths) <= mod.LINE + 1e-9, \
        "the short columns and the wrapped columns together must fit the page's line"
    assert min(widths) * line >= mod.MIN_SHARE


def test_table_block_landscape_threshold_is_the_stated_minimum_share():
    """Pinned against the constants' values, not derived from them: with MIN_LONG = 0.30 and
    LINE = 0.95, five short columns of width 0.125 each (0.625) plus one wrapped column stay
    portrait (0.925), six (0.75) go landscape (1.05). A larger minimum share (0.45) would send
    the five-column table to a landscape page."""
    mod = _load_script()
    assert (mod.MIN_LONG, mod.LINE, mod.PAD, mod.CHAR["footnotesize"]) == (0.30, 0.95, 0.025, 0.0100)
    cell = "x" * 10                                   # 10 chars: 0.100 + PAD = 0.125 per column
    header5 = [cell] * 5 + ["note"]
    body5 = [[cell] * 5 + [_LONG_A]]
    assert r"\begin{landscape}" not in mod.table_block(_markdown_table(header5, body5))
    header6 = [cell] * 6 + ["note"]
    body6 = [[cell] * 6 + [_LONG_A]]
    assert r"\begin{landscape}" in mod.table_block(_markdown_table(header6, body6))


def test_table_block_refuses_a_table_no_line_can_hold():
    mod = _load_script()
    header = [f"c{i}" for i in range(22)] + ["note"]
    body = [["x" * 12] * 22 + ["a long text cell " * 5]]
    with pytest.raises(ValueError, match="too wide"):
        mod.table_block(_markdown_table(header, body))


@pytest.mark.skipif(shutil.which("xelatex") is None, reason="xelatex is not installed")
def test_compiled_tables_have_no_overfull_boxes(tmp_path):
    """The width model is calibrated on rendered DejaVu text: the tables that overflowed the
    first build (wrapped columns beside wide short columns) must compile without an
    overfull alignment. Mutation: CHAR scaled down by 20 percent must fail."""
    mod = _load_script()
    tables = [_certificate_like(), _six_with_one_long(), _twelve_with_two_long()]
    pool_header = ["Index", "Point", "Kind", "Species (name, charge, spin)"]
    pool_body = [["21", "OH+N2_to_H+N2O", "reaction energy (BH76)",
                  "HO (0, 1), N2 (0, 0), H (0, 1), N2O (0, 0)"],
                 ["24", "Li_IP", "ionization potential (IP13)", "Li (0, 1), Li+ (1, 0)"]]
    tables.append((pool_header, pool_body))
    md = "# t\n\n" + "\n\n".join("\n".join(_markdown_table(h, b)) for h, b in tables) + "\n"
    src = tmp_path / "tables.md"
    src.write_text(md)
    tex = src.with_suffix(".tex")
    tex.write_text(mod.document(mod.convert(md, tmp_path), "t", "d"))
    import subprocess
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    log = tex.with_suffix(".log").read_text(errors="replace")
    over = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\) in alignment",
                                         log)]
    assert not [x for x in over if x > 2.0], f"overfull alignments (pt): {over}"


# ---------------------------------------------------------------------------
# blocks after a list, literal dollars, the title default, the failure excerpt
# ---------------------------------------------------------------------------

def test_convert_closes_the_list_before_a_heading_or_a_table(tmp_path):
    mod = _load_script()
    md = "* an item\n## 2. Next\n\n* another\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    body = mod.convert(md, tmp_path)
    assert body.count(r"\end{itemize}") == 2
    assert body.index(r"\end{itemize}") < body.index(r"\section*{2. Next}")
    assert body.rindex(r"\end{itemize}") < body.index(r"\begin{longtable}")


def test_inline_two_literal_dollars_are_not_a_math_span():
    mod = _load_script()
    out = mod.inline("costs $5 and $7 per run, but $x_1 = 2$ is math")
    assert r"\$5 and \$7" in out
    assert "$x_1 = 2$" in out


def test_default_title_is_the_h1_or_the_file_stem(tmp_path):
    mod = _load_script()
    assert mod._default_title("# The title\n\ntext\n", tmp_path / "doc.md") == "The title"
    assert mod._default_title("text only\n", tmp_path / "doc.md") == "doc"


def test_failure_excerpt_falls_back_to_the_log_tail():
    mod = _load_script()
    with_marker = "preamble\n! Undefined control sequence.\nl.12 \\foo\n" + "x" * 50
    assert mod.failure_excerpt(with_marker).startswith("\n! Undefined control sequence.")
    without = "line " * 500
    assert mod.failure_excerpt(without, width=40) == without[-40:]


def test_main_reports_a_missing_markdown_file(tmp_path, capsys):
    mod = _load_script()
    assert mod.main([str(tmp_path / "absent.md")]) == 1
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# T5  main: the .tex beside the markdown, and the PDF with --pdf
# ---------------------------------------------------------------------------

_TITLE = "v7 revision report"
_DATE = "2026-09-09"


def _document(tmp_path):
    """A markdown file exercising a heading, a paragraph, a table and a figure."""
    _png(tmp_path / "fig.png", 700, 500)
    md = tmp_path / "doc.md"
    md.write_text(
        "# DROPPED HEADING\n"
        "\n"
        "## 1. Section\n"
        "\n"
        "A paragraph with inline math $x_0 = 1$ and the code span `aux_log.pkl`.\n"
        "\n"
        "| arch | subset | E |\n"
        "|---|---|---|\n"
        "| medium | 12 | 8.92 |\n"
        "\n"
        "![](fig.png)\n"
    )
    return md


def test_main_writes_the_tex_beside_the_markdown_without_compiling(tmp_path):
    mod = _load_script()
    md = _document(tmp_path)
    rc = mod.main([str(md), "--title", _TITLE, "--date", _DATE])
    assert not rc, "a successful run returns 0 (or None)"

    tex = md.with_suffix(".tex")
    assert tex.exists(), "the .tex is written beside the markdown"
    text = tex.read_text()
    assert r"\documentclass" in text and r"\end{document}" in text
    assert f"\\title{{{_TITLE}}}" in text
    assert f"\\date{{{_DATE}}}" in text
    assert r"\section*{1. Section}" in text
    assert "DROPPED HEADING" not in text
    assert r"\usepackage{url}" in text, r"\path{} is defined by url"
    assert r"$x_0 = 1$" in text
    assert r"\path{aux_log.pkl}" in text
    assert r"\begin{longtable}" in text
    assert r"\includegraphics" in text and "{fig.png}" in text
    for package in ("amsmath", "graphicx", "longtable", "booktabs", "pdflscape", "hyperref",
                    "fontspec"):
        assert package in text, f"the preamble must load {package}"

    assert not md.with_suffix(".pdf").exists(), "nothing is compiled without --pdf"
    assert not md.with_suffix(".aux").exists()
    assert not md.with_suffix(".log").exists()


@pytest.mark.skipif(shutil.which("xelatex") is None, reason="xelatex is not installed")
def test_main_pdf_compiles_and_removes_the_auxiliary_files(tmp_path):
    """Mutation m6 (``--pdf`` ignored) is killed here."""
    mod = _load_script()
    md = _document(tmp_path)
    rc = mod.main([str(md), "--title", _TITLE, "--date", _DATE, "--pdf"])
    assert not rc, "the compilation must succeed and return 0 (or None)"

    pdf = md.with_suffix(".pdf")
    assert pdf.exists() and pdf.stat().st_size > 0
    assert pdf.read_bytes()[:5] == b"%PDF-"
    for suffix in (".aux", ".log", ".out"):
        assert not md.with_suffix(suffix).exists(), f"the {suffix} file must be removed"


# ---------------------------------------------------------------------------
# T4  check_figure_paths: the v7 documents reference the family sets only
#
# The merged family sets replace the per-run sets, so a figure path left pointing at a removed
# per-run directory must fail the build rather than reach a reader. The guard is the one
# described in the design page ``scratch/density_audit_2026-09-07/page_documents_merged_set.md``;
# its mutation m4 (the guard accepts every path) is killed by
# test_check_figure_paths_names_the_first_stale_per_run_path and
# test_main_refuses_a_stale_per_run_figure_path_in_a_v7_document.
# ---------------------------------------------------------------------------

_V7_PREFIXES = ("figures_dfs_step7_v7_family", "figures_dfs_step7_v7_subsets")

# a path of the per-run set the family figures replace
_STALE_PATH = "figures_dfs_step7_dfs6311_grid3_v7g1_size_val_best/holdout_overview.png"
_STALE_PATH_2 = ("figures_dfs_step7_dfs6311_grid3_v7g2a_families_core_val_best/"
                 "holdout_ed_combined.png")

_FAMILY_FIGURES_MD = (
    "## 5. Held out\n"
    "\n"
    "![](figures_dfs_step7_v7_family_val_best/x.png)\n"
    "\n"
    "![](figures_dfs_step7_v7_family_pretrain/uncertified_mgga/y.png)\n"
    "\n"
    "![](figures_dfs_step7_v7_subsets/z.png)\n"
)


def test_check_figure_paths_accepts_the_family_and_the_subsets_directories():
    """The three shapes the rebuilt documents use: a family set, a family subdirectory, the
    subsets set. The prefixes are directory prefixes, so the ``_val_best`` and ``_pretrain``
    siblings of ``figures_dfs_step7_v7_family`` are inside the allowed set."""
    mod = _load_script()
    mod.check_figure_paths(_FAMILY_FIGURES_MD, _V7_PREFIXES)


def test_check_figure_paths_names_the_first_stale_per_run_path():
    """Two stale paths after one good one: the error names the first of them."""
    mod = _load_script()
    md = ("![](figures_dfs_step7_v7_family_val_best/x.png)\n"
          "\n"
          f"![]({_STALE_PATH})\n"
          "\n"
          f"![]({_STALE_PATH_2})\n")
    with pytest.raises(mod.MarkdownError, match=re.escape(_STALE_PATH)):
        mod.check_figure_paths(md, _V7_PREFIXES)


def _v7_like_document(tmp_path, name):
    """A minimal document carrying one stale per-run figure path, the figure on disk.

    The PNG exists so that the translation succeeds today: the only thing that can refuse this
    document is the new guard, never ``image_aspect``'s missing-figure error.
    """
    figdir = tmp_path / "figures_dfs_step7_dfs6311_grid3_v7g1_size_val_best"
    figdir.mkdir(exist_ok=True)
    _png(figdir / "holdout_overview.png", 700, 500)
    md = tmp_path / name
    md.write_text(
        "# a v7 document\n"
        "\n"
        "## 1. Held out\n"
        "\n"
        "A paragraph before the figure.\n"
        "\n"
        f"![]({_STALE_PATH})\n"
    )
    return md


def test_main_refuses_a_stale_per_run_figure_path_in_a_v7_document(tmp_path):
    """A REPORT_v7 document is checked against the family prefixes and is refused by name."""
    mod = _load_script()
    md = _v7_like_document(tmp_path, "REPORT_v7_test.md")
    with pytest.raises(mod.MarkdownError, match=re.escape(_STALE_PATH)):
        mod.main([str(md), "--title", _TITLE, "--date", _DATE])
    assert not md.with_suffix(".tex").exists(), "a refused document leaves no .tex behind"


def test_main_translates_a_non_v7_document_carrying_the_same_path(tmp_path):
    """The guard is scoped to the v7 documents: the same body under another name translates."""
    mod = _load_script()
    md = _v7_like_document(tmp_path, "NOTES.md")
    rc = mod.main([str(md), "--title", _TITLE, "--date", _DATE])
    assert not rc, "a document outside the v7 set is translated as before"
    tex = md.with_suffix(".tex")
    assert tex.exists()
    assert _STALE_PATH in tex.read_text(), "its figure reference is translated unchanged"


def test_the_tracked_v7_sources_reference_family_figures_only():
    """The report, the summary and the slide frames in the tree pass the guard:
    every figure they reference sits in a family directory or the subset set,
    and the guard reads LaTeX includegraphics lines as well as markdown."""
    import os
    mod = _load_script()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("REPORT_v7_2026-09-09.md", "SUMMARY_v7_2026-09-09.md",
                 "SLIDES_v7_2026-09-09_frames.tex"):
        with open(os.path.join(here, name)) as fh:
            text = fh.read()
        paths = mod.figure_paths(text)
        assert paths, name
        mod.check_figure_paths(text)
    assert mod.figure_paths("\\includegraphics[width=\\linewidth]{figures_dfs_step7_v7_family/a.png}") == \
        ["figures_dfs_step7_v7_family/a.png"]


def test_convert_drops_the_html_comment_lines_that_mark_generated_tables(tmp_path):
    """The generated per-cell tables sit between ``<!-- table:NAME -->`` marker lines; the
    markers are for the generator and must not reach the PDF, while the table between them
    does."""
    mod = _load_script()
    md = ("Before.\n\n"
          "<!-- table:holdout_combined -->\n"
          "| Architecture | Subset |\n"
          "|---|---|\n"
          "| deep_3x16 | 1 |\n"
          "<!-- /table:holdout_combined -->\n\n"
          "After.\n")
    body = mod.convert(md, tmp_path)
    assert "<!--" not in body and "table:holdout" not in body
    assert r"\begin{longtable}" in body
    assert "Before." in body and "After." in body

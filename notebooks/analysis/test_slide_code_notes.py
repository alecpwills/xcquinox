"""Tests for ``notebooks/analysis/slide_code_notes.py`` (the code notes of the annotated deck).

The module under test carries a manifest of 88 (page, order, title, file, line range, tokens)
rows, cuts the named line ranges out of the repository sources into
``notebooks/analysis/slides_code/<slug>.txt``, and emits the ``\\codenote{...}{...}`` lines that
the deck's frames file carries after each slide's ``\\slidenote{...}``. The plain deck defines
``\\codenote`` as empty; the annotated deck expands it into a verbatim frame.

The tests below are the 13-test list of the design page ``scratch/slide_code_notes_page.md``:
the manifest's ranges and tokens against the current sources (1-3), the slug rule (4), the
refusals of ``excerpt_lines`` (5-6), the write path and its atomicity and stale removal (7-8),
the drift detector on the tracked excerpt files (9), the placement of the ``\\codenote`` lines in
the frames file (10), the two wrapper definitions (11), the LaTeX escaping (12) and the header
format (13).

The module is loaded by path inside each test, as the sibling ``test_md_to_tex`` does, so that
while it is absent every requirement reports its own ImportError instead of the file collapsing
into one collection error. Tests 9, 10 and 11 assert on the deck and excerpt files before the
module is touched: their first failure is the missing artefact, not the missing module.

Mutation coverage (the page's list): M1 -> test_excerpt_lines_refuses_missing_token; M2 ->
test_excerpt_lines_refuses_bad_range; M3 -> test_write_excerpts_is_verbatim_and_atomic; M4 ->
test_manifest_tokens_present and test_written_excerpts_match_sources; M5 ->
test_written_excerpts_match_sources; M6 -> test_frames_carry_codenotes; M7 ->
test_codenote_lines_escape_latex; M8 -> test_wrappers_define_codenote.

Standard library and pytest only; no repository package is imported, and no test writes into the
repository tree (the synthetic sources and outputs live under ``tmp_path``).
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SCRIPT = _HERE / "slide_code_notes.py"
_MODNAME = "slide_code_notes"

_EXCERPT_DIR = _HERE / "slides_code"
_FRAMES = _HERE / "SLIDES_v7_2026-09-09_frames.tex"
_PLAIN = _HERE / "SLIDES_v7_2026-09-09.tex"
_ANNOTATED = _HERE / "SLIDES_v7_2026-09-09_annotated.tex"

# The pages the manifest covers and the number of rows on each, from the manifest itself
# (88 rows: 3(7) 4(5) 5(6) 6(5) 7(6) 8(6) 9(4) 11(10) 12(5) 15(12) 16(11) 19(11)).
_PAGES = (3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 16, 19)
_ROWS_PER_PAGE = {3: 7, 4: 5, 5: 6, 6: 5, 7: 6, 8: 6, 9: 4, 11: 10, 12: 5, 15: 12, 16: 11, 19: 11}
_ROWS = 88

# ``# <path>:<first>-<last>`` -- the first line of every excerpt file. No commit hash: the
# excerpts are written before the commit that adds them, so a hash could only name the parent;
# the drift test below is the provenance.
_HEADER = re.compile(r"^# (?P<path>[^\s:]+):(?P<first>\d+)-(?P<last>\d+)$")

# The frames file holds this many ``\begin{frame}`` lines; page N of the plain deck is the N-th.
_FRAME_COUNT = 24

# One ``\codenote`` line of the frames file.
_CODENOTE = re.compile(r"^\\codenote\{(.*)\}\{slides_code/([^}]+)\.txt\}$")

_SLUG = re.compile(r"^p\d+_\d+_[a-z0-9_]+$")

# The slug's title part is capped; the cap is asserted in the weaker of the two readings of the
# page ("at most 40 characters of slug"), which both a whole-slug and a title-part cap satisfy.
_SLUG_TITLE_MAX = 40


def _load_script():
    """Load the module under test from its path, fresh on every call."""
    if not _SCRIPT.exists():
        raise ImportError(f"{_SCRIPT} does not exist: the module has not been written")
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


def _mod():
    """The module under test."""
    return _load_script()


def _entry(mod, page, order, title, path, first, last, tokens=()):
    """A synthetic manifest entry."""
    return mod.Entry(
        page=page, order=order, title=title, path=path, first=first, last=last,
        tokens=tuple(tokens),
    )


def _call_with_entries(mod, monkeypatch, func, entries, *args, **kwargs):
    """Call ``func`` over ``entries`` instead of the module's own manifest.

    The page's signatures take no entry list, so the substitution goes through the module
    attribute; a signature that does take one is used directly.
    """
    if "entries" in inspect.signature(func).parameters:
        return func(*args, entries=tuple(entries), **kwargs)
    monkeypatch.setattr(mod, "MANIFEST", tuple(entries))
    return func(*args, **kwargs)


def _source_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _macro_body(text: str, opening: str) -> str:
    """The brace-matched body that follows ``opening`` in ``text`` (escaped braces ignored)."""
    start = text.index(opening) + len(opening)
    depth = 1
    out: list[str] = []
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            out.append(text[i:i + 2])
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(char)
        i += 1
    raise AssertionError(f"unbalanced braces after {opening!r}")


# ---------------------------------------------------------------------------
# 1-4: the manifest against the current sources
# ---------------------------------------------------------------------------


def test_manifest_ranges_inside_files():
    """Every row names a file that exists and a range inside it."""
    mod = _mod()
    problems = []
    for entry in mod.MANIFEST:
        source = _ROOT / entry.path
        if not source.is_file():
            problems.append(f"p{entry.page}.{entry.order}: {entry.path} is not a file")
            continue
        count = len(_source_lines(source))
        if entry.first < 1:
            problems.append(f"p{entry.page}.{entry.order}: first={entry.first} < 1")
        if entry.last < entry.first:
            problems.append(
                f"p{entry.page}.{entry.order}: last={entry.last} < first={entry.first}")
        if entry.last > count:
            problems.append(
                f"p{entry.page}.{entry.order}: last={entry.last} past {entry.path} ({count} lines)")
    assert problems == []


def test_manifest_tokens_present():
    """Every token of every row occurs in the row's excerpt, line by line."""
    mod = _mod()
    problems = []
    for entry in mod.MANIFEST:
        try:
            lines = mod.excerpt_lines(_ROOT, entry)
        except ValueError as exc:
            problems.append(f"p{entry.page}.{entry.order}: excerpt refused: {exc}")
            continue
        body = "\n".join(lines)
        for token in entry.tokens:
            if token not in body:
                problems.append(
                    f"p{entry.page}.{entry.order}: token {token!r} absent from "
                    f"{entry.path}:{entry.first}-{entry.last}")
    assert problems == []


def test_manifest_pages():
    """The manifest covers the 12 annotated pages, each with orders 1..n and no gaps."""
    mod = _mod()
    assert len(mod.MANIFEST) == _ROWS
    orders: dict[int, list[int]] = {}
    for entry in mod.MANIFEST:
        orders.setdefault(entry.page, []).append(entry.order)
    assert tuple(sorted(orders)) == _PAGES
    assert tuple(mod.PAGES) == _PAGES
    assert {page: len(seen) for page, seen in orders.items()} == _ROWS_PER_PAGE
    for page, seen in sorted(orders.items()):
        assert sorted(seen) == list(range(1, len(seen) + 1)), f"page {page} orders {sorted(seen)}"


def test_slugs_unique_ascii():
    """Slugs are unique, lowercase ASCII, and carry the page and the order."""
    mod = _mod()
    slugs = [mod.slug(entry) for entry in mod.MANIFEST]
    assert len(set(slugs)) == len(slugs)
    for entry, name in zip(mod.MANIFEST, slugs):
        assert _SLUG.match(name), f"p{entry.page}.{entry.order}: slug {name!r}"
        assert name.startswith(f"p{entry.page}_{entry.order}_"), name
        assert name.isascii(), name
        assert "__" not in name, name
        title_part = name[len(f"p{entry.page}_{entry.order}_"):]
        assert len(title_part) <= _SLUG_TITLE_MAX, f"{name!r}: {len(title_part)} characters"


# ---------------------------------------------------------------------------
# 5-6: what ``excerpt_lines`` refuses
# ---------------------------------------------------------------------------


def test_excerpt_lines_refuses_bad_range(tmp_path):
    """A range past the end of the file, an inverted range and first=0 are refused."""
    mod = _mod()
    source = tmp_path / "pkg" / "mod.py"
    source.parent.mkdir(parents=True)
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")

    for first, last, why in ((1, 5, "past the end"), (3, 2, "inverted"), (0, 2, "first below 1")):
        entry = _entry(mod, 3, 1, "a synthetic range", "pkg/mod.py", first, last)
        with pytest.raises(ValueError) as excinfo:
            mod.excerpt_lines(tmp_path, entry)
        message = str(excinfo.value)
        assert entry.path in message or mod.slug(entry) in message, f"{why}: {message!r}"


def test_excerpt_lines_refuses_missing_token(tmp_path):
    """A token absent from the excerpt is refused, and the message names it."""
    mod = _mod()
    source = tmp_path / "pkg" / "mod.py"
    source.parent.mkdir(parents=True)
    source.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    absent = _entry(mod, 3, 1, "a synthetic token", "pkg/mod.py", 1, 2, ("delta",))
    with pytest.raises(ValueError) as excinfo:
        mod.excerpt_lines(tmp_path, absent)
    assert "delta" in str(excinfo.value)

    # A token is matched inside one line: the two lines are not concatenated into one string.
    across = _entry(mod, 3, 1, "a synthetic token", "pkg/mod.py", 1, 2, ("alphabeta",))
    with pytest.raises(ValueError) as excinfo:
        mod.excerpt_lines(tmp_path, across)
    assert "alphabeta" in str(excinfo.value)

    present = _entry(mod, 3, 1, "a synthetic token", "pkg/mod.py", 1, 2, ("alpha", "beta"))
    assert mod.excerpt_lines(tmp_path, present) == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# 7-8: the write path
# ---------------------------------------------------------------------------


def test_write_excerpts_is_verbatim_and_atomic(tmp_path, monkeypatch):
    """Each file is the header plus the verbatim source lines; a bad row writes nothing."""
    mod = _mod()
    source = tmp_path / "pkg" / "mod.py"
    source.parent.mkdir(parents=True)
    source.write_text("one\n  two  \nthree\nfour\nfive\n", encoding="utf-8")

    good_a = _entry(mod, 3, 1, "the middle rows", "pkg/mod.py", 2, 4, ("two",))
    good_b = _entry(mod, 3, 2, "the first rows", "pkg/mod.py", 1, 2)

    outdir = tmp_path / "out"
    written = _call_with_entries(
        mod, monkeypatch, mod.write_excerpts, (good_a, good_b), tmp_path, outdir)

    on_disk = sorted(p.name for p in outdir.glob("*.txt"))
    assert on_disk == sorted(f"{mod.slug(e)}.txt" for e in (good_a, good_b))
    assert sorted(Path(p).name for p in written) == on_disk

    body_a = (outdir / f"{mod.slug(good_a)}.txt").read_text(encoding="utf-8").splitlines()
    assert body_a == ["# pkg/mod.py:2-4", "  two  ", "three", "four"]
    body_b = (outdir / f"{mod.slug(good_b)}.txt").read_text(encoding="utf-8").splitlines()
    assert body_b == ["# pkg/mod.py:1-2", "one", "  two  "]

    # A bad row among good ones: nothing is written, and nothing already there is removed.
    bad = _entry(mod, 4, 2, "past the end", "pkg/mod.py", 4, 9)
    failing = tmp_path / "failing"
    failing.mkdir()
    prior = failing / "p1_1_prior.txt"
    prior.write_text("kept\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _call_with_entries(
            mod, monkeypatch, mod.write_excerpts, (good_a, bad), tmp_path, failing)
    assert sorted(p.name for p in failing.glob("*.txt")) == ["p1_1_prior.txt"]
    assert prior.read_text(encoding="utf-8") == "kept\n"


def test_write_excerpts_removes_stale(tmp_path, monkeypatch):
    """An excerpt file that no row produces is removed by a successful write."""
    mod = _mod()
    source = tmp_path / "pkg" / "mod.py"
    source.parent.mkdir(parents=True)
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")

    entry = _entry(mod, 3, 1, "the first rows", "pkg/mod.py", 1, 2)
    outdir = tmp_path / "out"
    outdir.mkdir()
    stale = outdir / "p9_9_stale.txt"
    stale.write_text("# pkg/mod.py:1-1\none\n", encoding="utf-8")

    _call_with_entries(mod, monkeypatch, mod.write_excerpts, (entry,), tmp_path, outdir)

    assert not stale.exists()
    assert sorted(p.name for p in outdir.glob("*.txt")) == [f"{mod.slug(entry)}.txt"]


# ---------------------------------------------------------------------------
# 9-11: the repository artefacts (deck and excerpts)
# ---------------------------------------------------------------------------


def test_written_excerpts_match_sources():
    """Every tracked excerpt still equals the source lines its header names."""
    assert _EXCERPT_DIR.is_dir(), (
        f"{_EXCERPT_DIR} does not exist: the excerpts have not been written")
    files = sorted(_EXCERPT_DIR.glob("*.txt"))
    assert files, f"{_EXCERPT_DIR} holds no excerpt file"

    problems = []
    headers = {}
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            problems.append(f"{path.name}: empty")
            continue
        match = _HEADER.match(lines[0])
        if match is None:
            problems.append(f"{path.name}: first line is not a header: {lines[0]!r}")
            continue
        headers[path.stem] = match
        source = _ROOT / match.group("path")
        if not source.is_file():
            problems.append(f"{path.name}: {match.group('path')} is not a file")
            continue
        first, last = int(match.group("first")), int(match.group("last"))
        expected = _source_lines(source)[first - 1:last]
        if lines[1:] != expected:
            problems.append(
                f"{path.name}: body differs from {match.group('path')}:{first}-{last} "
                f"({len(lines) - 1} lines written, {len(expected)} lines in the source)")
    assert problems == []

    mod = _mod()
    slugs = {mod.slug(entry): entry for entry in mod.MANIFEST}
    assert set(slugs) == {path.stem for path in files}
    for name, entry in sorted(slugs.items()):
        header = headers[name]
        assert (header.group("path"), int(header.group("first")), int(header.group("last"))) == (
            entry.path, entry.first, entry.last), f"{name}: header disagrees with the manifest"


def test_frames_carry_codenotes():
    """Each page's codenote lines sit, in order, after that page's slidenote."""
    lines = _FRAMES.read_text(encoding="utf-8").splitlines()
    found = [(i, _CODENOTE.match(line)) for i, line in enumerate(lines)]
    notes = [(i, m) for i, m in found if m is not None]
    assert notes, f"{_FRAMES.name} carries no \\codenote line"

    mod = _mod()
    slugs = {mod.slug(entry) for entry in mod.MANIFEST}
    named = [m.group(2) for _, m in notes]
    assert len(named) == len(set(named)), "a slug is cited twice in the frames"
    assert set(named) == slugs
    missing = [name for name in named if not (_EXCERPT_DIR / f"{name}.txt").is_file()]
    assert missing == [], f"cited excerpts absent: {missing}"

    frames = [i for i, line in enumerate(lines) if line.startswith("\\begin{frame}")]
    # page N is the N-th frame: pinned, so a frame added or removed fails here rather than
    # silently re-pointing every manifest page
    assert len(frames) == _FRAME_COUNT == mod.FRAME_COUNT, len(frames)
    for page in mod.PAGES:
        assert page <= len(frames), f"page {page} beyond the {len(frames)} frames"
        start = frames[page - 1]
        stop = frames[page] if page < len(frames) else len(lines)
        notes_here = [i for i in range(start, stop) if lines[i].startswith("\\slidenote{")]
        assert len(notes_here) == 1, f"page {page}: {len(notes_here)} slidenote lines"
        expected = mod.codenote_lines(page)
        assert expected, f"page {page}: no codenote line generated"
        region = lines[notes_here[0] + 1:stop]
        assert expected[0] in region, f"page {page}: first codenote line absent after the slidenote"
        at = region.index(expected[0])
        assert region[at:at + len(expected)] == expected, (
            f"page {page}: the codenote block is not contiguous and in order")


def test_wrappers_define_codenote():
    """The plain wrapper drops the code notes; the annotated one sets them verbatim."""
    plain = [line.rstrip() for line in _PLAIN.read_text(encoding="utf-8").splitlines()]
    assert "\\newcommand{\\codenote}[2]{}" in plain, (
        f"{_PLAIN.name} does not define \\codenote as empty")

    annotated = _ANNOTATED.read_text(encoding="utf-8")
    assert "\\usepackage{fancyvrb}" in annotated, f"{_ANNOTATED.name} does not load fancyvrb"
    opening = "\\newcommand{\\codenote}[2]{"
    assert opening in annotated, f"{_ANNOTATED.name} does not define \\codenote"
    body = _macro_body(annotated, opening)
    assert "\\VerbatimInput" in body, "the annotated \\codenote does not set the file verbatim"
    # a fragile frame is re-read from the input file up to a line \end{frame}; a frame born
    # from a macro has no such line and the scan runs past the call, so the frame is not
    # fragile: \VerbatimInput reads its material from the excerpt file and needs no fragility
    assert "fragile" not in body, "the annotated \\codenote frame must not be fragile"
    assert "allowframebreaks" in body, "a long excerpt must break over frames"
    assert "#1" in body and "#2" in body, "the annotated \\codenote drops one of its arguments"


# ---------------------------------------------------------------------------
# 12-13: the two rendered strings
# ---------------------------------------------------------------------------


def test_codenote_lines_escape_latex(monkeypatch):
    """The title and the path are escaped for LaTeX; the file argument is the slug."""
    mod = _mod()
    entry = _entry(mod, 3, 1, "a_b|c^d&e", "pkg/x_y.py", 1, 2)
    rendered = _call_with_entries(mod, monkeypatch, mod.codenote_lines, (entry,), 3)
    assert len(rendered) == 1, rendered
    match = _CODENOTE.match(rendered[0])
    assert match is not None, rendered[0]
    assert match.group(1) == "a\\_b$|$c\\^{}d\\&e [pkg/x\\_y.py:1--2]"
    assert match.group(2) == mod.slug(entry)


def test_header_names_path_and_range():
    """The header names the file and the range, nothing else."""
    mod = _mod()
    entry = _entry(mod, 3, 1, "k_F, s and the density floor", "xcquinox/alec/networks.py", 319, 324)
    assert mod.header(entry) == "# xcquinox/alec/networks.py:319-324"


# ---------------------------------------------------------------------------
# 14-15: the command line and the two deck builds
# ---------------------------------------------------------------------------


def test_main_writes_and_prints(tmp_path, capsys):
    """``main`` prints every page's codenote lines under a ``% page N`` line, and with
    ``--write`` writes the excerpts into ``--outdir``; without it nothing is written."""
    mod = _mod()
    outdir = tmp_path / "slides_code"

    assert mod.main(["--outdir", str(outdir)]) == 0
    assert not outdir.exists(), "main without --write wrote excerpts"
    printed = capsys.readouterr().out.splitlines()
    for page in mod.PAGES:
        assert f"% page {page}" in printed, f"page {page} missing from the listing"
        at = printed.index(f"% page {page}")
        expected = mod.codenote_lines(page)
        assert printed[at + 1:at + 1 + len(expected)] == expected, f"page {page}"

    assert mod.main(["--write", "--outdir", str(outdir)]) == 0
    files = sorted(p.name for p in outdir.glob("*.txt"))
    assert files == sorted(f"{mod.slug(e)}.txt" for e in mod.MANIFEST)
    assert len(files) == _ROWS
    listing = capsys.readouterr().out.splitlines()
    assert f"% page {mod.PAGES[0]}" in listing


def _pdf_pages(pdf: Path) -> int:
    """The page count of a PDF, read by pdfinfo: xelatex writes object streams, so the page
    objects are not visible as plain bytes."""
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    match = re.search(r"^Pages:\s+(\d+)$", out, re.M)
    assert match is not None, out
    return int(match.group(1))


@pytest.mark.skipif(shutil.which("xelatex") is None or shutil.which("pdfinfo") is None,
                    reason="xelatex or pdfinfo not installed")
def test_decks_build(tmp_path):
    """Both wrappers compile from the analysis directory: the plain deck keeps its 24 pages,
    the annotated one grows by the code frames, and no glyph of an excerpt is missing."""
    results = {}
    for wrapper in (_PLAIN, _ANNOTATED):
        out = tmp_path / wrapper.stem
        out.mkdir()
        proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={out}", wrapper.name],
            cwd=_HERE, capture_output=True, text=True, timeout=900)
        log = (out / f"{wrapper.stem}.log").read_text(encoding="utf-8", errors="replace")
        assert proc.returncode == 0, f"{wrapper.name}: xelatex failed\n{log[-3000:]}"
        assert "Missing character" not in log, f"{wrapper.name}: a glyph is missing"
        results[wrapper.stem] = _pdf_pages(out / f"{wrapper.stem}.pdf")
    assert results[_PLAIN.stem] == 24, results
    assert results[_ANNOTATED.stem] > 56, results

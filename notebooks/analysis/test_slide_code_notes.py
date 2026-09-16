"""Tests for ``notebooks/analysis/slide_code_notes.py`` (the code excerpts of the annotated deck).

The module under test carries a manifest of 88 (page, order, title, file, line range, tokens)
rows, cuts the named line ranges out of the repository sources into
``notebooks/analysis/slides_code/<slug>.txt``, renders the appendix file that sets one labelled
verbatim frame per row, and renders the pointer sentence with which the note of a manifest page
ends. Since 2026-09-16 the excerpts no longer sit behind their slide: the frames file carries no
``\\codenote`` line, the annotated wrapper inputs the frames file and then the appendix, and a
note reaches its excerpts through ``\\pageref``, which two xelatex passes resolve.

The requirements below are the manifest's ranges and tokens against the current sources, the
slug rule, the refusals of ``excerpt_lines``, the write path with its atomicity and its stale
removal, the drift detector on the tracked excerpt files, the pointer at the end of every
manifest page's note, the notes contract (the four headings in order, the word cap, the pointer
present exactly on the manifest pages), the two wrappers, the rendered appendix and pointer
strings, the excerpt header format, the tracked appendix file, the command line, and the build
of both decks.

The module is loaded by path inside each test, as the sibling ``test_md_to_tex`` does, so that
while it is absent every requirement reports its own ImportError instead of the file collapsing
into one collection error. The tests that read the deck files assert on those files before the
module is touched: their first failure is the missing artefact, not the missing module.

Mutation coverage: a token dropped from a row -> test_excerpt_lines_refuses_missing_token; a
range past the end of its file -> test_excerpt_lines_refuses_bad_range; a partial write left
behind -> test_write_excerpts_is_verbatim_and_atomic; a line range that no longer holds what it
claims -> test_manifest_tokens_present and test_written_excerpts_match_sources; a ``\\codenote``
line left in the frames file -> test_frames_carry_pointers; a label dropped from the appendix, or
a pointer naming a slug that no frame labels -> test_appendix_file and the unresolved-reference
check of test_decks_build; a note without its last heading, or padded past the word cap ->
test_notes_contract; the annotated wrapper not inputting the appendix -> test_wrappers and
test_decks_build; a notes frame padded to three pages -> test_decks_build.

Standard library and pytest only; no repository package is imported, and no test writes into the
repository tree (the synthetic sources and outputs live under ``tmp_path``). ``_plain`` and
``_normalize`` are copied from ``test_slides_overflow`` rather than imported from it: a note's
words are counted after the same LaTeX stripping that test applies to a slide, and the two files
stay independent of one another.
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
_FRAMES = _HERE / "SLIDES_v7_2026-09-15_frames.tex"
_PLAIN = _HERE / "SLIDES_v7_2026-09-15.tex"
_ANNOTATED = _HERE / "SLIDES_v7_2026-09-15_annotated.tex"

# The pages the manifest covers and the number of rows on each, from the manifest itself
# (88 rows over 15 pages since 2026-09-16: 5(7) 6(5) 7(3) 8(3) 9(5) 10(6) 11(6) 12(3) 13(1)
# 15(10) 16(5) 21(12) 22(6) 23(5) 27(11)). The deck was restructured that day into 37 main and
# 8 backup frames, which split three annotated pages in two (old 5 -> 7 and 8, old 9 -> 12 and
# 13, old 16 -> 22 and 23) and moved the rest; the row count is unchanged.
_PAGES = (5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 21, 22, 23, 27)
_ROWS_PER_PAGE = {5: 7, 6: 5, 7: 3, 8: 3, 9: 5, 10: 6, 11: 6, 12: 3, 13: 1, 15: 10, 16: 5,
                  21: 12, 22: 6, 23: 5, 27: 11}
_ROWS = 88

# ``# <path>:<first>-<last>`` -- the first line of every excerpt file. No commit hash: the
# excerpts are written before the commit that adds them, so a hash could only name the parent;
# the drift test below is the provenance.
_HEADER = re.compile(r"^# (?P<path>[^\s:]+):(?P<first>\d+)-(?P<last>\d+)$")

# The frames file holds this many ``\begin{frame}`` lines; page N of the plain deck is the N-th
# (25 since 2026-09-15, when the per-cell table frame became two frames over 64 cells; 45 since
# 2026-09-16, when the deck was restructured into 37 main and 8 backup frames).
_FRAME_COUNT = 45

# A distinctive part of the title of the frame each manifest page names: the count pins how many
# frames there are, this pins which frame each page is, so a reordering of the frames cannot
# re-point the code notes silently (2026-09-16).
_PAGE_TITLES = {
    5: "density variables", 6: "the extra descriptors", 7: "the network",
    8: "the functional form", 9: "the architectures", 10: "the 26-point DFS pool",
    11: "the descriptor distributions", 12: "the metric", 13: "the search",
    15: "targets and objective", 16: "the fidelity certificate", 21: "the training objective",
    22: "the training losses", 23: "validation and the arms", 27: "the set and the metrics",
}

# The opening of a note line, and the four headings its body carries in this order (the notes
# contract of 2026-09-16). On a manifest page the pointer sentence follows them; its invariant
# opening is ``_POINTER_MARK``, the rest is rendered by ``pointer_line``.
_SLIDENOTE_OPEN = "\\slidenote{"
_HEADINGS = ("\\textbf{On the slide:}", "\\textbf{The point:}", "\\textbf{Terms:}",
             "\\textbf{If asked:}")
_POINTER_MARK = "Code: appendix"

# A note is read from the built frame while the slide before it is on screen, so it is capped:
# this counts the words, test_decks_build measures the pages the frame takes.
_NOTE_WORD_MAX = 340   # tokens after the stripping below; measured: 282 to 390 such tokens already fill two pages

# The three lines of an appendix frame: the title carrying the page and the order of its row,
# the label the pointer refers to, and the excerpt set verbatim.
_APPENDIX_FRAME = re.compile(r"^\\begin\{frame\}\[allowframebreaks\]\{Code p(\d+)\.(\d+): ")
_APPENDIX_LABEL = re.compile(r"^\\label\{(code:[^{}]+)\}$")
_APPENDIX_VERBATIM = re.compile(r"^\\VerbatimInput\[fontsize=\\tiny\]\{([^{}]+)\}$")

# An ``\input`` line of a wrapper, and one page entry of a built deck's nav file.
_INPUT = re.compile(r"^\s*\\input\{([^{}]*)\}", re.M)
_FRAMEPAGES = re.compile(r"\\beamer@framepages\s*\{(\d+)\}\{(\d+)\}")

_SLUG = re.compile(r"^p\d+_\d+_[a-z0-9_]+$")

# The slug's title part is capped; the cap is asserted in the weaker of the two readings of the
# page ("at most 40 characters of slug"), which both a whole-slug and a title-part cap satisfy.
_SLUG_TITLE_MAX = 40

# The LaTeX stripping of ``test_slides_overflow``, copied so that the two files stay independent
# of one another: math is dropped, the commands whose braced argument is text keep that argument,
# every other command goes with its argument, and the ligature characters are mapped back.
_KEEP_COMMANDS = ("textbf", "texttt", "path", "emph", "textit")
_KEEP = re.compile(r"\\(?:" + "|".join(_KEEP_COMMANDS) + r")\s*\{([^{}]*)\}")
_DROP_ARG = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\s*\{[^{}]*\}")
_BARE = re.compile(r"\\[A-Za-z]+\*?")
_DISPLAY_MATH = re.compile(r"\\\[.*?\\\]", re.S)
_INLINE_MATH = re.compile(r"\\\(.*?\\\)", re.S)
_DOLLAR_MATH = re.compile(r"\$[^$]*\$")
_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\ufb05": "st", "\ufb06": "st",
}


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


def _plain(raw: str) -> str:
    """The text of a LaTeX fragment: math removed, markup resolved, escapes undone.

    Copied from ``test_slides_overflow`` so that the two deck test files stay independent."""
    text = _DISPLAY_MATH.sub(" ", raw)
    text = _INLINE_MATH.sub(" ", text)
    text = _DOLLAR_MATH.sub(" ", text)
    for pattern in (_KEEP, _DROP_ARG):
        while True:
            text, changed = pattern.subn(r"\1" if pattern is _KEEP else " ", text)
            if not changed:
                break
    text = _BARE.sub(" ", text)
    text = re.sub(r"\\([_%&#$])", r"\1", text)
    for char in "{}\\":
        text = text.replace(char, " ")
    return text


def _normalize(text: str) -> str:
    """The comparable form of a text: ligatures mapped back, every run of characters that is not
    a letter or a digit collapsed to one space. Copied from ``test_slides_overflow``."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    return " ".join(re.sub(r"[^A-Za-z0-9]+", " ", text).split())


def _frames_lines() -> list[str]:
    """The lines of the frames file, asserted present before the module is loaded."""
    assert _FRAMES.is_file(), f"{_FRAMES} does not exist"
    return _FRAMES.read_text(encoding="utf-8").splitlines()


def _frame_starts(lines: list[str]) -> list[int]:
    """The indices of the ``\\begin{frame}`` lines: page N of the plain deck is the N-th."""
    return [i for i, line in enumerate(lines) if line.startswith("\\begin{frame}")]


def _notes_of_frames(lines: list[str]) -> dict[int, str]:
    """The one ``\\slidenote`` line of each frame, by frame number.

    A note follows its frame's ``\\end{frame}``, so the note of frame N is the line between the
    N-th and the (N+1)-th ``\\begin{frame}``. A frame carrying none, or more than one, fails here
    rather than shifting which note every later page is read against.
    """
    starts = _frame_starts(lines)
    notes: dict[int, str] = {}
    for number, start in enumerate(starts, start=1):
        stop = starts[number] if number < len(starts) else len(lines)
        here = [lines[i] for i in range(start, stop) if lines[i].startswith(_SLIDENOTE_OPEN)]
        assert len(here) == 1, f"frame {number}: {len(here)} \\slidenote lines, expected one"
        notes[number] = here[0]
    return notes


# ---------------------------------------------------------------------------
# the manifest against the current sources
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
    """The manifest covers the 15 annotated pages, each with orders 1..n and no gaps."""
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
# what ``excerpt_lines`` refuses
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
# the write path
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
# the repository artefacts: the excerpts, the frames file, the wrappers and the appendix
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


def test_frames_carry_pointers():
    """The frames file cites the appendix instead of setting the code after each slide.

    No ``\\codenote`` line survives, every frame carries exactly one note, and the note of a
    manifest page ends in the pointer sentence ``pointer_line`` renders for that page. The page
    identity is pinned twice over: the frame count, and a distinctive part of the title of the
    frame each manifest page names, so a frame added, removed or reordered fails here instead of
    re-pointing every excerpt silently.
    """
    lines = _frames_lines()
    stray = [i + 1 for i, line in enumerate(lines) if line.startswith("\\codenote")]
    assert stray == [], f"{_FRAMES.name} still carries \\codenote lines (source lines {stray})"

    mod = _mod()
    starts = _frame_starts(lines)
    assert len(starts) == _FRAME_COUNT == mod.FRAME_COUNT, len(starts)
    notes = _notes_of_frames(lines)

    for page in mod.PAGES:
        assert page <= len(starts), f"page {page} beyond the {len(starts)} frames"
        assert _PAGE_TITLES[page] in lines[starts[page - 1]], (
            f"page {page}: the frame is {lines[starts[page - 1]]!r}, not the one the manifest "
            f"rows describe")
        pointer = mod.pointer_line(page)
        assert notes[page].endswith(pointer + "}"), (
            f"page {page}: the note does not end in the pointer {pointer!r}")

    pages = set(mod.PAGES)
    elsewhere = [number for number, note in sorted(notes.items())
                 if number not in pages and _POINTER_MARK in note]
    assert elsewhere == [], f"frames with no manifest row citing the appendix: {elsewhere}"


def test_notes_contract():
    """Every note carries the four headings in order, stays inside the word cap, and cites the
    appendix on the manifest pages and nowhere else.

    The body is the brace-matched argument of the one-line ``\\slidenote``; the words are counted
    after the LaTeX stripping ``test_slides_overflow`` applies to a slide, so a ``\\par`` and the
    ``\\pageref`` of the pointer count for nothing while the headings count as their text.
    """
    mod = _mod()
    notes = _notes_of_frames(_frames_lines())
    assert len(notes) == _FRAME_COUNT, len(notes)
    pages = set(mod.PAGES)

    problems: list[str] = []
    for number, line in sorted(notes.items()):
        body = _macro_body(line, _SLIDENOTE_OPEN)
        at = -1
        for heading in _HEADINGS:
            found = body.find(heading, at + 1)
            if found < 0:
                problems.append(f"frame {number}: {heading} is absent, or out of order")
                break
            at = found
        words = len(_normalize(_plain(body)).split())
        if words > _NOTE_WORD_MAX:
            problems.append(f"frame {number}: {words} words, the cap is {_NOTE_WORD_MAX}")
        cites = _POINTER_MARK in body
        if cites != (number in pages):
            problems.append(
                f"frame {number}: the appendix is {'cited' if cites else 'not cited'} and the "
                f"page {'carries a' if number in pages else 'carries no'} manifest row")
    assert problems == [], "\n".join(problems)


def test_wrappers():
    """Neither wrapper knows ``\\codenote`` any more.

    The annotated one loads fancyvrb (the appendix sets the excerpts with ``\\VerbatimInput``),
    keeps ``\\slidenote`` as a frame that may break, and inputs the frames file and then the
    appendix, which is the order that puts the code frames after the deck; the plain wrapper
    inputs the frames file alone.
    """
    mod = _mod()
    plain = _PLAIN.read_text(encoding="utf-8")
    annotated = _ANNOTATED.read_text(encoding="utf-8")
    for path, text in ((_PLAIN, plain), (_ANNOTATED, annotated)):
        assert "\\codenote" not in text, f"{path.name} still names \\codenote"

    assert "\\usepackage{fancyvrb}" in annotated, f"{_ANNOTATED.name} does not load fancyvrb"
    opening = "\\newcommand{\\slidenote}[1]{"
    assert opening in annotated, f"{_ANNOTATED.name} does not define \\slidenote"
    body = _macro_body(annotated, opening)
    assert "\\begin{frame}" in body, "the annotated \\slidenote is not a frame"
    assert "allowframebreaks" in body, "a long note must break over frames"
    assert "#1" in body, "the annotated \\slidenote drops its argument"

    inputs = _INPUT.findall(annotated)
    assert inputs == [_FRAMES.name, mod.APPENDIX_NAME], (
        f"{_ANNOTATED.name} inputs {inputs}: the frames file and then the appendix is what puts "
        f"the excerpts behind the deck")
    assert _INPUT.findall(plain) == [_FRAMES.name], (
        f"{_PLAIN.name} inputs {_INPUT.findall(plain)}: the appendix belongs to the annotated "
        f"deck alone")


def test_appendix_file():
    """The tracked appendix file is what ``appendix_lines`` renders.

    One labelled verbatim frame per manifest row, the frames in (page, order) order, the labels
    unique, and every file a frame sets verbatim present in the excerpt directory.
    """
    mod = _mod()
    path = _HERE / mod.APPENDIX_NAME
    assert path.is_file(), f"{path} does not exist: the appendix has not been written"
    lines = path.read_text(encoding="utf-8").splitlines()
    rendered = mod.appendix_lines()
    if lines != rendered:
        first = next(i for i in range(max(len(lines), len(rendered)))
                     if lines[i:i + 1] != rendered[i:i + 1])
        raise AssertionError(
            f"{path.name} ({len(lines)} lines) differs from appendix_lines() "
            f"({len(rendered)} lines) at line {first + 1}: {lines[first:first + 1]} against "
            f"{rendered[first:first + 1]}")

    order = sorted(mod.MANIFEST, key=lambda e: (e.page, e.order))
    titles = [line for line in lines if line.startswith("\\begin{frame}")]
    closes = [line for line in lines if line.startswith("\\end{frame}")]
    assert len(titles) == len(closes) == _ROWS, f"{len(titles)} frames, {len(closes)} closed"
    seen = []
    for line in titles:
        match = _APPENDIX_FRAME.match(line)
        assert match is not None, f"not an appendix frame title: {line!r}"
        seen.append((int(match.group(1)), int(match.group(2))))
    assert seen == [(e.page, e.order) for e in order], (
        "the appendix frames are not one per row in (page, order) order")

    named = []
    for line in [line for line in lines if line.startswith("\\label{")]:
        match = _APPENDIX_LABEL.match(line)
        assert match is not None, f"not a code label: {line!r}"
        named.append(match.group(1))
    assert len(named) == _ROWS, f"{len(named)} labels for {_ROWS} rows"
    assert len(set(named)) == len(named), "a label is written twice"
    assert named == [f"code:{mod.slug(e)}" for e in order]

    files = []
    for line in [line for line in lines if line.startswith("\\VerbatimInput")]:
        match = _APPENDIX_VERBATIM.match(line)
        assert match is not None, f"not a verbatim line: {line!r}"
        files.append(match.group(1))
    assert len(files) == _ROWS, f"{len(files)} verbatim lines for {_ROWS} rows"
    missing = [name for name in files if not (_HERE / name).is_file()]
    assert missing == [], f"excerpts the appendix sets but that are absent: {missing}"


# ---------------------------------------------------------------------------
# the rendered strings
# ---------------------------------------------------------------------------


def test_pointer_and_appendix_lines_escape_latex(monkeypatch):
    """The appendix frame escapes the title and the path for LaTeX, labels itself by the row's
    slug and sets the row's excerpt verbatim; the pointer names one appendix page, or the range
    of them, and a page with no row of its own is refused."""
    mod = _mod()
    one = _entry(mod, 3, 1, "a_b^c&d", "pkg/x_y.py", 1, 2)
    two = _entry(mod, 3, 2, "e&f", "pkg/x_y.py", 3, 4)

    rendered = _call_with_entries(mod, monkeypatch, mod.appendix_lines, (one,))
    assert rendered[0].startswith("\\section*{"), rendered[0]
    assert rendered[1:] == [
        "\\begin{frame}[allowframebreaks]{Code p3.1: a\\_b\\^{}c\\&d [pkg/x\\_y.py:1--2]}",
        f"\\label{{code:{mod.slug(one)}}}",
        f"\\VerbatimInput[fontsize=\\tiny]{{slides_code/{mod.slug(one)}.txt}}",
        "\\end{frame}",
    ], rendered

    single = _call_with_entries(mod, monkeypatch, mod.pointer_line, (one,), 3)
    assert single == f"Code: appendix p. \\pageref{{code:{mod.slug(one)}}}."
    pair = _call_with_entries(mod, monkeypatch, mod.pointer_line, (one, two), 3)
    assert pair == (f"Code: appendix pp. \\pageref{{code:{mod.slug(one)}}} to "
                    f"\\pageref{{code:{mod.slug(two)}}}.")

    with pytest.raises(ValueError):
        _call_with_entries(mod, monkeypatch, mod.pointer_line, (one, two), 4)


def test_header_names_path_and_range():
    """The header names the file and the range, nothing else."""
    mod = _mod()
    entry = _entry(mod, 3, 1, "k_F, s and the density floor", "xcquinox/alec/networks.py", 319, 324)
    assert mod.header(entry) == "# xcquinox/alec/networks.py:319-324"


# ---------------------------------------------------------------------------
# the command line and the two deck builds
# ---------------------------------------------------------------------------


def test_main_writes_and_prints(tmp_path, capsys):
    """``main`` prints the pointer line of every manifest page under a ``% page N`` line, and
    with ``--write`` writes the excerpts into ``--outdir`` and the appendix to ``--appendix``;
    without ``--write`` neither is written.

    Both paths are passed on both calls, so that no run of this test can reach the tracked
    appendix beside the module, which is where ``--appendix`` defaults to."""
    mod = _mod()
    outdir = tmp_path / "slides_code"
    appendix = tmp_path / "appendix.tex"

    assert mod.main(["--outdir", str(outdir), "--appendix", str(appendix)]) == 0
    assert not outdir.exists(), "main without --write wrote excerpts"
    assert not appendix.exists(), "main without --write wrote the appendix"
    printed = capsys.readouterr().out.splitlines()
    assert [line for line in printed if line.startswith("\\codenote")] == [], (
        "main still prints \\codenote lines")
    for page in mod.PAGES:
        assert f"% page {page}" in printed, f"page {page} missing from the listing"
        at = printed.index(f"% page {page}")
        assert printed[at + 1] == mod.pointer_line(page), f"page {page}: {printed[at + 1]!r}"

    assert mod.main(["--write", "--outdir", str(outdir), "--appendix", str(appendix)]) == 0
    files = sorted(p.name for p in outdir.glob("*.txt"))
    assert files == sorted(f"{mod.slug(e)}.txt" for e in mod.MANIFEST)
    assert len(files) == _ROWS
    assert appendix.read_text(encoding="utf-8").splitlines() == mod.appendix_lines()
    listing = capsys.readouterr().out.splitlines()
    assert f"% page {mod.PAGES[0]}" in listing


def _pdf_pages(pdf: Path) -> int:
    """The page count of a PDF, read by pdfinfo: xelatex writes object streams, so the page
    objects are not visible as plain bytes."""
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    match = re.search(r"^Pages:\s+(\d+)$", out, re.M)
    assert match is not None, out
    return int(match.group(1))


def _pdf_text(pdf: Path) -> str:
    """The text of the whole document, read by pdftotext."""
    out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, check=True).stdout
    return out.decode("utf-8", errors="replace")


def _flowed(text: str) -> str:
    """The text as one line: the end-of-line hyphens of TeX's hyphenation removed and every run
    of whitespace collapsed, so that a sentence broken over two lines is still one phrase."""
    return " ".join(re.sub(r"-\s*\n\s*", "", text).split())


def _frame_spans(nav: Path) -> list[tuple[int, int]]:
    """The (first page, last page) of every frame of a built deck, from its nav file.

    Beamer writes one ``\\beamer@framepages {first}{last}`` entry per built page, and the
    consecutive entries of one frame repeat that frame's first page: a frame breaking over pages
    46 and 47 writes {46}{46} then {46}{47}, while two one-page frames write {46}{46} then
    {47}{47}. The entries are therefore grouped by their first page and the last entry of a group
    carries the frame's last page. The grouping rests on the pages rising by one from frame to
    frame, which is asserted here: a reset page counter, or a frame that produced no page, would
    fold two frames into one span and is refused rather than counted.
    """
    entries = [(int(first), int(last)) for first, last
               in _FRAMEPAGES.findall(nav.read_text(encoding="utf-8", errors="replace"))]
    assert entries, f"{nav} carries no page entry"
    spans: list[tuple[int, int]] = []
    for first, last in entries:
        if spans and spans[-1][0] == first:
            spans[-1] = (first, last)
        else:
            spans.append((first, last))
    assert spans[0][0] == 1, f"the first frame starts on page {spans[0][0]}"
    jumps = [(i + 1, spans[i - 1], spans[i]) for i in range(1, len(spans))
             if spans[i][0] != spans[i - 1][1] + 1]
    assert jumps == [], f"the frames do not run over consecutive pages: {jumps[:3]}"
    return spans


@pytest.mark.skipif(shutil.which("xelatex") is None or shutil.which("pdfinfo") is None
                    or shutil.which("pdftotext") is None,
                    reason="xelatex, pdfinfo or pdftotext not installed")
def test_decks_build(tmp_path):
    """Both wrappers compile from the analysis directory.

    The plain deck holds its 45 pages (24 until 2026-09-15, when the per-cell table frame became
    two frames over 64 cells; 25 until 2026-09-16, when the deck was restructured into 37 main
    and 8 backup frames). The annotated wrapper is built twice into one output directory, which
    is what resolves a pointer's ``\\pageref``: the first pass writes the labels into the aux
    file, the second reads them. Its frames are then the 45 slides, the 45 notes and the 88
    appendix frames; a notes frame breaks over at most two pages; the frames account for every
    page of the document; and the text carries no unresolved reference and cites the appendix
    once per manifest page.
    """
    def build(wrapper: Path, out: Path) -> None:
        proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={out}", wrapper.name],
            cwd=_HERE, capture_output=True, text=True, timeout=900)
        log = (out / f"{wrapper.stem}.log").read_text(encoding="utf-8", errors="replace")
        assert proc.returncode == 0, f"{wrapper.name}: xelatex failed\n{log[-3000:]}"
        assert "Missing character" not in log, f"{wrapper.name}: a glyph is missing"

    plain_out = tmp_path / _PLAIN.stem
    plain_out.mkdir()
    build(_PLAIN, plain_out)
    assert _pdf_pages(plain_out / f"{_PLAIN.stem}.pdf") == _FRAME_COUNT == 45

    out = tmp_path / _ANNOTATED.stem
    out.mkdir()
    for _ in range(2):
        build(_ANNOTATED, out)
    pdf = out / f"{_ANNOTATED.stem}.pdf"
    pages = _pdf_pages(pdf)
    spans = _frame_spans(out / f"{_ANNOTATED.stem}.nav")
    assert len(spans) == _FRAME_COUNT + _FRAME_COUNT + _ROWS, (
        f"{len(spans)} frames built, expected {_FRAME_COUNT} slides, {_FRAME_COUNT} notes and "
        f"{_ROWS} appendix frames")
    accounted = sum(last - first + 1 for first, last in spans)
    assert accounted == pages, f"the frames account for {accounted} of the {pages} pages"

    # the notes frame of slide N is the frame after it: the even frame numbers up to 90
    long_notes = [(2 * i + 2, spans[2 * i + 1]) for i in range(_FRAME_COUNT)
                  if spans[2 * i + 1][1] - spans[2 * i + 1][0] + 1 > 2]
    assert long_notes == [], f"notes frames (number, pages) running past two pages: {long_notes}"

    log = (out / f"{_ANNOTATED.stem}.log").read_text(encoding="utf-8", errors="replace")
    if "Label(s) may have changed" in log:
        # the pointers' width changed between the passes and reflowed a page: one more pass
        build(_ANNOTATED, out)
        log = (out / f"{_ANNOTATED.stem}.log").read_text(encoding="utf-8", errors="replace")
    assert "Label(s) may have changed" not in log, "the page references have not converged"
    text = _pdf_text(pdf)
    flowed = _flowed(text)
    assert not re.search(r"appendix pp?\. \?\?", flowed), (
        "an unresolved reference in a pointer: the labels were not read")
    pointers = re.findall(r"Code: appendix pp?\. (\d+)(?: to (\d+))?\.", flowed)
    assert len(pointers) == len(_PAGES), f"{len(pointers)} appendix pointers in the text, {len(_PAGES)} pages"
    # every pointer names the page on which its manifest page's first excerpt frame begins
    for page, (first, last) in zip(_PAGES, pointers):
        target = int(first)
        head = subprocess.run(["pdftotext", "-f", str(target), "-l", str(target), str(pdf), "-"],
                              capture_output=True, check=True).stdout.decode("utf-8", "replace")
        assert f"Code p{page}.1" in _flowed(head), (
            f"page {page}: its pointer names page {target}, which does not carry Code p{page}.1")
        if last:
            assert int(last) >= target, f"page {page}: the range {first} to {last} is inverted"

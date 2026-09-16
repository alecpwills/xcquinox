"""Layout rules of the plain results deck ``SLIDES_v7_2026-09-15.tex``.

The deck is written as one frames file, ``SLIDES_v7_2026-09-15_frames.tex``, shared by the plain
and the annotated wrapper. Since 2026-09-16 it is restructured into 37 main frames followed by
8 backup frames after the line ``\\section*{Backup}``; page N of the plain deck is the N-th
``\\begin{frame}`` of the file, which is the identity the code-note manifest of
``slide_code_notes.py`` also rests on.

Two kinds of requirement are checked here.

Static (no build): a main frame carries at most five ``\\item`` lines, no ``\\tiny`` and no
tabular with more than eight body rows, no frame is allowed to break over pages, the frame count
is 45 with 8 backup frames, every ``\\hyperlink{x}{...}`` names a frame carrying ``[label=x]``,
and every backup frame carries a return link to a main frame.

Built (xelatex, pdfinfo, pdftotext; skipped when any is absent): the page count of the plain
deck equals the frame count, and every frame's text reaches its page. Beamer does not report an
overfull frame as an error, so material pushed past the bottom or the right edge of a slide is
silent in the log and visible only in the rendered page: the last three words of every
``\\item`` and the outer cells of every tabular's last body row are looked up in the text of the
page that frame produced. Those positions are the ones a cut removes first.

The word rule keeps runs of letters and digits of at least three characters after LaTeX markup,
math and braces are stripped, which makes the check insensitive to how the page breaks its lines
(each word is looked up on its own) but leaves it blind to an item whose tail is entirely math
or short numbers. Three readings of a page are accepted: the text with whitespace collapsed, the
same with an end-of-line hyphen removed (TeX hyphenates, the source word does not carry the
hyphen) and the same with all whitespace removed (the url package breaks a long ``\\path`` inside
a narrow column). Ligature characters are mapped back to their letters.

Standard library and pytest only; nothing is written into the repository tree (the build goes to
``tmp_path``).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple

import pytest

_HERE = Path(__file__).resolve().parent
_FRAMES = _HERE / "SLIDES_v7_2026-09-15_frames.tex"
_PLAIN = _HERE / "SLIDES_v7_2026-09-15.tex"

# The restructured layout of 2026-09-16: 37 main frames, then the backup section.
_FRAME_COUNT = 45
_BACKUP_COUNT = 8
_BACKUP_SECTION = "\\section*{Backup}"

# Rules of a main frame.
_MAX_ITEMS = 5
_MAX_BODY_ROWS = 8

# Commands whose braced argument is text on the slide; every other command is dropped with its
# argument (\includegraphics, \vspace, \begin, ...).
_KEEP_COMMANDS = ("textbf", "texttt", "path", "emph", "textit")

_BEGIN_FRAME = "\\begin{frame}"
_END_FRAME = "\\end{frame}"

_ROW_END = re.compile(r"\\\\(\s*\[[^\]]*\])?\s*$")
_WORD = re.compile(r"[A-Za-z0-9]+")
_HYPERLINK = re.compile(r"\\hyperlink\s*\{([^}]*)\}")
_LABEL = re.compile(r"\blabel\s*=\s*\{?([^,\]{}]+)\}?")
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


class Frame(NamedTuple):
    """One ``\\begin{frame}`` block of the frames file; ``number`` is its page."""

    number: int
    line: int                  # 1-based source line of the \begin{frame}
    options: str               # the [...] of \begin{frame}[...]
    title: str                 # the {...} that follows
    body: Tuple[str, ...]      # the lines between \begin{frame} and \end{frame}


# ---------------------------------------------------------------------------
# the frames file, parsed
# ---------------------------------------------------------------------------


def _lines() -> List[str]:
    assert _FRAMES.is_file(), f"{_FRAMES} does not exist"
    return _FRAMES.read_text(encoding="utf-8").splitlines()


def _balanced(text: str, start: int, opening: str, closing: str) -> Tuple[str, int]:
    """The content between ``text[start] == opening`` and its match, and the index past it."""
    assert text[start] == opening, (text, start)
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise AssertionError(f"unbalanced {opening!r} in {text!r}")


def _frame_head(line: str) -> Tuple[str, str]:
    """The options and the title of a ``\\begin{frame}`` line (both possibly empty)."""
    rest = line[len(_BEGIN_FRAME):].lstrip()
    options = ""
    if rest.startswith("["):
        options, at = _balanced(rest, 0, "[", "]")
        rest = rest[at:].lstrip()
    title = ""
    if rest.startswith("{"):
        title, _ = _balanced(rest, 0, "{", "}")
    return options, title


def _frames(lines: Sequence[str]) -> List[Frame]:
    """The frames, in file order: from a line starting ``\\begin{frame}`` to its ``\\end{frame}``.

    A frame title is read from the ``\\begin{frame}`` line itself; a title carried over several
    source lines is not supported and would be reported as an empty title.
    """
    frames: List[Frame] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith(_BEGIN_FRAME):
            i += 1
            continue
        depth = 0
        j = i
        while j < len(lines):
            if lines[j].startswith(_BEGIN_FRAME):
                depth += 1
            elif lines[j].lstrip().startswith(_END_FRAME):
                depth -= 1
                if depth == 0:
                    break
            j += 1
        assert depth == 0, f"line {i + 1}: {_BEGIN_FRAME} without {_END_FRAME}"
        options, title = _frame_head(lines[i])
        frames.append(Frame(len(frames) + 1, i + 1, options, title, tuple(lines[i + 1:j])))
        i = j + 1
    return frames


def _split_main_backup(lines: Sequence[str],
                       frames: Sequence[Frame]) -> Tuple[List[Frame], List[Frame]]:
    """The main frames and the backup frames: those after the backup section line.

    While that line is absent every frame is main, so the frame-count requirement reports the
    missing section rather than the parser reporting nothing.
    """
    at = None
    for index, line in enumerate(lines):
        if line.startswith(_BACKUP_SECTION):
            at = index + 1          # 1-based, as Frame.line
            break
    if at is None:
        return list(frames), []
    main = [frame for frame in frames if frame.line < at]
    backup = [frame for frame in frames if frame.line > at]
    return main, backup


def _items(frame: Frame) -> List[str]:
    """The raw text of every ``\\item`` of a frame, each joined over its continuation lines."""
    out: List[str] = []
    current: str = ""
    open_item = False
    for line in frame.body:
        stripped = line.lstrip()
        if stripped.startswith("\\item"):
            if open_item:
                out.append(current)
            current = stripped[len("\\item"):]
            open_item = True
            continue
        if not open_item:
            continue
        if stripped.startswith(("\\end{itemize}", "\\end{enumerate}", "\\end{description}",
                                _END_FRAME)):
            out.append(current)
            current = ""
            open_item = False
            continue
        current = f"{current} {stripped}"
    if open_item:
        out.append(current)
    return out


def _tabulars(frame: Frame) -> List[Tuple[str, ...]]:
    """The body rows of every tabular of a frame: the lines ending ``\\\\`` between the first
    ``\\midrule`` and the ``\\bottomrule``. A row spread over several source lines counts once,
    by its last line."""
    out: List[Tuple[str, ...]] = []
    rows: List[str] = []
    inside = False
    body = False
    for line in frame.body:
        if "\\begin{tabular}" in line:
            inside, body, rows = True, False, []
            continue
        if not inside:
            continue
        if "\\end{tabular}" in line:
            out.append(tuple(rows))
            inside, body, rows = False, False, []
            continue
        if "\\bottomrule" in line:
            body = False
            continue
        if not body:
            if "\\midrule" in line:
                body = True
            continue
        if _ROW_END.search(line):
            rows.append(line)
    if inside:
        out.append(tuple(rows))
    return out


def _cells(row: str) -> List[str]:
    """The cells of a tabular row, split on the unescaped ``&``."""
    return [cell.strip() for cell in re.split(r"(?<!\\)&", _ROW_END.sub("", row).strip())]


# ---------------------------------------------------------------------------
# LaTeX source to the text a reader sees
# ---------------------------------------------------------------------------


def _plain(raw: str) -> str:
    """The text of a LaTeX fragment: math removed, markup resolved, escapes undone."""
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


def _words(text: str, minimum: int = 3) -> List[str]:
    """The runs of letters and digits of at least ``minimum`` characters, in order."""
    return [word for word in _WORD.findall(text) if len(word) >= minimum]


def _normalize(text: str) -> str:
    """The comparable form of a text: ligatures mapped back, every run of characters that is
    not a letter or a digit collapsed to one space, so that punctuation, dashes and the
    underscore of a name compare alike on the source side and on the page side."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    return " ".join(re.sub(r"[^A-Za-z0-9]+", " ", text).split())


def _page_variants(text: str) -> Tuple[str, str, str]:
    """Three readings of one page of text, normalized: as laid out, with the end-of-line
    hyphens of TeX's hyphenation removed, and with every space removed (a name the url
    package broke over two lines of a narrow column)."""
    collapsed = _normalize(text)
    dehyphenated = _normalize(re.sub(r"-\s*\n\s*", "", text))
    squashed = collapsed.replace(" ", "")
    return collapsed, dehyphenated, squashed


def _present(needle: str, variants: Tuple[str, str, str]) -> bool:
    """Whether ``needle`` occurs, as a phrase, in any reading of the page."""
    collapsed, dehyphenated, squashed = variants
    probe = _normalize(needle)
    if not probe:
        return True
    if probe in collapsed or probe in dehyphenated:
        return True
    return probe.replace(" ", "") in squashed


_MATH_MARK = "MATHSENTINEL"


def _tail(raw: str, count: int = 4) -> str:
    """The last ``count`` tokens of the last text run of a LaTeX fragment, as one phrase: a
    cut that loses a short number or a single word at the end of an item breaks the phrase
    where the tokens taken one by one would still be found elsewhere on the page. Math is a
    boundary, not text: the page carries the rendered formula between the words, so the
    phrase is taken from the words after the last formula (or, when nothing follows it, from
    the run before it)."""
    text = raw
    for pattern in (_DISPLAY_MATH, _INLINE_MATH, _DOLLAR_MATH):
        text = pattern.sub(f" {_MATH_MARK} ", text)
    runs = [run.split() for run in _normalize(_plain(text)).split(_MATH_MARK)]
    runs = [run for run in runs if run]
    return " ".join(runs[-1][-count:]) if runs else ""


# ---------------------------------------------------------------------------
# the rules, as functions over parsed frames (exercised on a synthetic deck below)
# ---------------------------------------------------------------------------


def _layout_problems(frames: Sequence[Frame], main: Sequence[Frame]) -> List[str]:
    """Every breach of the main-frame rules, one line each."""
    problems: List[str] = []
    for frame in frames:
        if "allowframebreaks" in frame.options:
            problems.append(
                f"frame {frame.number} ({frame.title!r}): allowframebreaks, so the frame may "
                f"take more than one page and page N is no longer the N-th frame")
    for frame in main:
        where = f"frame {frame.number} ({frame.title!r})"
        items = _items(frame)
        if len(items) > _MAX_ITEMS:
            problems.append(f"{where}: {len(items)} items, at most {_MAX_ITEMS} on a main frame")
        if any("\\tiny" in line for line in frame.body):
            problems.append(f"{where}: \\tiny is not used on a main frame")
        for number, rows in enumerate(_tabulars(frame), start=1):
            if len(rows) > _MAX_BODY_ROWS:
                problems.append(
                    f"{where}: tabular {number} has {len(rows)} body rows, at most "
                    f"{_MAX_BODY_ROWS} on a main frame")
    return problems


def _link_problems(text: str, frames: Sequence[Frame], main: Sequence[Frame],
                   backup: Sequence[Frame]) -> List[str]:
    """Every dangling ``\\hyperlink`` and every backup frame without a return link."""
    problems: List[str] = []
    labels = {}
    for frame in frames:
        for label in _LABEL.findall(frame.options):
            labels[label.strip()] = frame.number
    for target in _HYPERLINK.findall(text):
        if target.strip() not in labels:
            problems.append(
                f"\\hyperlink{{{target}}} names no frame: the labels are {sorted(labels)}")
    main_labels = {label for label, number in labels.items()
                   if number in {frame.number for frame in main}}
    for frame in backup:
        targets = {t.strip() for line in frame.body for t in _HYPERLINK.findall(line)}
        if not targets & main_labels:
            problems.append(
                f"backup frame {frame.number} ({frame.title!r}) carries no return link to a "
                f"main frame (its links: {sorted(targets)})")
    return problems


# ---------------------------------------------------------------------------
# 1-2: the static rules
# ---------------------------------------------------------------------------


def test_frame_and_backup_counts():
    """The file holds 45 frames, the last 8 of them after the backup section."""
    lines = _lines()
    frames = _frames(lines)
    main, backup = _split_main_backup(lines, frames)
    assert len(frames) == _FRAME_COUNT, (
        f"{len(frames)} frames in {_FRAMES.name}, {_FRAME_COUNT} expected")
    assert len(backup) == _BACKUP_COUNT, (
        f"{len(backup)} frames after {_BACKUP_SECTION}, {_BACKUP_COUNT} expected "
        f"({len(main)} main frames)")


def test_main_frames_obey_the_layout_rules():
    """No main frame carries more than five items, a \\tiny, or a tabular over eight rows."""
    lines = _lines()
    frames = _frames(lines)
    main, _ = _split_main_backup(lines, frames)
    problems = _layout_problems(frames, main)
    assert problems == [], "\n".join(problems)


def test_hyperlinks_resolve_and_backups_return():
    """Every hyperlink names a labelled frame; every backup frame links back to a main frame."""
    lines = _lines()
    frames = _frames(lines)
    main, backup = _split_main_backup(lines, frames)
    problems = _link_problems("\n".join(lines), frames, main, backup)
    assert problems == [], "\n".join(problems)


def test_layout_rules_report_each_breach():
    """Each rule fires on a synthetic deck built to break it, one breach per frame."""
    deck = "\n".join([
        "\\begin{frame}[label=f1]{keeps the rules}",
        "  \\begin{itemize}",
        "    \\item one",
        "    \\item two",
        "  \\end{itemize}",
        "  \\hyperlink{b1}{full table}",
        "\\end{frame}",
        "\\begin{frame}{six items}",
        "  \\begin{itemize}",
        "    \\item one",
        "    \\item two",
        "    \\item three",
        "    \\item four",
        "    \\item five",
        "    \\item six",
        "  \\end{itemize}",
        "\\end{frame}",
        "\\begin{frame}{a tiny frame}",
        "  \\tiny",
        "  text",
        "\\end{frame}",
        "\\begin{frame}{a nine-row table}",
        "  \\begin{tabular}{ll}",
        "  \\toprule",
        "  head & head \\\\ \\midrule",
        ] + [f"  row{n} & {n} \\\\" for n in range(1, 10)] + [
        "  \\bottomrule",
        "  \\end{tabular}",
        "\\end{frame}",
        "\\begin{frame}[allowframebreaks]{a breaking frame}",
        "  text",
        "\\end{frame}",
        "\\begin{frame}{a dangling link}",
        "  \\hyperlink{nowhere}{full table}",
        "\\end{frame}",
        "\\section*{Backup}",
        "\\begin{frame}[label=b1]{a backup with a return}",
        "  \\hyperlink{f1}{back}",
        "\\end{frame}",
        "\\begin{frame}[label=b2]{a backup without a return}",
        "  text",
        "\\end{frame}",
    ])
    lines = deck.splitlines()
    frames = _frames(lines)
    main, backup = _split_main_backup(lines, frames)
    assert [frame.number for frame in main] == [1, 2, 3, 4, 5, 6]
    assert [frame.number for frame in backup] == [7, 8]
    assert frames[0].title == "keeps the rules"
    assert len(_items(frames[1])) == 6
    assert len(_tabulars(frames[3])[0]) == 9

    layout = _layout_problems(frames, main)
    assert len(layout) == 4, layout
    assert "6 items" in layout[1]
    assert "\\tiny" in layout[2]
    assert "9 body rows" in layout[3]
    assert "allowframebreaks" in layout[0]

    links = _link_problems(deck, frames, main, backup)
    assert len(links) == 2, links
    assert "nowhere" in links[0]
    assert "backup frame 8" in links[1]

    # the backup section moved past a backup frame: it counts as main and breaks a rule there
    moved = deck.replace("\\section*{Backup}\n\\begin{frame}[label=b1]",
                         "\\begin{frame}[label=b1]")
    moved_lines = moved.splitlines()
    moved_frames = _frames(moved_lines)
    moved_main, moved_backup = _split_main_backup(moved_lines, moved_frames)
    assert len(moved_main) == 8 and moved_backup == []


def test_item_and_cell_text_stripping():
    """The stripping rules on the constructs the deck uses."""
    # "py" and "39" are shorter than three characters and drop out of the word list
    assert _words(_plain(" The $10^{-5}$ in $x_0$ is the constant [dpyscf net.py:39].")) == [
        "The", "the", "constant", "dpyscf", "net"]
    assert " ".join(_plain("\\textbf{bold, negative} marks").split()) == "bold, negative marks"
    assert " ".join(_plain("\\path{run_20260908T153908Z}").split()) == "run_20260908T153908Z"
    assert " ".join(_plain("deep\\_3x16 [25 cycles]").split()) == "deep_3x16 [25 cycles]"
    assert " ".join(_plain("\\includegraphics[width=\\linewidth]{a/b.png}").split()) == ""
    assert " ".join(_plain("\\quad [PW92] J. P. Perdew").split()) == "[PW92] J. P. Perdew"
    assert _plain("$\\Delta$ED = ED NN $-$ ED PBE:").split() == ["ED", "=", "ED", "NN", "ED",
                                                                 "PBE:"]
    row = "  deep\\_3x16 [25 cycles] & 12 & 7.17 & \\textbf{-0.99} &  & \\\\"
    cells = [" ".join(_plain(cell).split()) for cell in _cells(row)]
    assert cells == ["deep_3x16 [25 cycles]", "12", "7.17", "-0.99", "", ""]

    page = _page_variants("a line ending in gener-\nalization and run_\n20260908T153908Z here")
    assert _present("generalization", page)
    assert _present("run_20260908T153908Z", page)
    assert not _present("nowhere", page)
    # the tail is a phrase of the last four tokens, compared without punctuation: a cut that
    # drops the closing "12." breaks it although every remaining word is on the page
    assert _tail("both arms at 7 and 12.") == "at 7 and 12"
    assert _tail("(Sec. 5). \\hfill \\hyperlink{b1}{\\beamergotobutton{run table}}") == "Sec 5"
    # math is a boundary: the page shows the formula between the words
    assert _tail("the zero-init clones within $4.2\\times10^{-3}$ everywhere.") == "everywhere"
    assert _tail("the vacuum tail ($\\rho < 9$) for $\\rho^{1/3}$.") == "for"
    assert _tail("a plateau schedule at $10^{-4}$ that never triggered.") == "that never triggered"
    cut = _page_variants("  - the arms: both arms at 7 and\n  - next item, 12 in the figures")
    assert not _present("at 7 and 12", cut)
    whole = _page_variants("  - the arms: both arms at 7 and 12.\n")
    assert _present("at 7 and 12", whole)
    assert _present("deep_3x16 [25 cycles]", _page_variants("deep_3x16 [25 cycles] & 12"))


# ---------------------------------------------------------------------------
# 3-4: the built deck
# ---------------------------------------------------------------------------


def _pdf_pages(pdf: Path) -> int:
    """The page count of a PDF, read by pdfinfo: xelatex writes object streams, so the page
    objects are not visible as plain bytes."""
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    match = re.search(r"^Pages:\s+(\d+)$", out, re.M)
    assert match is not None, out
    return int(match.group(1))


def _page_text(pdf: Path, page: int) -> str:
    """The text of one page, in the layout the page has."""
    out = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True, check=True).stdout
    return out.decode("utf-8", errors="replace")


@pytest.mark.skipif(shutil.which("xelatex") is None or shutil.which("pdfinfo") is None
                    or shutil.which("pdftotext") is None,
                    reason="xelatex, pdfinfo or pdftotext not installed")
def test_plain_deck_shows_every_item_and_table_row(tmp_path):
    """The plain deck has one page per frame and every page carries its frame's text.

    An overfull frame is not an error to beamer: the material that no longer fits is dropped
    from the page without a word in the log. The end of every item and the outer cells of every
    tabular's last body row are looked up in the page the frame produced, which is where a cut
    shows first.
    """
    lines = _lines()
    frames = _frames(lines)
    out = tmp_path / _PLAIN.stem
    out.mkdir()
    proc = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={out}", _PLAIN.name],
        cwd=_HERE, capture_output=True, text=True, timeout=900)
    log = (out / f"{_PLAIN.stem}.log").read_text(encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"{_PLAIN.name}: xelatex failed\n{log[-3000:]}"
    pdf = out / f"{_PLAIN.stem}.pdf"
    # beamer drops nothing for an overfull box but warns in the log: a vbox too high puts the
    # last line on the footer with no margin, an hbox too wide runs text past the right edge;
    # neither is visible to the text lookup below, so the log is read as well
    overfull = re.findall(r"Overfull \\[hv]box \([^)]*\)[^\n]*", log)
    assert overfull == [], f"{_PLAIN.name}: overfull boxes (text past the frame edge): {overfull}"
    pages = _pdf_pages(pdf)
    assert pages == len(frames), (
        f"{pages} pages from {len(frames)} frames: page N is no longer the N-th frame")

    problems: List[str] = []
    for frame in frames:
        variants = _page_variants(_page_text(pdf, frame.number))
        where = f"frame {frame.number} ({frame.title!r})"
        for number, raw in enumerate(_items(frame), start=1):
            tail = _tail(raw)
            if tail and not _present(tail, variants):
                problems.append(f"{where}: item {number} ends in {tail!r}, not on the page")
        for number, rows in enumerate(_tabulars(frame), start=1):
            if not rows:
                continue
            texts = [" ".join(_plain(cell).split()) for cell in _cells(rows[-1])]
            filled = [text for text in texts if _words(text, 1)]
            if not filled:
                continue
            # the first cell may sit in a p{} column and wrap over two lines with the other
            # columns' text between its halves, so its words are looked up one by one; the
            # last filled cell (a number or a short name at the row's end) is one phrase
            first_missing = [word for word in _normalize(texts[0]).split()
                             if not _present(word, variants)]
            if first_missing:
                problems.append(
                    f"{where}: tabular {number}, the first cell of the last body row "
                    f"({texts[0]!r}) is not on the page (missing {first_missing})")
            if _words(filled[-1], 1) and not _present(filled[-1], variants):
                problems.append(
                    f"{where}: tabular {number}, the last cell of the last body row "
                    f"({filled[-1]!r}) is not on the page")
    assert problems == [], "\n".join(problems)

"""The documentation site carries what the repository knows, and says nothing that is false.

Five rules:

* every entry of the site's table of contents resolves to a tracked page;
* every tracked page under ``docs/`` is reachable from that table of contents, so a document
  cannot be written and then left where no reader finds it;
* the citation file is readable and names the work;
* the README names what the repository holds, and every repository-relative link in it
  resolves;
* the documentation requirements pin nothing the packaging file contradicts.

The rules read the files; the build itself (``sphinx-build -W``) runs in the environment the
packaging file's documentation extra describes, and its result is recorded with the change.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"
_INDEX = _DOCS / "index.rst"
_README = _ROOT / "README.md"
_CITATION = _ROOT / "CITATION.cff"
_READTHEDOCS = _ROOT / ".readthedocs.yml"

#: pages the table of contents does not carry and does not have to: the build instructions,
#: which belong to whoever builds the site, and the two placeholder pages of the static and
#: template directories
UNLISTED = (
    "docs/README.md",
    "docs/_static/README.md",
    "docs/_templates/README.md",
)

#: what the README must name, so that a reader arriving at the repository finds the work
README_SUBJECTS = (
    "pip install",
    "xcquinox-cluster",
    "docs/getting_started.md",
    "CITATION.cff",
)

#: the keys the citation format requires, plus the ones a reader needs
CITATION_KEYS = ("cff-version", "message", "title", "authors", "license")


def _tracked(*paths: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", *paths], cwd=_ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p]


def toctree_entries(text: str) -> list[str]:
    """The document names of every ``toctree`` block of a reStructuredText page, in order."""
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".. toctree::"):
            in_block = True
            continue
        if in_block:
            if not stripped or stripped.startswith(":"):
                continue
            if not line.startswith((" ", "\t")) or stripped.startswith(".."):
                in_block = False
                continue
            out.append(stripped.lstrip("./"))
    return out


def listed_documents(pages) -> set[str]:
    """Every document a table of contents carries, as a path under ``docs/`` without its
    extension. ``pages`` is ``{path: text}``; an entry is relative to the page that lists
    it, so a nested page's entries resolve from its own directory."""
    out = set()
    for path, text in pages.items():
        parent = str(pathlib.PurePosixPath(path).parent)
        for entry in toctree_entries(text):
            name = entry.rsplit("#", 1)[0]
            if name.startswith("http"):
                continue
            joined = name if parent in ("docs", ".") else f"{parent[len('docs/'):]}/{name}"
            out.add(str(pathlib.PurePosixPath(joined)))
    return out


def unresolved_entries(listed, tracked) -> list[str]:
    """The listed documents that name no tracked page (a name carries no extension)."""
    names = {p[len("docs/"):].rsplit(".", 1)[0] for p in tracked
             if p.endswith((".rst", ".md"))}
    return sorted(name for name in listed if name not in names)


def unreachable_pages(listed, tracked, included) -> list[str]:
    """The tracked pages under docs/ that no table of contents carries, that nothing includes
    and that the exemption does not name."""
    out = []
    for path in tracked:
        if not path.endswith((".rst", ".md")) or path in UNLISTED:
            continue
        name = path[len("docs/"):].rsplit(".", 1)[0]
        if name == "index" or name in listed or path in included:
            continue
        out.append(path)
    return out


#: the two spellings of an include: the reStructuredText directive and the markdown one the
#: site writes (```{include} path)
_INCLUDE = re.compile(r"(?:\.\.\s+include::|\{include\})\s*([^\s`}]+)")


def included_targets(path: str, text: str) -> set[str]:
    """The repository-relative paths ``text`` pulls in, resolved from ``path``'s directory."""
    out = set()
    for match in _INCLUDE.finditer(text):
        target = (_ROOT / path).parent / match.group(1)
        try:
            out.add(str(target.resolve().relative_to(_ROOT)))
        except ValueError:
            continue
    return out


def _included_pages() -> set[str]:
    """Every page another page of the site pulls in."""
    out = set()
    for path in _tracked("docs"):
        if path.endswith((".md", ".rst")):
            out |= included_targets(path, (_ROOT / path).read_text(encoding="utf-8"))
    return out


def readme_links(text: str) -> list[str]:
    """The repository-relative link targets of a markdown page."""
    out = []
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = match.group(1).split("#")[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        out.append(target.lstrip("./"))
    return out


def test_the_rules_fire_on_fixtures():
    """Each rule fires on a page that breaks it and passes one that does not."""
    page = (".. toctree::\n   :maxdepth: 2\n\n   setup\n   ./notes/primer\n\n"
            "Some prose.\n\n.. toctree::\n\n   api\n")
    assert toctree_entries(page) == ["setup", "notes/primer", "api"]
    tracked = ["docs/index.rst", "docs/setup.md", "docs/notes/primer.md",
               "docs/api.rst", "docs/docpages/net.rst", "docs/orphan.md"]
    listed = listed_documents({"docs/index.rst": page,
                               "docs/api.rst": ".. toctree::\n\n   docpages/net\n"})
    assert listed == {"setup", "notes/primer", "api", "docpages/net"}
    assert unresolved_entries(listed, tracked) == []
    assert unresolved_entries({"gone"}, tracked) == ["gone"]
    assert unreachable_pages(listed, tracked, set()) == ["docs/orphan.md"]
    assert unreachable_pages(listed, tracked, {"docs/orphan.md"}) == []
    nested = listed_documents({"docs/notes/index.md": ".. toctree::\n\n   deeper\n"})
    assert nested == {"notes/deeper"}
    assert included_targets("docs/x.md", "```{include} ../README.md\n```") == {
        "README.md"}
    assert included_targets("docs/notes/a.md", ".. include:: ../items.md") == {
        "docs/items.md"}
    assert included_targets("docs/x.md", "no include here") == set()
    assert readme_links("[a](docs/guide.md) [b](https://x) [c](./CITATION.cff)") == [
        "docs/guide.md", "CITATION.cff"]


def _site_pages() -> dict[str, str]:
    """``{path: text}`` for every tracked page of the site."""
    return {path: (_ROOT / path).read_text(encoding="utf-8")
            for path in _tracked("docs") if path.endswith((".rst", ".md"))}


def test_every_toctree_entry_resolves():
    """Every entry of every table of contents names a tracked page."""
    listed = listed_documents(_site_pages())
    missing = unresolved_entries(listed, _tracked("docs"))
    assert missing == [], f"entries that name no page: {missing}"
    assert listed, "no table of contents lists a page"


def test_every_document_is_reachable_from_the_table_of_contents():
    """No page under docs/ sits where no reader finds it."""
    tracked = _tracked("docs")
    orphans = unreachable_pages(listed_documents(_site_pages()), tracked, _included_pages())
    assert orphans == [], f"pages no table of contents carries: {orphans}"


def test_the_citation_file_is_readable_and_names_the_work():
    """The citation file parses and carries the keys a citation needs."""
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(_CITATION.read_text(encoding="utf-8"))
    missing = [key for key in CITATION_KEYS if key not in data]
    assert missing == [], f"the citation file lacks {missing}"
    assert data["authors"] and data["authors"][0].get("family-names"), data["authors"]
    assert str(data["cff-version"]).startswith("1."), data["cff-version"]


def test_the_readme_names_what_the_repository_holds():
    """The README names the install, the harness, the guide and the citation, and every
    repository-relative link in it resolves."""
    text = _README.read_text(encoding="utf-8")
    missing = [subject for subject in README_SUBJECTS if subject not in text]
    assert missing == [], f"the README does not name {missing}"
    broken = [target for target in readme_links(text) if not (_ROOT / target).exists()]
    assert broken == [], f"links that resolve to nothing: {broken}"


def test_the_published_build_installs_the_package_with_its_documentation_extra():
    """The published site is built the way this commit verifies it: the package and its
    documentation extra, with warnings as errors. A second dependency list for the site is
    what rotted before (it pinned a stack the packaging file had already corrected), so
    there is one."""
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(_READTHEDOCS.read_text(encoding="utf-8"))
    assert config["sphinx"]["fail_on_warning"] is True, config["sphinx"]
    installs = config["python"]["install"]
    assert any(entry.get("path") == "." and "docs" in entry.get("extra_requirements", [])
               for entry in installs), installs
    assert not (_DOCS / "requirements.txt").exists(), (
        "a second dependency list for the site is back")

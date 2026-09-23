# Building the documentation

The site is built with [Sphinx](https://www.sphinx-doc.org/). The tools it needs are the
packaging file's documentation extra, so the whole build is two commands from the repository
root:

```bash
pip install ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

`-W` turns warnings into errors. That is deliberate: a page left out of the table of
contents, a reference that does not resolve or a broken directive fails the build instead of
producing a site with a hole in it. `tools/test_docs.py` holds the same rules without a
build, so a missing page is caught by the test suite too.

The built site lands in `docs/_build/html`; open `index.html` from there. The published copy
is at <https://xcquinox.readthedocs.io/en/latest/>, built from `.readthedocs.yml`, which
installs the package with the same extra.

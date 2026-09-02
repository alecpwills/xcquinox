"""Comment/docstring-stripped source scanning for structural test pins.

A pin that greps RAW module source is satisfiable by its own documentation:
a comment quoting the forbidden (or required) construct matches the scan,
so the test passes whether or not the code changed -- the vacuity class the
2026-09-01 test-suite review found in 18 source-pin tests, with
``test_metagga.test_compute_alpha_has_no_stop_gradient_on_the_energy_path``
as the corrected model. Every structural pin scans code-only text through
:func:`code_only` instead.
"""
from __future__ import annotations

import ast
import inspect
import io
import tokenize


def code_only(source_or_obj) -> str:
    """The CODE of ``source_or_obj`` with comments and docstrings removed.

    Accepts a source string, or any object ``inspect.getsource`` resolves
    (module, class, function). Comments are dropped by tokenization;
    docstrings by AST position (every string-expression statement). String
    LITERALS that are part of expressions (error messages, dict keys)
    survive -- they are code. Falls back to a plain comment strip when the
    text does not tokenize as standalone Python (e.g. an indented method
    body from ``getsource``)."""
    src = (source_or_obj if isinstance(source_or_obj, str)
           else inspect.getsource(source_or_obj))
    try:
        tree = ast.parse(src)
    except (SyntaxError, IndentationError):
        try:
            import textwrap
            src = textwrap.dedent(src)
            tree = ast.parse(src)
        except (SyntaxError, IndentationError):
            return "\n".join(line.split("#", 1)[0]
                             for line in src.splitlines())
    doc_lines: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    body[0].value, ast.Constant) and isinstance(
                    body[0].value.value, str):
                doc_lines.update(range(body[0].lineno,
                                       body[0].end_lineno + 1))
    out_lines = []
    for i, line in enumerate(src.splitlines(), 1):
        out_lines.append("" if i in doc_lines else line)
    stripped = "\n".join(out_lines)
    # Drop comments via tokenize (handles '#' inside strings correctly).
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(stripped).readline))
    except (tokenize.TokenError, IndentationError):
        return "\n".join(line.split("#", 1)[0]
                         for line in stripped.splitlines())
    keep = [t for t in toks if t.type != tokenize.COMMENT]
    return tokenize.untokenize(keep)

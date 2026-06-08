from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import thrombxai

ROOT = Path(thrombxai.__file__).resolve().parent


def _sources() -> list[Path]:
    return sorted(ROOT.rglob("*.py"))


def test_sources_have_no_comments() -> None:
    for path in _sources():
        source = path.read_text()
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            assert token.type != tokenize.COMMENT, f"{path.name}:{token.start[0]}"


def test_sources_have_no_docstrings() -> None:
    targets = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for path in _sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, targets):
                assert ast.get_docstring(node) is None, path.name

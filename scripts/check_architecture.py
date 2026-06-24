#!/usr/bin/env python3
"""Architecture health check for the aurelius codebase.

Counts lines of code, overlong functions (>50 lines as a proxy for
cyclomatic complexity), third-party dependencies, and public architectural
surface area. Exits with code 0 on PASS, 1 on FAIL.

Usage:
    python scripts/check_architecture.py
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Final

SRC_DIR: Final[str] = os.path.join(
    os.path.dirname(__file__), "..", "engine", "src", "aurelius"
)

# Normalisation thresholds (mirror NET_PROGRESS_* constants from constants.py)
LOC_NORM: Final[float] = 10000.0
OVERLONG_NORM: Final[float] = 50.0
DEP_NORM: Final[float] = 15.0
ARCH_NORM: Final[float] = 250.0

EXCLUDED_FILES: Final[frozenset[str]] = frozenset({
    "chem_utils.py", "dependencies.py", "__init__.py", "__main__.py", "reporting.py",
})


def count_lines_of_code() -> int:
    """Count total non-empty, non-comment lines in src/aurelius/.py files."""
    total = 0
    for root, _dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            filepath = os.path.join(root, fn)
            with open(filepath) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        total += 1
    return total


def count_overlong_functions() -> int:
    """Count functions/classes whose body exceeds 50 lines of code.

    A simpler alternative to radon-based cyclomatic complexity — long
    functions are a reliable proxy for high complexity and are trivial
    to compute via the ``ast`` module.
    """
    count = 0
    for root, _dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith(".py") or fn in EXCLUDED_FILES:
                continue
            filepath = os.path.join(root, fn)
            with open(filepath) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                        n_lines = node.end_lineno - node.lineno
                        if n_lines > 50:
                            count += 1
    return count


def count_dependency_imports() -> int:
    """Count unique third-party imports in src/aurelius/ (excluding aurelius itself)."""
    stdlib: Final[frozenset[str]] = frozenset({
        "os", "sys", "json", "math", "re", "time", "io", "abc", "typing",
        "collections", "functools", "itertools", "pathlib", "copy", "inspect",
        "logging", "contextlib", "subprocess", "tempfile", "threading",
        "concurrent", "dataclasses", "warnings", "pickle", "enum", "hashlib",
        "textwrap", "bisect", "random", "__future__", "atexit", "datetime",
        "importlib", "shutil",
    })
    deps: set[str] = set()
    for root, _dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            filepath = os.path.join(root, fn)
            with open(filepath) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        pkg = alias.name.split(".")[0]
                        if pkg not in stdlib and not pkg.startswith("aurelius"):
                            deps.add(pkg)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    pkg = node.module.split(".")[0]
                    if pkg not in stdlib and not pkg.startswith("aurelius"):
                        deps.add(pkg)
    return len(deps)


def count_architectural_surface_area() -> int:
    """Count public classes and functions in core src/aurelius/ modules.

    Tracks the number of public (non-underscore-prefixed) classes and
    top-level functions as a proxy for architectural complexity.
    """
    count = 0
    for root, _dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith(".py") or fn in EXCLUDED_FILES:
                continue
            filepath = os.path.join(root, fn)
            with open(filepath) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
                    count += 1
    return count


def main() -> int:
    loc = count_lines_of_code()
    overlong = count_overlong_functions()
    n_deps = count_dependency_imports()
    arch_surface = count_architectural_surface_area()

    print("=" * 55)
    print("  ARCHITECTURE HEALTH CHECK")
    print("=" * 55)
    print(f"  Lines of code:               {loc}")
    print(f"  Overlong functions (>50):    {overlong}")
    print(f"  Third-party deps:            {n_deps}")
    print(f"  Architectural surface area:  {arch_surface}")
    print("=" * 55)

    failures: list[str] = []
    if loc > LOC_NORM:
        failures.append(f"Lines of code {loc} exceeds limit {LOC_NORM}")
    if overlong > OVERLONG_NORM:
        failures.append(f"Overlong functions {overlong} exceeds limit {OVERLONG_NORM}")
    if n_deps > DEP_NORM:
        failures.append(f"Dependencies {n_deps} exceeds limit {DEP_NORM}")
    if arch_surface > ARCH_NORM:
        failures.append(f"Architectural surface area {arch_surface} exceeds limit {ARCH_NORM}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

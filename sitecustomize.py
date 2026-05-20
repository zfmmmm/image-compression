"""Project-local import compatibility.

Python imports the standard-library ``codecs`` module during interpreter
startup.  This project intentionally has a ``codecs/`` directory because the
assignment requires that layout.  Adding the local directory to the stdlib
module's submodule search path lets imports such as ``from codecs.base`` work
while preserving all normal stdlib ``codecs`` attributes.
"""

from __future__ import annotations

from pathlib import Path
import codecs as _stdlib_codecs


_PROJECT_CODECS_DIR = Path(__file__).resolve().parent / "codecs"

if _PROJECT_CODECS_DIR.is_dir():
    existing = list(getattr(_stdlib_codecs, "__path__", []))
    project_path = str(_PROJECT_CODECS_DIR)
    if project_path not in existing:
        _stdlib_codecs.__path__ = [project_path, *existing]

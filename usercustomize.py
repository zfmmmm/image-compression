"""Expose project ``codecs/`` submodules when Python loads user customizations."""

from __future__ import annotations

from pathlib import Path
import codecs as _stdlib_codecs


_PROJECT_CODECS_DIR = Path(__file__).resolve().parent / "codecs"

if _PROJECT_CODECS_DIR.is_dir():
    existing = list(getattr(_stdlib_codecs, "__path__", []))
    project_path = str(_PROJECT_CODECS_DIR)
    if project_path not in existing:
        _stdlib_codecs.__path__ = [project_path, *existing]

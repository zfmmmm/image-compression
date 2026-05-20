from __future__ import annotations

from pathlib import Path
import codecs as _stdlib_codecs


def ensure_project_codecs_importable() -> None:
    """Expose this project's ``codecs/`` directory as stdlib-codecs submodules."""

    project_root = Path(__file__).resolve().parents[1]
    codecs_dir = project_root / "codecs"
    existing = list(getattr(_stdlib_codecs, "__path__", []))
    path_text = str(codecs_dir)
    if codecs_dir.is_dir() and path_text not in existing:
        _stdlib_codecs.__path__ = [path_text, *existing]

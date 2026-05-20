from __future__ import annotations

from pathlib import Path
import shutil

from utils.codec_imports import ensure_project_codecs_importable


ensure_project_codecs_importable()

from codecs.base import CodecResult  # noqa: E402


def copy_best_bitstream(result: CodecResult, output_dir: Path, filename_stem: str) -> Path:
    if not result.bitstream_path.exists():
        raise FileNotFoundError(f"Best bitstream does not exist: {result.bitstream_path}")

    suffix = result.bitstream_path.suffix or ".bin"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename_stem}{suffix}"
    shutil.copy2(result.bitstream_path, output_path)
    return output_path

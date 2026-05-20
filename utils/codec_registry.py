from __future__ import annotations

from collections.abc import Iterable
import logging

from utils.codec_imports import ensure_project_codecs_importable


ensure_project_codecs_importable()

from codecs.avif_codec import AVIFCodec  # noqa: E402
from codecs.bpg_codec import BPGCodec  # noqa: E402
from codecs.jpeg2000_codec import JPEG2000Codec  # noqa: E402
from codecs.jpeg_codec import JPEGCodec  # noqa: E402
from codecs.jxl_codec import JXLCodec  # noqa: E402


LOGGER = logging.getLogger(__name__)

CODEC_CLASSES = {
    "jpeg": JPEGCodec,
    "jpeg2000": JPEG2000Codec,
    "jp2": JPEG2000Codec,
    "jxl": JXLCodec,
    "jpegxl": JXLCodec,
    "avif": AVIFCodec,
    "bpg": BPGCodec,
}


def build_codecs(names: Iterable[str], *, timeout: float = 600.0):
    codecs = []
    for raw_name in names:
        name = raw_name.strip().lower()
        if not name:
            continue
        codec_class = CODEC_CLASSES.get(name)
        if codec_class is None:
            LOGGER.warning("Unknown codec '%s'; skipping", raw_name)
            continue
        codecs.append(codec_class(timeout=timeout))
    return codecs

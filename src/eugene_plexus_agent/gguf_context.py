"""The one number a profile build needs from a GGUF header: its trained context.

The library parses GGUF fully and is asked first. This is the fallback for
a node that cannot reach it, so a build never tries a context the model was
not trained for. Duplicated rather than shared, by the polyrepo rule; it
reads only the key-value header, stops at the answer, and never loads
tensor data.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

MAGIC = b"GGUF"
# Fixed-size value types by GGUF type id.
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9
_UINT32, _INT32, _UINT64, _INT64 = 4, 5, 10, 11
# A header this long without the key means something is wrong with the file.
_MAX_KEYS = 100_000


def _u64(stream: BinaryIO) -> int:
    return int(struct.unpack("<Q", stream.read(8))[0])


def _string(stream: BinaryIO) -> bytes:
    return stream.read(_u64(stream))


def _skip(stream: BinaryIO, kind: int) -> None:
    if kind in _FIXED:
        stream.seek(_FIXED[kind], 1)
    elif kind == _STRING:
        stream.seek(_u64(stream), 1)
    elif kind == _ARRAY:
        inner = struct.unpack("<I", stream.read(4))[0]
        count = _u64(stream)
        if inner in _FIXED:
            stream.seek(_FIXED[inner] * count, 1)
        else:
            for _ in range(count):
                _skip(stream, inner)
    else:
        raise ValueError(f"unknown GGUF value type {kind}")


def trained_context(path: Path) -> int | None:
    """`<architecture>.context_length`, or None when it cannot be read."""
    try:
        with path.open("rb") as stream:
            if stream.read(4) != MAGIC:
                return None
            version = struct.unpack("<I", stream.read(4))[0]
            if version < 2:
                return None
            _u64(stream)  # tensor count
            keys = _u64(stream)
            for _ in range(min(keys, _MAX_KEYS)):
                key = _string(stream).decode("utf-8", errors="replace")
                kind = struct.unpack("<I", stream.read(4))[0]
                if key.endswith(".context_length") and kind in (_UINT32, _INT32, _UINT64, _INT64):
                    size = 4 if kind in (_UINT32, _INT32) else 8
                    fmt = {_UINT32: "<I", _INT32: "<i", _UINT64: "<Q", _INT64: "<q"}[kind]
                    value = struct.unpack(fmt, stream.read(size))[0]
                    return int(value) if value > 0 else None
                _skip(stream, kind)
    except (OSError, ValueError, struct.error):
        return None
    return None

"""Pure-Python 8-bit grayscale PNG codec (stdlib zlib only) — no OpenCV/PIL.

The from-scratch VIO records and replays single-channel 8-bit grayscale frames
(``_L.png`` / ``_R.png``, color type 0, bit depth 8 — what ``cv2.imwrite``
produced for a uint8 image). This module reads and writes exactly that format so
the record -> replay loop is library-free.

PNG is lossless, so :func:`imread_gray` returns pixel values byte-for-byte equal
to ``cv2.imread(path, IMREAD_UNCHANGED)`` on those files. Only 8-bit grayscale,
non-interlaced PNGs are supported; anything else raises ``ValueError`` so a
format surprise is loud rather than silent.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def imread_gray(path: str | Path) -> np.ndarray:
    """Decode an 8-bit grayscale PNG to an ``(H, W)`` uint8 array.

    Equivalent to ``cv2.imread(path, cv2.IMREAD_UNCHANGED)`` for the recorder's
    single-channel frames (exact pixel values — PNG is lossless).
    """
    data = Path(path).read_bytes()
    if data[:8] != _SIG:
        raise ValueError(f"not a PNG: {path}")
    pos = 8
    width = height = bit_depth = color_type = interlace = -1
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # 4 len + 4 type + body + 4 crc
        if ctype == b"IHDR":
            (width, height, bit_depth, color_type, _comp, _filt,
             interlace) = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
    if width < 0:
        raise ValueError(f"PNG missing IHDR: {path}")
    if bit_depth != 8 or color_type != 0:
        raise ValueError(
            f"unsupported PNG (bit_depth={bit_depth}, color_type={color_type}); "
            f"only 8-bit grayscale is supported: {path}")
    if interlace != 0:
        raise ValueError(f"interlaced PNG not supported: {path}")

    raw = zlib.decompress(bytes(idat))
    bpp = 1                      # bytes per pixel for 8-bit grayscale
    stride = width              # bytes per scanline (excl. filter byte)
    out = np.empty((height, width), dtype=np.uint8)
    prev = bytearray(stride)    # previous reconstructed scanline
    rp = 0
    for y in range(height):
        ftype = raw[rp]; rp += 1
        line = bytearray(raw[rp:rp + stride]); rp += stride
        if ftype == 0:
            pass
        elif ftype == 1:        # Sub
            for x in range(bpp, stride):
                line[x] = (line[x] + line[x - bpp]) & 0xFF
        elif ftype == 2:        # Up
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif ftype == 3:        # Average
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:        # Paeth
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                c = prev[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + _paeth(a, prev[x], c)) & 0xFF
        else:
            raise ValueError(f"bad PNG filter type {ftype}: {path}")
        out[y] = np.frombuffer(bytes(line), dtype=np.uint8)
        prev = line
    return out


def _chunk(ctype: bytes, body: bytes) -> bytes:
    return (struct.pack(">I", len(body)) + ctype + body
            + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF))


def imwrite_gray(path: str | Path, img: np.ndarray) -> None:
    """Encode an ``(H, W)`` uint8 array as an 8-bit grayscale PNG.

    Uses filter type 0 (None) per scanline — a valid PNG that
    :func:`imread_gray` (and any standard decoder) reads back exactly.
    """
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim != 2:
        raise ValueError(f"imwrite_gray expects a 2-D array, got {img.shape}")
    h, w = img.shape
    rows = bytearray()
    zero = b"\x00"
    for y in range(h):
        rows += zero            # filter byte (None)
        rows += img[y].tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    out = (_SIG
           + _chunk(b"IHDR", ihdr)
           + _chunk(b"IDAT", zlib.compress(bytes(rows), 6))
           + _chunk(b"IEND", b""))
    Path(path).write_bytes(out)

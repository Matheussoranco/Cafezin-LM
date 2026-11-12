"""Minimal safetensors reader/writer — torch + stdlib only.

Safetensors is deliberately simple:

    [8 bytes little-endian header length N][N bytes JSON header][tensor bytes]

The JSON header maps each tensor name to ``{"dtype", "shape", "data_offsets"}``,
where the offsets are relative to the start of the data section (immediately
after the header). The header is space-padded so the data section starts on an
8-byte boundary, which is what released checkpoints do.

Implementing this here keeps the dependency list at torch + numpy while still
reading and writing the exact files Hugging Face ships — including
``bfloat16``, which numpy has no dtype for (it is carried as raw ``uint16``).
`load_header` is the interesting one: it reads only the header, so the weight
*spec* of a multi-gigabyte checkpoint can be verified over HTTP range requests
without downloading a single weight.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO

import torch

# safetensors type string -> torch dtype
ST_TO_TORCH: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
TORCH_TO_ST: dict[torch.dtype, str] = {v: k for k, v in ST_TO_TORCH.items()}

_DTYPE_SIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

MAX_HEADER_BYTES = 100_000_000


def nbytes_of(dtype: str, shape: tuple[int, ...]) -> int:
    n = _DTYPE_SIZE[dtype]
    for dim in shape:
        n *= dim
    return n


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return memoryview(tensor.reshape(-1).numpy()).cast("B")


def build_header(tensors: dict[str, tuple[torch.Tensor, str]]) -> tuple[bytes, int]:
    """Serialise a header. Returns (padded header bytes, data section size)."""
    header: dict[str, dict] = {}
    offset = 0
    for name, (tensor, st_dtype) in tensors.items():
        shape = tuple(int(s) for s in tensor.shape)
        size = nbytes_of(st_dtype, shape)
        header[name] = {
            "dtype": st_dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(raw)) % 8
    return raw + b" " * pad, offset


def save_file(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> int:
    """Write a safetensors file. Returns the byte size written."""
    prepared: dict[str, tuple[torch.Tensor, str]] = {}
    for name, tensor in tensors.items():
        if tensor.dtype not in TORCH_TO_ST:
            raise ValueError(f"tensor {name!r} has unsupported dtype {tensor.dtype}")
        prepared[name] = (tensor.detach().cpu(), TORCH_TO_ST[tensor.dtype])

    header, data_size = build_header(prepared)
    if metadata:
        # re-emit with __metadata__ included; it must not offset the data section
        merged = json.loads(header.decode("utf-8"))
        merged["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
        raw = json.dumps(merged, separators=(",", ":")).encode("utf-8")
        pad = (-len(raw)) % 8
        header = raw + b" " * pad

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        for _, (tensor, _) in prepared.items():
            fh.write(_tensor_bytes(tensor))
    written = out.stat().st_size
    expected = 8 + len(header) + data_size
    if written != expected:
        raise OSError(f"{out}: wrote {written} bytes, expected {expected}")
    return written


def _read_header(fh: BinaryIO) -> dict:
    raw_len = fh.read(8)
    if len(raw_len) != 8:
        raise ValueError(f"{fh.name}: missing safetensors header")
    (length,) = struct.unpack("<Q", raw_len)
    if not 0 < length <= MAX_HEADER_BYTES:
        raise ValueError(f"{fh.name}: invalid header length {length}")
    raw = fh.read(length)
    if len(raw) != length:
        raise ValueError(f"{fh.name}: truncated safetensors header")
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header


def load_header(path: str | Path) -> dict:
    """Read only the JSON header of a safetensors file."""
    with Path(path).open("rb") as fh:
        return _read_header(fh)


def parse_header_bytes(header: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Header JSON -> {name: (dtype, shape)}."""
    return {name: (meta["dtype"], tuple(meta["shape"])) for name, meta in header.items()}


def load_file(
    path: str | Path,
    keys: list[str] | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Read selected tensors using offsets relative to the data section."""
    path = Path(path)
    out = {}
    with path.open("rb") as fh:
        header = _read_header(fh)
        data_start = fh.tell()
        data_size = fh.seek(0, 2) - data_start
        wanted = list(header) if keys is None else keys
        for name in wanted:
            meta = header[name]
            lo, hi = meta["data_offsets"]
            dtype = meta["dtype"]
            shape = tuple(meta["shape"])
            if (
                any(dim < 0 for dim in shape)
                or not 0 <= lo <= hi <= data_size
                or hi - lo != nbytes_of(dtype, shape)
            ):
                raise ValueError(f"{path}: invalid offsets or shape for {name}")
            if hi == lo:
                out[name] = torch.empty(shape, dtype=ST_TO_TORCH[dtype], device=device)
                continue
            fh.seek(data_start + lo)
            raw = bytearray(hi - lo)
            if fh.readinto(raw) != len(raw):
                raise ValueError(f"{path}: truncated tensor {name}")
            out[name] = torch.frombuffer(raw, dtype=ST_TO_TORCH[dtype]).reshape(shape).to(device)
    return out

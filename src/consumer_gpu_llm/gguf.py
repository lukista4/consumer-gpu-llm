"""Minimal, read-only GGUF parser plus a tiny writer used to build synthetic test files.

Only the parts needed by the tools in this repo are implemented: the header, the
key/value metadata, the tensor table, the data-section start and per-type byte sizes.
GGUF versions 2 and 3 are supported (64-bit counts).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterable

MAGIC = b"GGUF"

# ggml tensor type ids -> names (subset that appears in common GGUF files)
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16",
}
TYPE_IDS = {v: k for k, v in GGML_TYPES.items()}

# (elements per block, bytes per block)
BLOCK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 36),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176),
    "Q6_K": (256, 210), "Q8_K": (256, 292),
}

# GGUF metadata value types
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
T_STRING, T_ARRAY = 8, 9


def tensor_nbytes(type_name: str, n_elements: int) -> int:
    """Payload bytes of a tensor of `n_elements` stored as `type_name`."""
    per_block, size = BLOCK[type_name]
    if n_elements % per_block:
        raise ValueError(f"{n_elements} elements is not a multiple of the {type_name} block ({per_block})")
    return n_elements // per_block * size


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    type_id: int
    offset: int  # relative to the start of the data section

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, str(self.type_id))

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self.type_name, self.n_elements)


@dataclass
class GGUFInfo:
    version: int
    kv: dict[str, Any] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    alignment: int = 32
    data_start: int = 0

    def tensor(self, name: str) -> TensorInfo:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    def type_summary(self) -> dict[str, tuple[int, int]]:
        """type name -> (tensor count, element count)."""
        out: dict[str, list[int]] = {}
        for t in self.tensors:
            c = out.setdefault(t.type_name, [0, 0])
            c[0] += 1
            c[1] += t.n_elements
        return {k: (v[0], v[1]) for k, v in out.items()}


def _rd(f: BinaryIO, fmt: str):
    size = struct.calcsize("<" + fmt)
    buf = f.read(size)
    if len(buf) != size:
        raise EOFError("truncated GGUF file")
    return struct.unpack("<" + fmt, buf)


def _rstr(f: BinaryIO) -> str:
    (n,) = _rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def _rval(f: BinaryIO, t: int, max_array: int) -> Any:
    if t == T_STRING:
        return _rstr(f)
    if t == T_ARRAY:
        (et,) = _rd(f, "I")
        (n,) = _rd(f, "Q")
        vals = [_rval(f, et, max_array) for _ in range(n)]
        return vals if n <= max_array else f"<array {n}>"
    if t not in _SCALAR:
        raise ValueError(f"unknown GGUF value type {t}")
    return _rd(f, _SCALAR[t])[0]


def read_gguf(path: str, max_array: int = 64) -> GGUFInfo:
    """Parse the header, metadata and tensor table of a GGUF file (no tensor data is read)."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = _rd(f, "I")
        if version < 2:
            raise ValueError(f"GGUF version {version} not supported")
        (n_tensors,) = _rd(f, "Q")
        (n_kv,) = _rd(f, "Q")
        info = GGUFInfo(version=version)
        for _ in range(n_kv):
            key = _rstr(f)
            (t,) = _rd(f, "I")
            info.kv[key] = _rval(f, t, max_array)
        info.alignment = int(info.kv.get("general.alignment", 32))
        for _ in range(n_tensors):
            name = _rstr(f)
            (nd,) = _rd(f, "I")
            dims = [_rd(f, "Q")[0] for _ in range(nd)]
            (ty,) = _rd(f, "I")
            (off,) = _rd(f, "Q")
            info.tensors.append(TensorInfo(name, dims, ty, off))
        pos = f.tell()
        a = info.alignment
        info.data_start = (pos + a - 1) // a * a
    return info


def read_tensor_bytes(path: str, info: GGUFInfo, name: str) -> bytes:
    t = info.tensor(name)
    with open(path, "rb") as f:
        f.seek(info.data_start + t.offset)
        return f.read(t.nbytes)


# ---------------------------------------------------------------- writer (for synthetic tests)

def _wstr(out: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    out += struct.pack("<Q", len(b)) + b


def _wval(out: bytearray, t: int, v: Any, elem_type: int | None = None) -> None:
    if t == T_STRING:
        _wstr(out, v)
    elif t == T_ARRAY:
        assert elem_type is not None
        out += struct.pack("<IQ", elem_type, len(v))
        for x in v:
            _wval(out, elem_type, x)
    else:
        out += struct.pack("<" + _SCALAR[t], v)


def write_gguf(path: str, kv: Iterable[tuple], tensors: Iterable[tuple[str, list[int], str, bytes]],
               alignment: int = 32) -> None:
    """Write a small GGUF v3 file.

    kv: tuples (key, value_type, value) or (key, T_ARRAY, values, elem_type).
    tensors: (name, dims, type_name, payload_bytes); payload length must match the type.
    """
    kv = list(kv)
    tensors = list(tensors)
    out = bytearray(MAGIC)
    out += struct.pack("<IQQ", 3, len(tensors), len(kv))
    for item in kv:
        key, t, v = item[0], item[1], item[2]
        _wstr(out, key)
        out += struct.pack("<I", t)
        _wval(out, t, v, item[3] if len(item) > 3 else None)
    offset = 0
    offsets = []
    for name, dims, tname, payload in tensors:
        n = 1
        for d in dims:
            n *= d
        if len(payload) != tensor_nbytes(tname, n):
            raise ValueError(f"{name}: payload size does not match {tname}")
        offsets.append(offset)
        _wstr(out, name)
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", TYPE_IDS[tname], offset)
        offset += (len(payload) + alignment - 1) // alignment * alignment
    out += b"\0" * ((-len(out)) % alignment)
    for (name, dims, tname, payload), off in zip(tensors, offsets):
        out += payload
        out += b"\0" * ((-len(out)) % alignment)
    with open(path, "wb") as f:
        f.write(bytes(out))


def main(argv: list[str] | None = None) -> int:
    """CLI: print metadata summary and tensor table of a GGUF file."""
    import argparse

    ap = argparse.ArgumentParser(description="Print GGUF metadata and the tensor table (read-only).")
    ap.add_argument("file")
    ap.add_argument("--table-only", action="store_true", help="print 'name dims type' lines only")
    a = ap.parse_args(argv)
    info = read_gguf(a.file)
    if not a.table_only:
        print("version", info.version, "tensors", len(info.tensors), "kv", len(info.kv))
        for k, v in info.kv.items():
            if isinstance(v, str) and len(v) > 200:
                v = f"<string {len(v)} chars>"
            print("KV", k, "=", v)
        print("TYPE SUMMARY (count, elements):", info.type_summary())
    for t in info.tensors:
        print(t.name, "x".join(map(str, t.dims)), t.type_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

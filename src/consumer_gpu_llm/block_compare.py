"""Provenance probe: how many quantization blocks of one tensor are byte-identical in two GGUF files?

Two files quantized from the same source with the same recipe and the same importance matrix
should agree block for block. A low share of identical blocks means a different source,
recipe or imatrix, whatever the metadata says.

The original measurement compared a local file against a header copy plus an HTTP range read
of a public file; this version compares two local files (or a header+range copy saved to disk).
"""
from __future__ import annotations

import hashlib

from .gguf import BLOCK, read_gguf, read_tensor_bytes


def compare_blocks(a: bytes, b: bytes, block_bytes: int) -> tuple[int, int]:
    """(identical blocks, total blocks) for two payloads of equal length."""
    if len(a) != len(b):
        raise ValueError("payload lengths differ")
    if len(a) % block_bytes:
        raise ValueError("payload is not a whole number of blocks")
    total = len(a) // block_bytes
    same = sum(1 for i in range(0, len(a), block_bytes) if a[i:i + block_bytes] == b[i:i + block_bytes])
    return same, total


def compare_tensor(path_a: str, path_b: str, name: str) -> dict:
    ia, ib = read_gguf(path_a), read_gguf(path_b)
    ta, tb = ia.tensor(name), ib.tensor(name)
    if ta.type_id != tb.type_id or ta.dims != tb.dims:
        raise ValueError(f"{name}: type/shape differ ({ta.type_name} {ta.dims} vs {tb.type_name} {tb.dims})")
    pa, pb = read_tensor_bytes(path_a, ia, name), read_tensor_bytes(path_b, ib, name)
    same, total = compare_blocks(pa, pb, BLOCK[ta.type_name][1])
    return {
        "tensor": name, "type": ta.type_name, "bytes": len(pa),
        "identical_blocks": same, "total_blocks": total,
        "share": same / total if total else 0.0,
        "sha1_a": hashlib.sha1(pa).hexdigest()[:12], "sha1_b": hashlib.sha1(pb).hexdigest()[:12],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Count byte-identical quant blocks of one tensor in two GGUF files.")
    ap.add_argument("gguf_a")
    ap.add_argument("gguf_b")
    ap.add_argument("tensor", help="e.g. blk.0.ffn_gate.weight")
    a = ap.parse_args(argv)
    print(json.dumps(compare_tensor(a.gguf_a, a.gguf_b, a.tensor)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

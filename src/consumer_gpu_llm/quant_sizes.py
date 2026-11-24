"""Payload size of mixed-precision GGUF recipes, computed from a tensor table.

Input: lines "name dims type" (as printed by `gguf.py --table-only`, dims joined by 'x').
A recipe is a list of rules; each rule retypes the tensors whose module name matches,
optionally only inside a block range. Sizes are exact payload bytes (no header, no padding).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .gguf import tensor_nbytes

_BLK = re.compile(r"blk\.(\d+)\.(.*)\.weight$")


@dataclass(frozen=True)
class Row:
    name: str
    n_elements: int
    type_name: str


@dataclass(frozen=True)
class Rule:
    module: str                      # exact module name, e.g. "output", "ssm_out", "ffn_down"
    new_type: str
    blocks: tuple[int, int] | None = None  # inclusive block range; None = any block (or non-block tensor)


def parse_table(lines: Iterable[str]) -> list[Row]:
    rows = []
    for line in lines:
        p = line.split()
        if len(p) != 3:
            continue
        n = 1
        for d in p[1].split("x"):
            n *= int(d)
        rows.append(Row(p[0], n, p[2]))
    return rows


def role(name: str) -> tuple[int | None, str]:
    """('blk.3.ffn_down.weight') -> (3, 'ffn_down'); ('output.weight') -> (None, 'output')."""
    m = _BLK.match(name)
    if m:
        return int(m.group(1)), m.group(2)
    return None, name.replace(".weight", "")


def apply_recipe(rows: list[Row], rules: list[Rule]) -> tuple[int, Counter]:
    """Return (payload bytes, Counter of (module, old_type, new_type) changes). Later rules win."""
    total = 0
    changes: Counter = Counter()
    for r in rows:
        b, mod = role(r.name)
        t = r.type_name
        for rule in rules:
            if rule.module != mod:
                continue
            if rule.blocks is not None and (b is None or not rule.blocks[0] <= b <= rule.blocks[1]):
                continue
            t = rule.new_type
        total += tensor_nbytes(t, r.n_elements)
        if t != r.type_name:
            changes[(mod, r.type_name, t)] += 1
    return total, changes


_UNSLOTH_LIKE = [Rule("output", "Q6_K"), Rule("ssm_out", "Q5_K"), Rule("ffn_down", "Q4_1", (0, 3))]

# Recipes evaluated for a 32-block Qwen3.5-9B Q4_0 base (see docs/GGUF_TOOLS.md).
RECIPES: dict[str, list[Rule]] = {
    "plain": [],
    "head_Q6K": [Rule("output", "Q6_K")],
    "ssmout_Q5K": [Rule("ssm_out", "Q5_K")],
    "unsloth_like": _UNSLOTH_LIKE,
    # unsloth_like + attn_output / attn_v at Q8_0 in every full-attention block
    "mixA": _UNSLOTH_LIKE + [Rule("attn_output", "Q8_0"), Rule("attn_v", "Q8_0")],
    # unsloth_like + ffn_down Q4_1 also in blocks 28-31
    "mixB": _UNSLOTH_LIKE + [Rule("ffn_down", "Q4_1", (28, 10**6))],
    "headQ6K_ssmoutQ5K": [Rule("output", "Q6_K"), Rule("ssm_out", "Q5_K")],
}


def report(rows: list[Row], recipes: dict[str, list[Rule]] = RECIPES) -> list[tuple[str, int, int, Counter]]:
    """[(recipe, bytes, delta_vs_first, changes)]."""
    out = []
    base = None
    for name, rules in recipes.items():
        total, chg = apply_recipe(rows, rules)
        if base is None:
            base = total
        out.append((name, total, total - base, chg))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Payload size per recipe from a GGUF tensor table.")
    ap.add_argument("table", help="file with 'name dims type' lines")
    a = ap.parse_args(argv)
    with open(a.table, encoding="utf-8") as f:
        rows = parse_table(f)
    for name, total, delta, chg in report(rows):
        print(f"{name:18s} {total / 1e9:.4f} GB  delta {delta / 1e6:+8.1f} MB  changes {dict(chg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

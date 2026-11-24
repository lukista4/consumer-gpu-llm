"""Read and check the per-micro-batch ledger written by patch 0003 (LORA_LEDGER_PATH).

Columns: kind epoch pos pos_max updates_done lr item_loss item_correct item_masked run_loss run_ndata

The coverage check is the one that exposes the fresh-epoch skip (patch 0002): with the bug,
an epoch of N training items yields N - 1 train records and position 0 is missing.
"""
from __future__ import annotations

import csv
from collections import defaultdict

INT_COLS = {"epoch", "pos", "pos_max", "updates_done", "item_correct", "item_masked", "run_ndata"}
FLOAT_COLS = {"lr", "item_loss", "run_loss"}


def read_ledger(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            row = {}
            for k, v in r.items():
                if k in INT_COLS:
                    row[k] = int(v)
                elif k in FLOAT_COLS:
                    row[k] = float(v)
                else:
                    row[k] = v
            rows.append(row)
    return rows


def summarize(rows: list[dict], kind: str = "train") -> dict:
    sel = [r for r in rows if r["kind"] == kind]
    masked = sum(r["item_masked"] for r in sel)
    correct = sum(r["item_correct"] for r in sel)
    return {
        "records": len(sel),
        "supervised_tokens": masked,
        "token_accuracy": correct / masked if masked else float("nan"),
        "mean_item_loss": sum(r["item_loss"] for r in sel) / len(sel) if sel else float("nan"),
    }


def check_train_coverage(rows: list[dict], expected_items: int) -> dict:
    """Per epoch: exactly `expected_items` train records with contiguous, distinct positions."""
    by_epoch: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["kind"] == "train":
            by_epoch[r["epoch"]].append(r["pos"])
    problems = []
    for ep, pos in sorted(by_epoch.items()):
        if len(pos) != expected_items:
            problems.append(f"epoch {ep}: {len(pos)} train records, expected {expected_items}")
        if len(set(pos)) != len(pos):
            problems.append(f"epoch {ep}: repeated positions")
        if pos and sorted(pos) != list(range(min(pos), min(pos) + len(pos))):
            problems.append(f"epoch {ep}: positions not contiguous")
    if not by_epoch:
        problems.append("no train records")
    return {"ok": not problems, "problems": problems, "epochs": len(by_epoch)}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Summarize and check a fine-tuning ledger TSV.")
    ap.add_argument("ledger")
    ap.add_argument("--expected-train-items", type=int, required=True)
    a = ap.parse_args(argv)
    rows = read_ledger(a.ledger)
    out = {"train": summarize(rows, "train"), "val": summarize(rows, "val"),
           "coverage": check_train_coverage(rows, a.expected_train_items)}
    print(json.dumps(out, indent=1))
    return 0 if out["coverage"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

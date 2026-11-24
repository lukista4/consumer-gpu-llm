"""Lever register: one entry per performance lever, validated so no number travels without its regime.

Entry fields
  id, problem, lever, rationale       free text
  predicted                           written BEFORE measuring (or "none")
  observed                            what was measured (free text with numbers)
  label                               MEASURED | DERIVED | PROJECTED
  regime                              dict, required keys below
  gate                                adoption rule
  status                              DEPLOYED | RETEST | RETIRED
  status_reason                       required for RETEST and RETIRED
  attempts                            list of later attempts; appended, never rewritten

Rules
  1. Every entry has a complete regime (model file, runtime build, slots x ctx, KV type, call shape).
  2. A measured entry has a non-empty `observed`.
  3. RETIRED / RETEST carry a reason.
  4. Ids are unique.
  5. `append_attempt` never edits earlier attempts.
"""
from __future__ import annotations

import copy
import json

REGIME_KEYS = ("model_file", "runtime", "slots_x_ctx", "kv_type", "call_shape")
STATUSES = ("DEPLOYED", "RETEST", "RETIRED")
LABELS = ("MEASURED", "DERIVED", "PROJECTED")
REQUIRED = ("id", "problem", "lever", "predicted", "observed", "label", "regime", "gate", "status")


def validate_entry(e: dict) -> list[str]:
    errs = []
    eid = e.get("id", "<no id>")
    for k in REQUIRED:
        if k not in e:
            errs.append(f"{eid}: missing field {k}")
    reg = e.get("regime", {}) or {}
    for k in REGIME_KEYS:
        if not str(reg.get(k, "")).strip():
            errs.append(f"{eid}: regime.{k} missing")
    if e.get("label") not in LABELS:
        errs.append(f"{eid}: label must be one of {LABELS}")
    if e.get("status") not in STATUSES:
        errs.append(f"{eid}: status must be one of {STATUSES}")
    if e.get("status") in ("RETEST", "RETIRED") and not str(e.get("status_reason", "")).strip():
        errs.append(f"{eid}: {e.get('status')} needs status_reason")
    if e.get("label") == "MEASURED" and not str(e.get("observed", "")).strip():
        errs.append(f"{eid}: MEASURED entry without observed value")
    for i, a in enumerate(e.get("attempts", [])):
        if "observed" not in a or "regime" not in a:
            errs.append(f"{eid}: attempt {i} needs observed and regime")
    return errs


def validate_register(entries: list[dict]) -> list[str]:
    errs = []
    seen = set()
    for e in entries:
        if e.get("id") in seen:
            errs.append(f"duplicate id {e.get('id')}")
        seen.add(e.get("id"))
        errs += validate_entry(e)
    return errs


def append_attempt(entry: dict, attempt: dict) -> dict:
    """Return a new entry with `attempt` appended; earlier attempts are copied unchanged."""
    new = copy.deepcopy(entry)
    new.setdefault("attempts", []).append(copy.deepcopy(attempt))
    return new


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate a lever register JSON file.")
    ap.add_argument("register")
    a = ap.parse_args(argv)
    errs = validate_register(load(a.register))
    for e in errs:
        print("ERROR", e)
    print("OK" if not errs else f"{len(errs)} error(s)")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())

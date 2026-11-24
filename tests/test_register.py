import copy
import os

from consumer_gpu_llm import register

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "lever_register_home.json")


def test_example_register_is_valid():
    entries = register.load(EXAMPLE)
    assert register.validate_register(entries) == []
    assert {e["status"] for e in entries} <= set(register.STATUSES)


def test_missing_regime_field_is_an_error():
    e = copy.deepcopy(register.load(EXAMPLE)[0])
    e["regime"]["kv_type"] = ""
    assert any("regime.kv_type" in x for x in register.validate_entry(e))


def test_retired_needs_reason_and_ids_unique():
    entries = register.load(EXAMPLE)
    bad = copy.deepcopy(entries[1])
    bad.pop("status_reason")
    assert any("status_reason" in x for x in register.validate_entry(bad))
    assert any("duplicate id" in x for x in register.validate_register(entries + [entries[0]]))


def test_append_attempt_keeps_history():
    e = register.load(EXAMPLE)[0]
    before = copy.deepcopy(e["attempts"])
    new = register.append_attempt(e, {"observed": "x", "regime": "y"})
    assert new["attempts"][:-1] == before and new["attempts"][-1]["observed"] == "x"
    assert e["attempts"] == before  # original untouched


def test_cli(capsys):
    assert register.main([EXAMPLE]) == 0
    assert "OK" in capsys.readouterr().out

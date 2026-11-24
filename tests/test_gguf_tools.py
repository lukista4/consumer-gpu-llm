import os
import random

import pytest

from consumer_gpu_llm import block_compare, gguf, quant_sizes

HERE = os.path.dirname(__file__)
TABLE = os.path.join(HERE, "..", "examples", "qwen35_9b_q4_0_tensor_table.txt")


def _rand(n, seed):
    r = random.Random(seed)
    return bytes(r.randrange(256) for _ in range(n))


def _write(path, ffn_payload):
    gguf.write_gguf(
        path,
        kv=[("general.architecture", gguf.T_STRING, "toy"),
            ("toy.block_count", 4, 2),
            ("tokenizer.ggml.tokens", gguf.T_ARRAY, ["a", "b", "c"], gguf.T_STRING)],
        tensors=[("token_embd.weight", [64, 3], "F32", _rand(64 * 3 * 4, 1)),
                 ("blk.0.ffn_gate.weight", [64, 8], "Q4_0", ffn_payload),
                 ("output.weight", [32, 3], "Q8_0", _rand(3 * 34, 2))],
    )


def test_roundtrip_header_and_table(tmp_path):
    p = str(tmp_path / "a.gguf")
    _write(p, _rand(16 * 18, 3))
    info = gguf.read_gguf(p)
    assert info.version == 3
    assert info.kv["general.architecture"] == "toy"
    assert info.kv["toy.block_count"] == 2
    assert info.kv["tokenizer.ggml.tokens"] == ["a", "b", "c"]
    assert [t.name for t in info.tensors] == ["token_embd.weight", "blk.0.ffn_gate.weight", "output.weight"]
    t = info.tensor("blk.0.ffn_gate.weight")
    assert t.type_name == "Q4_0" and t.n_elements == 512 and t.nbytes == 16 * 18
    assert info.type_summary()["Q4_0"] == (1, 512)
    assert gguf.read_tensor_bytes(p, info, "output.weight") == _rand(3 * 34, 2)


def test_long_arrays_are_summarized(tmp_path):
    p = str(tmp_path / "b.gguf")
    gguf.write_gguf(p, kv=[("x", gguf.T_ARRAY, list(range(100)), 4)], tensors=[])
    assert gguf.read_gguf(p).kv["x"] == "<array 100>"


def test_rejects_non_gguf(tmp_path):
    p = tmp_path / "c.bin"
    p.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(ValueError):
        gguf.read_gguf(str(p))


def test_cli_table_only(tmp_path, capsys):
    p = str(tmp_path / "d.gguf")
    _write(p, _rand(16 * 18, 3))
    assert gguf.main([p, "--table-only"]) == 0
    assert "blk.0.ffn_gate.weight 64x8 Q4_0" in capsys.readouterr().out


def test_block_compare_counts_identical_blocks(tmp_path):
    base = bytearray(_rand(16 * 18, 3))
    a, b = str(tmp_path / "a.gguf"), str(tmp_path / "b.gguf")
    _write(a, bytes(base))
    changed = bytearray(base)
    for blk in (0, 5, 9):  # perturb 3 of 16 blocks
        changed[blk * 18] ^= 0xFF
    _write(b, bytes(changed))
    r = block_compare.compare_tensor(a, b, "blk.0.ffn_gate.weight")
    assert (r["identical_blocks"], r["total_blocks"]) == (13, 16)
    assert r["type"] == "Q4_0"


def test_block_compare_refuses_shape_mismatch(tmp_path):
    a, b = str(tmp_path / "a.gguf"), str(tmp_path / "b.gguf")
    _write(a, _rand(16 * 18, 3))
    gguf.write_gguf(b, kv=[], tensors=[("blk.0.ffn_gate.weight", [32, 8], "Q4_0", _rand(8 * 18, 4))])
    with pytest.raises(ValueError):
        block_compare.compare_tensor(a, b, "blk.0.ffn_gate.weight")


def test_tensor_nbytes():
    assert gguf.tensor_nbytes("Q4_0", 4096 * 4096) == 4096 * 4096 // 32 * 18
    assert gguf.tensor_nbytes("Q6_K", 256) == 210
    with pytest.raises(ValueError):
        gguf.tensor_nbytes("Q4_K", 100)


def test_recipe_sizes_reproduce_published_table():
    # Published (docs/GGUF_TOOLS.md): plain 5.5519 GB; head Q6_K +262.2 MB; ssm_out Q5_K +50.3 MB;
    # unsloth-like +325.1 MB; mixA +409.0 MB; mixB +337.7 MB.
    with open(TABLE, encoding="utf-8") as f:
        rows = quant_sizes.parse_table(f)
    rep = {name: (total, delta, chg) for name, total, delta, chg in quant_sizes.report(rows)}
    assert round(rep["plain"][0] / 1e9, 4) == 5.5519
    assert round(rep["head_Q6K"][1] / 1e6, 1) == 262.2
    assert round(rep["ssmout_Q5K"][1] / 1e6, 1) == 50.3
    assert round(rep["unsloth_like"][1] / 1e6, 1) == 325.1
    assert round(rep["mixA"][1] / 1e6, 1) == 409.0
    assert round(rep["mixB"][1] / 1e6, 1) == 337.7
    assert rep["unsloth_like"][2][("ffn_down", "Q4_0", "Q4_1")] == 4
    assert rep["mixB"][2][("ffn_down", "Q4_0", "Q4_1")] == 8
    assert rep["ssmout_Q5K"][2][("ssm_out", "Q4_0", "Q5_K")] == 24


def test_tensor_table_matches_architecture():
    with open(TABLE, encoding="utf-8") as f:
        rows = quant_sizes.parse_table(f)
    roles = [quant_sizes.role(r.name) for r in rows]
    full_attn = sorted({b for b, m in roles if m == "attn_q"})
    deltanet = sorted({b for b, m in roles if m == "attn_qkv"})
    assert full_attn == list(range(3, 32, 4))          # 8 full-attention blocks
    assert len(deltanet) == 24 and not set(deltanet) & set(full_attn)

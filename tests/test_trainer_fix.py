import os
import re

from consumer_gpu_llm import epoch_loop, ledger

PATCHES = os.path.join(os.path.dirname(__file__), "..", "patches")


def test_upstream_fresh_epoch_skips_position_zero():
    pos = epoch_loop.trained_positions(380, fixed=False)
    assert pos[0] == 1 and len(pos) == 379          # the published case: 379 of 380 updates


def test_fixed_fresh_epoch_covers_all():
    assert epoch_loop.trained_positions(380, fixed=True) == list(range(380))


def test_fixed_resume_continues_after_completed_updates():
    # k updates completed (positions 0..k-1) -> resume at k, nothing repeated or skipped
    for k in (1, 7, 379):
        pos = epoch_loop.trained_positions(380, start_step=k, fixed=True)
        assert pos[0] == k and len(pos) == 380 - k


def test_later_epochs_start_at_zero_after_fix():
    assert epoch_loop.trained_positions(10, start_step=3, fixed=True, epoch=1, start_epoch=0)[0] == 0
    assert epoch_loop.trained_positions(10, start_step=3, fixed=False, epoch=1, start_epoch=0)[0] == 1


def _write_ledger(path, positions, val=2):
    cols = "kind\tepoch\tpos\tpos_max\tupdates_done\tlr\titem_loss\titem_correct\titem_masked\trun_loss\trun_ndata"
    lines = [cols]
    for i, p in enumerate(positions):
        lines.append(f"train\t0\t{p}\t{len(positions)}\t{i + 1}\t5e-05\t1.0\t20\t30\t1.0\t{i + 1}")
    for j in range(val):
        lines.append(f"val\t0\t{j + 1}\t{val}\t{len(positions)}\t5e-06\t0.5\t9\t10\t0.5\t{j + 1}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_ledger_coverage_catches_the_skip(tmp_path):
    n = 12
    buggy = str(tmp_path / "buggy.tsv")
    good = str(tmp_path / "good.tsv")
    _write_ledger(buggy, [p + 1 for p in epoch_loop.trained_positions(n, fixed=False)])
    _write_ledger(good, [p + 1 for p in epoch_loop.trained_positions(n, fixed=True)])
    rb = ledger.check_train_coverage(ledger.read_ledger(buggy), n)
    rg = ledger.check_train_coverage(ledger.read_ledger(good), n)
    assert not rb["ok"] and "11 train records" in rb["problems"][0]
    assert rg["ok"]


def test_ledger_summary(tmp_path):
    p = str(tmp_path / "l.tsv")
    _write_ledger(p, list(range(1, 5)), val=3)
    rows = ledger.read_ledger(p)
    s = ledger.summarize(rows, "train")
    assert s["records"] == 4 and s["supervised_tokens"] == 120 and abs(s["token_accuracy"] - 20 / 30) < 1e-12
    v = ledger.summarize(rows, "val")
    assert v["records"] == 3 and abs(v["token_accuracy"] - 0.9) < 1e-12
    assert ledger.main([p, "--expected-train-items", "4"]) == 0


def test_patch_series_is_well_formed():
    files = sorted(f for f in os.listdir(PATCHES) if f.endswith(".patch"))
    assert [f[:4] for f in files] == ["0001", "0002", "0003", "0004"]
    text = {f: open(os.path.join(PATCHES, f), encoding="utf-8").read() for f in files}
    for f, t in text.items():
        assert re.search(r"^diff --git a/", t, re.M), f
        assert "@" not in t.split("\n")[1], "no e-mail in patch headers"
    p2 = text[files[1]]
    assert "-        int64_t resume_batch = 0;" in p2
    assert "+        int64_t resume_batch = -1;" in p2
    assert "+            resume_batch = (start_step % training_batches_per_epoch) - 1;" in p2
    p1 = text[files[0]]
    for name in ("attn_qkv", "attn_gate", "ssm_out", "deltanet"):
        assert f'{{"{name}",' in p1

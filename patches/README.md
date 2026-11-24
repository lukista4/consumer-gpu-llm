# Patch set for llama.cpp LoRA fine-tuning (QVAC Fabric fork)

Base: `tetherto/qvac-fabric-llm.cpp` at commit `2db8b046d555316019ae75d1be90484adc8b9330`
(2026-11-05). The fork and llama.cpp/ggml are MIT-licensed (Copyright (c) 2023-2026 The ggml
authors); these patches are offered under the same licence.

| patch | files | what | used in the reported runs |
|---|---|---|---|
| 0001 | `include/llama.h`, `src/llama-lora-training.cpp`, `examples/training/finetune-lora.cpp` | LoRA target bits for the Gated DeltaNet projections `attn_qkv`, `attn_gate`, `ssm_out`; CLI group `deltanet` (and `all` now includes them) | yes |
| 0002 | `examples/training/finetune-lora.cpp` | fix: a fresh epoch skipped dataset position 0 (docs/TRAINER_OFF_BY_ONE.md) | yes |
| 0003 | `ggml/include/ggml-opt.h`, `ggml/src/ggml-opt.cpp`, `examples/training/finetune-lora.cpp` | optional TSV ledger, one line per evaluated micro-batch, enabled by `LORA_LEDGER_PATH` | yes |
| 0004 | `src/llama-lora-training.cpp` | optional block-range filter `LORA_BLOCKS=LO-HI` (0-based, inclusive; malformed value aborts) | built, not used in the reported runs |

The environment variable names and comments were renamed for publication (the measured builds
used project-internal names); the logic is unchanged. Later patches use earlier ones' context
lines, so apply them in order.

```sh
git clone https://github.com/tetherto/qvac-fabric-llm.cpp
cd qvac-fabric-llm.cpp
git checkout 2db8b046d555316019ae75d1be90484adc8b9330
for p in /path/to/consumer-gpu-llm/patches/000*.patch; do git apply "$p"; done
```

On Windows checkouts with `core.autocrlf=true`, use `git apply --ignore-whitespace`.

Verification done here: the series applies cleanly, in order, to the files of that commit
(`git apply`, checked to reproduce the intended tree). The patched tree was built with Vulkan and used for the runs in
docs/LORA_FINETUNE_9B.md; the rename for publication was not rebuilt.

Ledger columns (0003):
`kind epoch pos pos_max updates_done lr item_loss item_correct item_masked run_loss run_ndata`.
Check one with `python scripts/check_ledger.py LEDGER.tsv --expected-train-items N`.

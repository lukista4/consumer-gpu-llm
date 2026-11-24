# GGUF tools

Small, dependency-free, read-only tools (Python standard library only).

| script | what it does |
|---|---|
| `scripts/gguf_info.py FILE` | metadata, tensor-type census, tensor table (`--table-only` for `name dims type` lines) |
| `scripts/quant_sizes.py TABLE` | exact payload size of mixed-precision recipes from a tensor table |
| `scripts/gguf_block_compare.py A B TENSOR` | how many quantization blocks of one tensor are byte-identical in two files |

## Tensor-type census: "can this kernel fire on this file?"

A custom Vulkan kernel for Q4_0 x F32 matmuls was benchmarked on a Q4_0 file and then tried on
the deployed Q4_K_M file: +3%, a tie. The census answered why in one command: the Q4_K_M file
holds **0 Q4_0 tensors** (Q4_K 125, Q6_K 49, Q5_K 16, Q8_0 20, F32 232). A missing log line from
the kernel would not have shown this; the census did.

## Recipe sizes (COMPUTED from the tensor table in `examples/`)

Base: the local Qwen3.5-9B Q4_0 training file (`token_embd` and `ssm_alpha/beta` at Q8_0).
`python scripts/quant_sizes.py examples/qwen35_9b_q4_0_tensor_table.txt`:

| recipe | payload GB | delta MB |
|---|---|---|
| plain (as is) | 5.5519 | 0 |
| head (`output`) at Q6_K | 5.8141 | +262.2 |
| `ssm_out` at Q5_K (24 tensors) | 5.6022 | +50.3 |
| head Q6_K + `ssm_out` Q5_K | 5.8644 | +312.6 |
| Unsloth-like: head Q6_K, `ssm_out` Q5_K, `ffn_down` Q4_1 in blocks 0-3 | 5.8770 | +325.1 |
| mixA: Unsloth-like + `attn_output`/`attn_v` Q8_0 in all 8 attention blocks | 5.9609 | +409.0 |
| mixB: Unsloth-like + `ffn_down` Q4_1 also in blocks 28-31 | 5.8896 | +337.7 |

With 0.4-0.65 GB of training headroom on the 8 GB card, +325 MB was the practical ceiling. The
planned KL measurement per recipe was **not run** (no quality numbers exist for these recipes).
The tests check that these sizes are reproduced exactly.

## Provenance by block comparison (MEASURED)

The local Q4_0 file's header names the same public importance matrix as a publicly shipped
Q4_0 of the same model. Comparing `blk.0.ffn_gate.weight` block by block (18-byte Q4_0 blocks)
against the shipped file: **257,005 of 1,572,864 blocks (16%) identical**. So the two files were
not produced the same way, whatever the metadata says (different source precision, recipe or
tool version; which one was not determined).

The original probe read the public file's tensor through an HTTP range request; the version here
compares two local files.

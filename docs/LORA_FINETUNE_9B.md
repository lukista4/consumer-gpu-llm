# LoRA fine-tuning Qwen3.5-9B on an 8 GB RX 7600

All runs: one machine, one run per configuration, done by one person working with AI coding
assistants. The runs test **training mechanics** (does it train, fit, load, and come off
cleanly), not usefulness. No task evaluation of any adapter is reported here.

## Regime

| item | value |
|---|---|
| GPU | AMD Radeon RX 7600, 8 GB (RDNA3), proprietary Windows driver, Vulkan backend |
| host | Ryzen 5 7600, 16 GB RAM at the time of these runs (15.1 GB visible), Windows 11 |
| trainer | `llama-finetune-lora` from the QVAC Fabric llama.cpp fork, commit `2db8b04`, built locally (Vulkan, llvm-mingw) plus patches 0001-0003 where stated |
| checks | stock llama.cpp `llama-perplexity` (Vulkan prebuilt, build b12110), `-c 128 -b 128 -ngl 999 -fa off` |
| model | Qwen3.5-9B: 32 blocks = 24 Gated DeltaNet blocks + 8 full-attention blocks (3, 7, ..., 31); FFN 4096 -> 12288; vocab 248,320 |
| training base | Q4_0 GGUF made locally from a Q8_0: `--pure` Q4_0, `ssm_alpha`/`ssm_beta` and `token_embd` kept at Q8_0, public importance matrix; 5,562,836,224 bytes, 4.96 bits/weight |
| common flags | context = batch = micro-batch unless stated; `-fa off`; `--load-mode none` (no mmap); LoRA rank 8, alpha 16 |
| guard | a watchdog stops a run when free VRAM < 0.375 GB (about 7.625 GB used) or GPU shared memory grows by > 1 GB (raised to 3 GB where stated) |

Prebuilt fork release b7349 refuses the `qwen35` architecture, so a local build of the fork's
master was needed.

## 1. Training base quality gate (MEASURED)

Held-out text perplexity (public llama.cpp documentation, not used in training):

| file | held-out PPL | ratio |
|---|---|---|
| Q4_K_M (public imatrix) | 3.2592 +/- 0.096 | 1 |
| local Q4_0 training base | 3.3066 +/- 0.099 | 1.0145 (gate <= 1.05: PASS) |

## 2. Patch 0001: LoRA on the DeltaNet projections (MEASURED)

The trainer's module filter knew `attn_q/k/v/o`, `ffn_*` and `output` only; `--lora-modules all`
sets bits but cannot reach names it does not match. The `qwen35` graph already routes
`attn_qkv`, `attn_gate` and `ssm_out` through `build_lora_mm`, so only the targeting was missing
(DERIVED from source). Patch 0001 adds three target bits and the group name `deltanet`.

Checks on the patched build:
- the trainer writes adapters for exactly the DeltaNet blocks (2B test model: 3 modules x 18 blocks; 9B: 24 blocks);
- **stock** `llama-perplexity` loads the adapter, the adapter changes the model, and `--lora-scaled ADAPTER:0.0` reproduces the base perplexity exactly (to all printed digits);
- cost on the 9B at context 64 (attention + FFN modules, with vs without `deltanet`): wall 663 s vs 621 s (**+7%**), GPU peak 6.65 vs 6.48 GB (**+0.18 GB**), adapter 80.2 vs 58.2 MB.

Not shown: whether DeltaNet LoRA helps on any task.

## 3. Memory ceilings on 8 GB (MEASURED; extrapolation DERIVED)

GPU dedicated peak, all processes (the idle desktop uses about 0.7 GB):

| modules | context / micro-batch | GPU peak GB | shared GB | note |
|---|---|---|---|---|
| output head only | 128 / 128 | 5.34 | 0.08 | |
| attention + FFN | 64 / 64 | 6.48 | 0.10 | |
| attention + FFN + DeltaNet | 64 / 64 | 6.65 | 0.10 | |
| attention + FFN | 128 / 128 | 7.14 | 0.10 | 0.49 GB under the stop line |
| attention + FFN + DeltaNet | 128 / 128 | 6.61-6.82 | 1.07-1.09 | runs only by spilling about 1 GB to shared memory (spill cap raised to 3 GB) |
| attention(q,o) + FFN | 512 / 128 (4 micro-batches) | 7.15 | 0.10 | flat in context; K/V get no gradient in this mode |

- Going from context 64 to 128 added 0.66 GB, so 192 or 256 would cross the stop line for the
  broad module set (DERIVED; not run).
- With context > micro-batch, memory stays flat (7.15 GB at 512 vs 7.14 GB at 128), so longer
  windows fit; earlier micro-batches act as stop-gradient context. The trainer warns that the
  K and V projections receive no gradient in this mode (MEASURED: warning printed).

### Shared-memory spill from `--load-mode none` (MEASURED fix, DERIVED cause)

With `--load-mode none`, the first broad run on the Q4_0 base was stopped by the guard: GPU
*shared* memory jumped by about 1.04 GB at load. The Q8_0 `token_embd` of this base is exactly
1.006 GiB (the Q4_K_M file's is 0.533 GiB, and that run grew by 0.57 GB), so the CPU-side
embedding most likely lands in a Vulkan pinned host buffer, which Windows counts as GPU shared
memory. Adding `--no-host` to the training command kept shared memory at 0.07 -> 0.10 GB.

## 4. Where the time goes: OUT_PROD is 92% (MEASURED from the Vulkan perf logger)

Config: Q4_0 base, context 512, micro-batch 128, modules attn_q, attn_o, ffn_gate/up/down,
`GGML_VK_PERF_LOGGER=1`, 6 CPU threads.

| op | GPU time for 3 micro-batches | share |
|---|---|---|
| OUT_PROD (2,064 calls, 5.13 ms each) | 10.59 s of 11.53 s | **92%** |
| GATED_DELTA_NET_BACK | 0.32 s | 2.8% |
| forward Q4_0 MUL_MAT | about 0.39 s (10.8-16.6 TFLOP/s) | about 3% |

- Each micro-batch has 688 OUT_PROD nodes; the dominant cost is the input gradient through the
  frozen quantized weights (`d(x) = out_prod(W, grad)`) (DERIVED from the scheduler listing).
- The fork has tiled OUT_PROD shaders, but `ggml-vulkan.cpp` selects them only when
  `device->uma` **and** the driver is Mesa RADV **or** the vendor is Qualcomm (a workaround for
  device-lost errors on integrated GPUs). A discrete card on the proprietary AMD driver always
  takes the simple path (verified in source at `2db8b04`).
- Effective training throughput at this config was about 1.3-1.9 TFLOP/s (DERIVED), far below
  the forward matmul rate. A faster OUT_PROD path for discrete GPUs is the obvious lever; it was
  not built.

## 5. Step speed at context 128 (MEASURED, after the off-by-one fix)

Same data (400 short chat items, 380 train + 20 validation, assistant-only loss, LR 5e-5 -> 5e-6
cosine, seed 42, one epoch):

| modules | wall | per item | context tok/s | supervised tok/s |
|---|---|---|---|---|
| output head only | 115.1 s | 0.282 s | 454 | 103.8 |
| attention + FFN + DeltaNet in all 32 blocks | 2,737.7 s | 7.181 s | 17.8 | 4.1 |

Teacher-forced validation token accuracy on the 20 held-out items of the same file: 0.794
(head only) and 0.993 (broad). These items share format and templates with the training items,
so the numbers show the format was learned. **They say nothing about task performance**; no
generation-based evaluation is reported.

## 6. Adapter mechanics on toy text (MEASURED)

Toy data: public llama.cpp documentation (MIT), 36.6 KB train / separate held-out text.

| run | text | base PPL | with adapter | adapter at scale 0 |
|---|---|---|---|---|
| 9B, output head, ctx 128 | train | 7.5386 | 6.1930 | 7.5386 (exact) |
| 9B, output head, ctx 128 | held-out | 3.2592 | 3.5699 (worse) | - |
| 2B, attention + FFN, ctx 64 | train | 10.2080 | 6.4945 | 10.2080 (exact) |
| 2B, DeltaNet only (patch 0001), ctx 64 | train | 10.2080 | 7.6681 | 10.2080 (exact) |

Toy adapters overfit the small training text and hurt held-out text, as expected.

## Other practical rules learned

- Use context = batch = micro-batch with `-fa off` for training on this stack.
- Pass the adapter to `--lora-scaled` by a **relative** path on Windows (the drive colon breaks the `PATH:SCALE` parser).
- The SFT loader reads only `role` and `content`, so training renders are always thinking-off for Qwen3.5's chat template; a conversation longer than the context is **skipped with only a warning**. Pre-count tokens and require zero skip lines in the log.
- The trainer pads each conversation to the full context (no packing).

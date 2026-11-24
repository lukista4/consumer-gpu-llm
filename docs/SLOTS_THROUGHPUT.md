# Slots x context: aggregate throughput on one 8 GB card

On this card, aggregate decode speed is set mainly by **how many sequences decode together**,
and VRAM caps the **total** context (slots x context per slot). So long per-request context and
high aggregate throughput compete for the same memory.

## 1. Sweep with long outputs, thinking off (MEASURED)

Regime: Qwen3.5-9B **Q4_0** file, RX 7600 8 GB, Vulkan, `llama-server` from a custom build with
a small-N Q4_0 matmul kernel (env `GGML_VK_Q4_SMALLN=16`, `GGML_VK_Q4_SMALLN_ROWS=2`, in-place
recurrent state), 3 x slots requests per arm, `n_predict = ctx - 128`, temperature 0.7,
top-p 0.8, top-k 20. One run per arm. Peak = GPU dedicated, all processes.

| slots x ctx | KV | aggregate tok/s | per-slot tok/s | peak VRAM GB |
|---|---|---|---|---|
| 8 x 2,048 | f16 | 112.5 | 14.1 | 6.08 |
| 8 x 8,192 | q8_0 | 118.2 | 14.8 | 6.71 |
| 16 x 1,536 | f16 | 288.5 | 18.3 | 6.63 |
| 16 x 4,096 | q8_0 | 288.9 | 18.2 | 7.12 |
| 32 x 768 | f16 | 431.2 | 14.0 | 7.46 |
| 32 x 1,536 | q8_0 | 429.7 | 13.6 | 7.52 |
| 32 x 768, WMMA build | f16 | 531.1 | 17.3 | 7.47 |
| 16 x 4,096, WMMA build | q8_0 | 286.0 | 18.1 | 7.09 |

All arms finished with 0 errors. The WMMA build is a speed reference only: its output-agreement
gate was never run (see NEGATIVE_RESULTS.md).

Reading (DERIVED): about 115 tok/s at 8 slots, about 289 at 16, about 430 at 32, nearly
independent of context from 768 to 8,192 per slot. The 32-slot layouts sit at 7.5 GB, at the
edge of the card.

## 2. Think-on calls, stock server build (MEASURED)

Regime: Qwen3.5-9B **Q4_K_M** (public imatrix), `llama-server` stock build of the fork, Vulkan,
q8_0 KV, `-fa on`. Workload: real reasoning ("think-on") calls, prompt median 1,884 tokens,
thinking capped at 4,096 tokens. Each arm ran for a 540 s window; decode speed is read live from
the server's `/slots` endpoint (`n_decoded`), because calls last longer than the window.

| arm | slots x ctx | live decode tok/s | median s per call | VRAM peak GB |
|---|---|---|---|---|
| stock | 3 x 8,192 | 79.0 | 158 | 5.89 |
| stock | 8 x 8,192 | **101.4 (+28%)** | 342 | 6.89 |
| small-N kernel env (cannot fire, see NEGATIVE_RESULTS) | 8 x 8,192 | 104.9 (tie) | 339 | 6.89 |
| small-N kernel env | 10 x 8,192 | 75.9 | - | 7.07 |
| n-gram drafter build | 8 x 8,192 | 74.1 (-27%) | 433 | 6.87 |

- 8 slots beat 3 slots by 28% in aggregate decode; each call takes about 2.2x longer, so this
  is a throughput lever, not a latency lever.
- 10 slots were slower than 8.
- In a longer live run at 8 x 8,192 the same workload completed 1.5 calls/min vs 1.07-1.2 calls/min at
  3 x 16,384 (MEASURED, second-hand from run notes).
- Output quality on a downstream comparison was unchanged across slot counts (0.604 vs 0.594 on a
  private pairwise metric; the data is not published).

## 3. The trade-off (DERIVED)

| use | best layout seen | aggregate tok/s |
|---|---|---|
| short prompts, short outputs | 32 x 768-1,536 | about 430 |
| prompts of a few thousand tokens | 16 x 4,096 | about 289 |
| reasoning calls with up to 8k context | 8 x 8,192 | about 101-118 |
| reasoning calls with 24k context | 4 x 24k | 74 (MEASURED, second-hand) |

Long thinking and a wide batch compete for the same VRAM. Pick the layout per workload and keep
the slot count where the kernel's batch sweet spot is (8 on this build), not above it.

## 4. Lever register format

Every speed number above was entered in a register with one entry per lever:
predicted gain (written before measuring), observed gain, regime, gate, one status
(DEPLOYED / RETEST / RETIRED + reason), and later attempts appended, never overwritten.
A number without its regime (model file, build, slots x ctx, KV type, call shape) is rejected.

- Example: `examples/lever_register_home.json`
- Validator: `python scripts/check_register.py examples/lever_register_home.json`

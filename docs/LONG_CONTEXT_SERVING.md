# Long-context serving of a 9B hybrid model on 8 GB

Qwen3.5-9B has only 8 full-attention blocks out of 32 (the other 24 are Gated DeltaNet blocks
with a fixed-size recurrent state), so its KV cache per token is about a quarter of what the
same attention layout in all 32 blocks would need. That is why 80k tokens of context fit next to a 5.6 GB model on an 8 GB card.

Scripts: `scripts/server_bench.py depth` (one request per depth) and
`scripts/server_bench.py parallel` (k requests at once). Speeds come from the server's own
`timings` block. Every request uses `cache_prompt: false`, temperature 0.6, 256 generated tokens
(512 in section 3).

## 1. Single slot, depth sweep to 78k (MEASURED)

Regime: RX 7600 8 GB, Vulkan, `llama-server` (llama.cpp Vulkan build), model = the local
Qwen3.5-9B **Q4_0** file (5.56 GB), all layers on GPU, 1 slot x 81,920 context, 6 threads.
KV-cache type and flash-attention setting were **not recorded** in the server log.

| prompt tokens | prefill tok/s | decode tok/s | wall s |
|---|---|---|---|
| 501 | 965 | 50.0 | 5.6 |
| 16,000 | 1,006 | 45.5 | 21.5 |
| 32,001 | 850 | 41.4 | 43.8 |
| 59,999 | 640 | 35.9 | 100.9 |
| 77,999 | 561 | 32.9 | 146.9 |

Decode loses about 34% from 0.5k to 78k tokens of depth; prefill peaks around 16k and then
falls to 56% of the peak.

## 2. Parallel long prompts: decode starves (MEASURED)

Same file and build, 3 slots x 65,536. k requests of the same depth sent at once:

| depth | parallel | per-sequence decode tok/s | aggregate decode tok/s | wall s |
|---|---|---|---|---|
| 500 | 1 | 50.1 | 50.3 | 5.6 |
| 500 | 3 | 42.2 / 42.3 / 42.3 | **127.2** | 7.4 |
| 30,000 | 1 | 41.6 | 41.8 | 42.6 |
| 30,000 | 3 | 3.3 / 6.2 / 28.0 | **9.8** | 119.7 |

Aggregate = all generated tokens / the longest decode time.

What happens (from the server log): the three 30k prefills run interleaved for about 40 s each.
A slot that finished its prefill decodes only between the other slots' prefill chunks: 306.6 ms
per token for the first slot, 160.8 ms for the second, 35.7 ms for the last one (which decodes
alone at the end).

Reading (DERIVED):
- At short depth, 3 concurrent requests give 2.5x aggregate decode and 2.3x less wall time than 3 sequential calls (7.4 s vs about 16.8 s).
- At 30k depth, concurrency buys almost nothing in wall time (119.7 s vs about 128 s for 3 sequential calls, 3 x 42.6 s), and per-request latency becomes very uneven (one request decodes at 3.3 tok/s).
- Practical rule for this card: do not start several long prompts at the same moment. Keep the queue deeper than the slot count so arrivals are staggered, or prefill long prompts one at a time. The same rule applies to "waves" of identical requests started together.

## 3. 32k slot with q8_0 KV (MEASURED)

Regime: Qwen3.5-9B **Q4_K_M** (public imatrix), `llama-server` (fork build), Vulkan,
`-ngl 999 -c 32768 -np 1 -ctk q8_0 -ctv q8_0 -fa on --load-mode none --cache-ram 0
--ctx-checkpoints 0`, 6 threads, 512 generated tokens.

| prompt tokens | prefill tok/s | decode tok/s | wall s |
|---|---|---|---|
| 8,000 | 925.4 | 40.5 | 21.3 |
| 20,000 | 818.9 | 38.2 | 37.8 |
| 28,000 | 763.3 | 36.6 | 50.7 |

GPU dedicated peak 6.07 GB of 8 (1.93 GB headroom); shared-memory peak 0.03 GB; load 8.2 s.

## 4. Flags that buy context on 8 GB

| lever | effect | label |
|---|---|---|
| `-ctk q8_0 -ctv q8_0` (needs `-fa on`) | clean single-slot context 65k -> 131k; 3 x 32k fits only with q8_0 | MEASURED, second-hand (from run notes; raw logs not re-checked for this repo) |
| `--load-mode none` | no memory-mapped weights; keeps host working set small | practice |
| `--cache-ram 0 --ctx-checkpoints 0` | no host-side prompt cache / context checkpoints | practice |

The quality cost of q8_0 KV on real tasks was **never measured**. At 16 slots, 16 x 4,096 with q8_0 KV and
16 x 1,536 with f16 KV gave the same throughput (288.9 vs 288.5 tok/s; context also differs, see
SLOTS_THROUGHPUT.md).

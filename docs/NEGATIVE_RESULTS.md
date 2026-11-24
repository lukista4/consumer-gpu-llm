# Negative results: custom kernels and drafting on a saturated consumer GPU

Four custom llama.cpp Vulkan builds were tried for serving. None was kept. The kernel sources
are not part of this repo (they were never reviewed for release); this is the write-up.

| build | idea | best case (MEASURED) | deployed regime (MEASURED) | status |
|---|---|---|---|---|
| small-N kernel | faster Q4_0 x F32 matmul for 8-32 rows | 289 tok/s at 16 x 4,096; 431 at 32 x 768 (Q4_0 file) | 104.9 vs 101.4 tok/s on Q4_K_M at 8 x 8,192: a tie | retired |
| in-place recurrent state | no DeltaNet state copies | 373 vs 331 tok/s, bit-exact on 34 output hashes (Q4_0 file) | no gain in the deployed regime | retired |
| WMMA path | RDNA3 cooperative-matrix Q4_0 matmul for wide batches | 531 vs 431 tok/s at 32 x 768 | tie at 16 x 4,096 (286.0 vs 288.9); agreement gate never run | retired |
| block verifier + n-gram drafter | speculative decoding without a draft model | 333-363 vs 191 tok/s at 16 x 1,664, short thinking-off calls, outputs never quality-checked | **74.1 vs 101.4 tok/s (-27%)** on think-on calls at 8 x 8,192 | retired |

Sources: the small-N and WMMA "best case" numbers and all "deployed regime" numbers were checked
against the raw sweep notes and result JSON; the in-place-state and drafter "best case" numbers are
second-hand (quoted in run notes, raw reports not re-opened).

## Why they failed

1. **The kernel could not run on the file that mattered.** The deployed file is Q4_K_M, with 0
   Q4_0 tensors (census in GGUF_TOOLS.md); the kernel routes Q4_0 x F32 only (DERIVED from the
   ggml-vulkan source). The +3% difference is therefore not a kernel effect; with one run per arm it is
   within what the measurement can resolve.
2. **A Q4_0 file good enough to use did not exist.** The best Q4_0-format candidate had mean KLD
   0.0072 against 0.0042 for Q4_K_M (QUANT_KL.md), outside the 0.005 screen.
3. **Drafting does not help when the batch already saturates the GPU.** At 8 slots, verify steps
   cost compute the batch was already using. Offline replay of real outputs gave only 1.75-1.77
   emitted tokens per verify step at draft length 4, too low to pay for the verifier.
4. **Wins at 32 slots do not transfer.** No real workload fit a 32 x 768 layout; at 16 slots the
   WMMA build tied.

## Rules kept

- Before benchmarking a format-specific kernel, run a tensor-type census of the target file.
- A speed number measured on a different file, call length or slot layout is a different regime;
  re-measure before adopting.
- Retire a lever with a written reason and keep the entry (see `examples/lever_register_home.json`).

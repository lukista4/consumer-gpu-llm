# Quantization quality by KL divergence (Qwen3.5-9B)

Mean KL divergence of each quantized file's next-token distribution against a BF16 reference,
on fixed text (llama.cpp `--kl-divergence` style: per-token KL, then mean / p99, plus the share
of tokens where the top-1 token agrees).

**Label:** MEASURED, second-hand. The numbers come from run notes that quote one KL job run off
the home machine (reference with f16 KV). The raw job output and the exact text set were not
re-opened for this repo. Single run per file.

| file | size | mean KLD | p99 KLD | same top-1 |
|---|---|---|---|---|
| Q8_0 | - | 0.00015 | - | - |
| mixed-bit "MIX6" | 6.32 GB | 0.0032 | - | - |
| **Q4_K_M, public imatrix (deployed)** | - | **0.0042** | - | - |
| Q4_0 / Q8_0 mixed recipe | - | 0.0072 | 0.077 | 97.77% |
| Q4_0 + own imatrix | - | 0.0270 | 0.302 | 95.67% |
| plain Q4_0 (proxy) | - | about 0.035 | - | - |

Reading:
- The K-quant file with an importance matrix is about 6x better in mean KLD than a Q4_0 file
  with its own imatrix and about 8x better than plain Q4_0, at a similar size.
- This decided the kernel work (NEGATIVE_RESULTS.md): fast Q4_0-only kernels were worth
  something only if a Q4_0-format file could pass a KL screen of about 0.005. None did.
- KL against BF16 is a proxy. A task level comparison of the deployed file against BF16 was
  planned and never completed, so no task quality claim is made here.

To rerun on public text: `llama-perplexity -m BF16.gguf -f text.txt --kl-divergence-base base.kld`
then `llama-perplexity -m QUANT.gguf --kl-divergence-base base.kld --kl-divergence`.

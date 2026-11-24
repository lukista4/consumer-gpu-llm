# consumer-gpu-llm

Running and LoRA-fine-tuning a 9B hybrid LLM on an 8 GB consumer AMD GPU: measured results,
a llama.cpp patch set, and small tools.

Hardware: AMD Radeon RX 7600 8 GB (RDNA3), Ryzen 5 7600, Windows 11, llama.cpp **Vulkan**
backend on the proprietary AMD driver. Model: **Qwen3.5-9B** (32 blocks: 24 Gated DeltaNet +
8 full-attention). Everything here is from one machine, mostly one run per configuration.

## Results

Every number is labelled **MEASURED** (from a log or result file), **DERIVED** (arithmetic or
source reading on measured data) or **PROJECTED** (not run). "Second-hand" means quoted from run
notes whose raw file was not re-opened for this repo.

| topic | result | label | doc |
|---|---|---|---|
| LoRA on DeltaNet projections | patch 0001 makes `attn_qkv`, `attn_gate`, `ssm_out` trainable; stock llama.cpp loads the adapter; scale 0 reproduces base PPL exactly; +7% time, +0.18 GB | MEASURED | [LORA_FINETUNE_9B](docs/LORA_FINETUNE_9B.md) |
| trainer off-by-one | a fresh epoch skipped dataset item 0 (379 of 380 updates); fixed by patch 0002 | DERIVED from source, MEASURED in logs | [TRAINER_OFF_BY_ONE](docs/TRAINER_OFF_BY_ONE.md) |
| training memory on 8 GB | broad LoRA (attention + FFN) at context 128: 7.14 GB peak; context 512 with micro-batch 128 stays flat at 7.15 GB | MEASURED | [LORA_FINETUNE_9B](docs/LORA_FINETUNE_9B.md) |
| training step profile | OUT_PROD = 92% of GPU time; the tiled OUT_PROD shaders are gated to integrated RADV/Qualcomm GPUs only | MEASURED / DERIVED from source | [LORA_FINETUNE_9B](docs/LORA_FINETUNE_9B.md) |
| training speed, context 128 | head-only 0.28 s/item (454 ctx tok/s); attention + FFN + DeltaNet in all 32 blocks 7.18 s/item (17.8 ctx tok/s) | MEASURED | [LORA_FINETUNE_9B](docs/LORA_FINETUNE_9B.md) |
| long context, 1 slot | decode 50.0 -> 32.9 tok/s from 0.5k to 78k tokens; prefill 1,006 -> 561 tok/s | MEASURED | [LONG_CONTEXT_SERVING](docs/LONG_CONTEXT_SERVING.md) |
| parallel long prefills | 3 concurrent 30k-token requests: aggregate decode **9.8 tok/s** vs 41.8 for one request; wall time about the same as running them in sequence | MEASURED / DERIVED | [LONG_CONTEXT_SERVING](docs/LONG_CONTEXT_SERVING.md) |
| slots x context | about 430 tok/s aggregate at 32 x 768-1,536; 289 at 16 x 4,096; 118 at 8 x 8,192 (Q4_0 file, custom kernel build) | MEASURED | [SLOTS_THROUGHPUT](docs/SLOTS_THROUGHPUT.md) |
| reasoning calls | 8 x 8,192 slots: 101.4 vs 79.0 tok/s at 3 x 8,192 (+28%) | MEASURED | [SLOTS_THROUGHPUT](docs/SLOTS_THROUGHPUT.md) |
| quantization | mean KLD vs BF16: Q4_K_M + imatrix 0.0042; Q4_0 + own imatrix 0.0270; Q8_0 0.00015 | MEASURED, second-hand | [QUANT_KL](docs/QUANT_KL.md) |
| custom kernels, drafting | small-N Q4_0 kernel: a tie (cannot fire on a Q4_K_M file); n-gram drafter: -27% on a saturated batch; all retired | MEASURED | [NEGATIVE_RESULTS](docs/NEGATIVE_RESULTS.md) |
| GGUF tools | tensor census, exact recipe sizes, block-level provenance check (16% identical blocks between two "same-imatrix" files) | MEASURED / COMPUTED | [GGUF_TOOLS](docs/GGUF_TOOLS.md) |

What is **not** claimed: no adapter here has a task level evaluation; teacher-forced token
accuracy on held-out items of the training file is reported only as a mechanics check. The
quality cost of the q8_0 KV cache was never measured. The kernel and drafter source code is not
published.

## Layout

```
patches/   llama.cpp patch series against the QVAC Fabric fork, commit 2db8b04 (MIT)
docs/      write-ups with regime-tagged tables
src/consumer_gpu_llm/
  gguf.py           read-only GGUF parser (+ a tiny writer for synthetic tests)
  quant_sizes.py    payload size of mixed-precision recipes from a tensor table
  block_compare.py  count byte-identical quant blocks of one tensor in two files
  server_bench.py   llama-server depth sweep and parallel-depth probe
  epoch_loop.py     model of the trainer's epoch loop, upstream vs fixed
  ledger.py         read/check the per-item training ledger written by patch 0003
  register.py       lever register validator (no number without its regime)
scripts/   thin CLI wrappers for the modules
examples/  Qwen3.5-9B Q4_0 tensor table, home lever register (JSON)
tests/     CPU-only tests on synthetic data
```

## Quick start

```sh
python -m pip install -e ".[test]"      # or just: pip install pytest
python -m pytest -q                      # CPU only, no model files needed

python scripts/quant_sizes.py examples/qwen35_9b_q4_0_tensor_table.txt
python scripts/gguf_info.py MODEL.gguf --table-only
python scripts/server_bench.py depth --url http://127.0.0.1:8080
python scripts/server_bench.py parallel --url http://127.0.0.1:8080 --targets 500,30000 --parallel 1,3
python scripts/check_ledger.py LEDGER.tsv --expected-train-items 380
```

Server commands used for the q8_0-KV probe (docs/LONG_CONTEXT_SERVING.md section 3):

```sh
llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 999 -t 6 -c 32768 -np 1 \
  -ctk q8_0 -ctv q8_0 -fa on --load-mode none --cache-ram 0 --ctx-checkpoints 0
```

## License

This repository: public domain (the Unlicense), see [LICENSE](LICENSE).
The files in `patches/` modify llama.cpp/ggml and stay under its MIT licence (see `patches/README.md`).

## Related

- `llm-inference-levers`: the datacenter-GPU (L40S, A100-40GB) side of the same work: vLLM
  levers, speculative-decoding reproducibility, generation-tail analysis.
- `judge-lab-public`: LLM-judge training and reader-equivalence testing.

Author: Lukáš Staník. The work was done with AI coding assistants; all numbers were produced
by runs on the hardware named above.

## AI usage disclosure

This work is run as a human-directed, AI-agent-executed workflow.

- **My role:** I set the goals, priorities, constraints and budgets, chose what to pursue or drop, ran it on my own hardware and accounts, and made the key decisions.
- **The agents' role:** most of the code, experiment runs, analysis and documentation were produced by AI agents:
  - Anthropic Claude (Claude Code) wrote most code and docs; commits carry a `Co-Authored-By: Claude` trailer;
  - OpenAI GPT models via Codex did code and statistics reviews;
  - Google Gemini did design and code reviews.
- **Models as instruments:** several model families (including Mistral-hosted models) were also used as graders. Those grades are model judgements, not human ground truth, and are labelled as such.
- **Checks:** results were checked by automated tests and by reviews from a different model family than the author's. An AI review is not an expert human review; where only one agent checked something, it says so.
- **Measurements:** all numbers come from real runs on real hardware. Each is labelled MEASURED, DERIVED or PROJECTED.

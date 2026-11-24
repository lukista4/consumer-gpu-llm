# Fresh-epoch off-by-one in the llama.cpp LoRA fine-tuning example

**Where:** QVAC Fabric fork of llama.cpp (`tetherto/qvac-fabric-llm.cpp`), commit
`2db8b046d555316019ae75d1be90484adc8b9330`. The same code was not checked in upstream
`ggml-org/llama.cpp`.

## The defect (DERIVED from source, confirmed in training logs)

`examples/training/finetune-lora.cpp` (line 959 at that commit):

```cpp
int64_t resume_batch = 0;
if (start_step > 0 && epoch == start_epoch) {
    resume_batch = start_step % training_batches_per_epoch;
}
... llama_opt_epoch_resume(ctx, dataset, ..., idata_split, ..., resume_batch);
```

`src/llama-context.cpp` (line 4417):

```cpp
int64_t idata = (resume_from_batch >= 0) ? resume_from_batch + 1 : 0;
```

`resume_from_batch` means "the last position already done". A fresh epoch passes `0`, which
means "position 0 is done", so the loop starts at index 1. **Dataset item 0 is never trained, and
every epoch has N - 1 optimizer updates.** There is no shuffle on this code path, so the skipped
item is always the first line of the training file. A cosine schedule sized for N steps also
ends one step early.

## How it showed up (MEASURED)

Two training runs on a 400-conversation file (380 train + 20 validation, context 128, one
epoch) each reported 380 steps. A per-micro-batch ledger (patch 0003) showed **379 train records,
numbered 2 to 380**: 379 updates on 48,512 positions, not 380 on 48,640. Both runs were discarded.

## The fix (patch 0002)

```cpp
int64_t resume_batch = -1;                                   // fresh epoch: start at 0
if (start_step > 0 && epoch == start_epoch) {
    resume_batch = (start_step % training_batches_per_epoch) - 1;  // last completed position
}
```

After the fix (MEASURED): both reruns logged 380 train records, 0 skipped conversations, the
ledger coverage check passed, and the same 11,112 supervised tokens were counted in both.
**The resume branch (`start_step > 0`) was not exercised** by any run; only its arithmetic is
modelled and tested here.

## Reproduce the logic on CPU

`src/consumer_gpu_llm/epoch_loop.py` models both versions; `tests/test_trainer_fix.py` shows
upstream covering positions 1..N-1 and the fix covering 0..N-1, and runs the ledger coverage
check (`consumer_gpu_llm.ledger`) on a synthetic ledger with and without the skip.

## Lesson

A training log that prints "step k / N" is not evidence that N items were trained. A ledger
with one line per item (position, loss, supervised token counts), checked for coverage, found
this in minutes.

The bug has not been reported upstream yet.

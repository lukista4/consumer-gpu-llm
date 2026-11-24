"""Python model of the epoch loop in the llama.cpp LoRA fine-tuning example (upstream vs patch 0002).

Upstream (fork commit 2db8b04):
  finetune-lora.cpp:   resume_batch = 0;  if resuming: resume_batch = start_step % N
  llama-context.cpp:   idata = (resume_from_batch >= 0) ? resume_from_batch + 1 : 0
So a FRESH epoch passes 0 and the loop starts at dataset position 1: position 0 is never
trained and every epoch has N - 1 updates.

Patch 0002:
  resume_batch = -1;   if resuming: resume_batch = (start_step % N) - 1
A fresh epoch starts at 0; a resume after k completed updates starts at position k.
"""
from __future__ import annotations


def loop_start(resume_from_batch: int) -> int:
    """First dataset index visited by llama_opt_epoch_resume."""
    return resume_from_batch + 1 if resume_from_batch >= 0 else 0


def resume_batch_upstream(start_step: int, n_train: int, epoch: int, start_epoch: int) -> int:
    rb = 0
    if start_step > 0 and epoch == start_epoch:
        rb = start_step % n_train
    return rb


def resume_batch_fixed(start_step: int, n_train: int, epoch: int, start_epoch: int) -> int:
    rb = -1
    if start_step > 0 and epoch == start_epoch:
        rb = (start_step % n_train) - 1
    return rb


def trained_positions(n_train: int, start_step: int = 0, fixed: bool = True,
                      epoch: int = 0, start_epoch: int = 0) -> list[int]:
    f = resume_batch_fixed if fixed else resume_batch_upstream
    return list(range(loop_start(f(start_step, n_train, epoch, start_epoch)), n_train))

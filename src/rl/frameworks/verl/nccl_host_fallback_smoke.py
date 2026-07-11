#!/usr/bin/env python3
"""Minimal two-GPU NCCL smoke for hosts without CUDA peer access.

Run with ``torchrun --standalone --nproc_per_node=2`` after exporting the intended NCCL transport
variables.  It is intentionally independent of Verl, so transport failures stay easy to diagnose.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([float(local_rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    assert value.item() == 3.0, value
    print(f"rank={local_rank} all_reduce={value.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

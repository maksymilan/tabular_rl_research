"""Two-GPU NCCL sanity check for the table_rl SQL-R1 compatibility stack."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([float(local_rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    dist.barrier()
    print(
        {
            "rank": dist.get_rank(),
            "device": local_rank,
            "all_reduce": value.item(),
            "nccl": torch.cuda.nccl.version(),
            "p2p_disabled": os.environ.get("NCCL_P2P_DISABLE"),
        },
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

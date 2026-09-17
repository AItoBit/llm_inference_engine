import torch
import torch.distributed as dist
import os

def init_tensor_parallel():
    """Initializes PyTorch distributed process group for Tensor Parallelism."""
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if torch.cuda.is_available():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        else:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        # Single GPU fallback
        pass

def get_tp_group():
    return dist.group.WORLD if dist.is_initialized() else None

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer partitioned along the output dimension. Used in Attention QKV projection."""
    pass

class RowParallelLinear(torch.nn.Module):
    """Linear layer partitioned along the input dimension. Used in Attention Output projection."""
    pass

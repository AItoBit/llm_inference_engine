import torch

class CacheEngine:
    def __init__(self, config, num_gpu_blocks: int, num_cpu_blocks: int, block_size: int = 16):
        self.config = config
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        
        # Dimensions for K and V cache blocks
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        
        # Shape: [num_blocks, num_kv_heads, block_size, head_dim]
        # (Using PagedAttention layout)
        cache_shape = (num_gpu_blocks, self.num_kv_heads, block_size, self.head_dim)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = getattr(torch, getattr(config, "dtype", "float16"), torch.float16)
        
        self.gpu_k_cache = torch.empty(cache_shape, dtype=dtype, device=device)
        self.gpu_v_cache = torch.empty(cache_shape, dtype=dtype, device=device)
        
        # Block allocator tracking
        self.free_gpu_blocks = list(range(num_gpu_blocks))
        
    def allocate_block(self):
        if not self.free_gpu_blocks:
            raise RuntimeError("Out of GPU cache blocks")
        return self.free_gpu_blocks.pop(0)
        
    def free_block(self, block_id: int):
        self.free_gpu_blocks.append(block_id)
        
    def get_num_free_blocks(self):
        return len(self.free_gpu_blocks)

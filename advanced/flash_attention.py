import torch

def paged_attention_reference(
    query: torch.Tensor, # [batch_size, num_heads, 1, head_dim]
    key_cache: torch.Tensor, # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,
    block_tables: torch.Tensor, # [batch_size, max_num_blocks]
    context_lens: torch.Tensor, # [batch_size]
    scale: float
):
    """
    Reference PyTorch implementation of PagedAttention for testing and fallback.
    In production, this would be replaced by highly optimized Triton/CUDA kernels
    like those found in vLLM.
    """
    batch_size, num_heads, _, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    num_queries_per_kv = num_heads // num_kv_heads
    block_size = key_cache.shape[2]
    
    out = torch.empty_like(query)
    
    for i in range(batch_size):
        seq_len = context_lens[i].item()
        if seq_len == 0:
            continue
            
        # Reconstruct the continuous KV for this sequence from the block table
        blocks = block_tables[i]
        
        # Grab physical blocks and concatenate [num_kv_heads, num_blocks * block_size, head_dim]
        k_seq = torch.cat([key_cache[b] for b in blocks], dim=1)[:, :seq_len, :]
        v_seq = torch.cat([value_cache[b] for b in blocks], dim=1)[:, :seq_len, :]
        
        # Repeat KV for GQA
        if num_queries_per_kv > 1:
            k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=0)
            v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=0)
            
        q_i = query[i].squeeze(1) # [num_heads, head_dim]
        
        # Scaled Dot-Product Attention
        attn_weights = torch.einsum("hd,hsd->hs", q_i, k_seq) * scale
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum("hs,hsd->hd", attn_probs, v_seq)
        
        out[i, :, 0, :] = attn_output
        
    return out

class FlashAttentionWrapper:
    @staticmethod
    def forward(q, k, v, is_causal=True, attention_mask=None):
        """
        PyTorch 2.0 SDPA automatically dispatches to FlashAttention-2 if available.
        Provides vast speedups for non-paged continuous batching prefill phase.
        """
        return torch.nn.functional.scaled_dot_product_attention(
             q, k, v, is_causal=is_causal, attn_mask=attention_mask
        )

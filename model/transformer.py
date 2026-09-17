import torch
import torch.nn as nn
from .attention import LlamaAttention
from .rope import LlamaRotaryEmbedding

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids, cos, sin, past_key_value=None, attention_mask=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config.hidden_size // config.num_attention_heads, config.max_position_embeddings)
        
    def forward(self, input_ids, position_ids=None, past_key_values=None, attention_mask=None, use_cache=False):
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
            if past_key_values is not None:
                position_ids = position_ids + past_key_values[0][0].shape[2]
                
        hidden_states = self.embed_tokens(input_ids)
        
        kv_seq_len = seq_len + (past_key_values[0][0].shape[2] if past_key_values else 0)
        cos, sin = self.rotary_emb(position_ids, seq_len=kv_seq_len)
        
        next_decoder_cache = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = layer(
                hidden_states, 
                position_ids, 
                cos, 
                sin, 
                past_key_value=past_key_value, 
                attention_mask=attention_mask, 
                use_cache=use_cache
            )
            if use_cache:
                next_decoder_cache += (present_key_value,)
                
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, position_ids=None, past_key_values=None, attention_mask=None, use_cache=False):
        hidden_states, next_decoder_cache = self.model(
            input_ids, position_ids, past_key_values, attention_mask, use_cache
        )
        logits = self.lm_head(hidden_states)
        return logits, next_decoder_cache

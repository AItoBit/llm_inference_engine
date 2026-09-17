from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    intermediate_size: int = 11008
    dtype: str = "float16"

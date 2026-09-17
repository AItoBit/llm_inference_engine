import torch
import torch.nn as nn

class W8A16Linear(nn.Module):
    """Linear layer representing INT8 weights and FP16 activations."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weights in localized INT8 chunks
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty((out_features,), dtype=torch.float16))
        
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None
            
    def compute_scaling_factors(self, float_weight: torch.Tensor):
        """Quantizes an incoming FP16 vector to INT8."""
        abs_max = torch.amax(torch.abs(float_weight), dim=1)
        scale = abs_max / 127.0
        scale = torch.clamp(scale, min=1e-5)
        
        quantized = torch.clamp(torch.round(float_weight / scale.unsqueeze(1)), -127, 127).to(torch.int8)
        
        self.weight.copy_(quantized)
        self.weight_scale.copy_(scale)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On-the-fly dequantization. In an ideal Triton engine this is fused 
        # into the matmul kernel to avoid memory bandwidth bottlenecks.
        dequant_weight = self.weight.to(x.dtype) * self.weight_scale.unsqueeze(1)
        return torch.nn.functional.linear(x, dequant_weight, bias=self.bias)

import torch
from safetensors.torch import load_file
import os
from typing import Dict

class CheckpointLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
    def load_weights(self) -> Dict[str, torch.Tensor]:
        """Loads weights from a safetensors file or directory of safetensors."""
        tensors = {}
        if os.path.isdir(self.model_path):
            files = [f for f in os.listdir(self.model_path) if f.endswith(".safetensors")]
            if not files:
                 # Note: in a real implementation we could fall back to torch load if bin
                 raise ValueError(f"No .safetensors files found in {self.model_path}")
            for file in files:
                file_path = os.path.join(self.model_path, file)
                tensors.update(load_file(file_path))
        elif self.model_path.endswith(".safetensors"):
             tensors.update(load_file(self.model_path))
        else:
            raise ValueError("model_path must be a safetensors file or a directory containing safetensors files")
            
        return tensors

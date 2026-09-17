from transformers import AutoTokenizer
from typing import List, Union

class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # Set pad token if not present to avoid errors during batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def encode(self, text: Union[str, List[str]], **kwargs) -> Union[List[int], List[List[int]]]:
        """Encodes string or list of strings to token ids."""
        return self.tokenizer(text, **kwargs).input_ids
        
    def decode(self, token_ids: Union[List[int], List[List[int]]], **kwargs) -> Union[str, List[str]]:
        """Decodes token ids to string or list of strings."""
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            return self.tokenizer.batch_decode(token_ids, **kwargs)
        return self.tokenizer.decode(token_ids, **kwargs)
        
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

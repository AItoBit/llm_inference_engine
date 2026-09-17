from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .kv_cache import CacheEngine

@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    status: str = "WAITING" # WAITING, RUNNING, FINISHED
    block_table: List[int] = field(default_factory=list)
    
    def get_len(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

class Scheduler:
    def __init__(self, cache_engine: CacheEngine, max_num_seqs: int = 256):
        self.cache_engine = cache_engine
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.block_size = cache_engine.block_size
        self.max_num_seqs = max_num_seqs
        
    def add_sequence(self, seq: Sequence):
        self.waiting.append(seq)
        
    def step(self):
        """Schedules sequences for the next forward pass."""
        preempted = []
        
        # 1. Manage running sequences
        for seq in list(self.running):
            # If generating a new token requires a new block
            if seq.get_len() % self.block_size == 0 and seq.get_len() > 0:
                if self.cache_engine.get_num_free_blocks() > 0:
                    new_block = self.cache_engine.allocate_block()
                    seq.block_table.append(new_block)
                else:
                    preempted.append(seq)
                    
        # 2. Preempt (evict to waiting and free blocks - naive approach)
        for seq in preempted:
            self.running.remove(seq)
            self.waiting.insert(0, seq) # put it at the front of waiting queue
            for b in seq.block_table:
                self.cache_engine.free_block(b)
            seq.block_table.clear()
            seq.status = "WAITING"
             
        # 3. Schedule waiting sequences
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            num_blocks_needed = (seq.get_len() + self.block_size - 1) // self.block_size
            if self.cache_engine.get_num_free_blocks() >= num_blocks_needed:
                self.waiting.pop(0)
                for _ in range(num_blocks_needed):
                    seq.block_table.append(self.cache_engine.allocate_block())
                seq.status = "RUNNING"
                self.running.append(seq)
            else:
                # Can't fit this seq, stop scheduling
                break
                
        return self.running

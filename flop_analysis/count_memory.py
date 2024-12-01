import os
import torch
import numpy as np
from model import Model
from fvcore.nn import FlopCountAnalysis
import time


m = Model().cuda()
chunk_size = [4,8,16,32,64,128,256,512]
n_cols = 512 #which are fired by the spm

def get_stats(n_cols, chunk_size = 64):
    n_fwds = n_cols // chunk_size
    total_memory = 0
    
    for i in range(n_fwds):    
        with torch.no_grad():
            with torch.cuda.device('cuda'):
                torch.cuda.reset_peak_memory_stats()
                x = torch.randn(1, chunk_size, 2048).cuda()
                output = m(x)
                memory_usage_bytes = torch.cuda.max_memory_allocated('cuda')#memory_stats["allocated_bytes.all.current"]
                memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # Convert bytes to MB
                total_memory += memory_usage_mb
    return total_memory

total_memory = get_stats(n_cols, chunk_size[-1])
for chunk in chunk_size:
    memory_usage = get_stats(n_cols, chunk)
    print("Chunk size:", chunk, "Memory usage:", memory_usage, "MB")
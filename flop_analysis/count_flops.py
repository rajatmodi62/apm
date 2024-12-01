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
    flop_count = 0
    total_time = 0
    with torch.no_grad():
        for i in range(n_fwds):
            
            x = torch.randn(1, chunk_size, 2048).cuda()
            print("doing", i, "of", n_fwds, x.shape)
            tic = time.time()
            flop_iter = FlopCountAnalysis(m, x).total()
            flop_count += flop_iter
            toc = time.time()
            total_time += toc - tic
    return flop_count, total_time, flop_iter

data = {}
for chunk_size in chunk_size:
    flop_count, total_time, flop_iter = get_stats(n_cols, chunk_size)
    data[chunk_size] = {"flop_count": flop_count/ (10 ** 9), "total_time": total_time, "flop_iter": flop_iter/ (10 ** 9)}
print(data)
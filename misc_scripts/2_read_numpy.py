#author: rmodi
import os 
import numpy as np 
from  einops import rearrange, reduce, repeat
from sklearn.manifold import TSNE


x = np.load('/home/rmodi/krishna/hinton/morphogenesis/data/coco/features/val2017/000000000139.npy')
x = TSNE(n_components=3, learning_rate='auto',
                 init='random', perplexity=3).fit_transform(x)

print(x.shape)
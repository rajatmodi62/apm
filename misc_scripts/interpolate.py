#author: rmodi 
# takes two images (coco val), estimates their summarized column vector. 
# interpolates linearly between two summary vectors.
# resultant vector acts as a trigger seed: should help in yielding new iimages  
# should be evidence of the  learnt generative latent space
# therefore a single architecture is generative, and predicted columns can be used for any downstream discriminative task  
# however, i dont expect much changes in the image since the latent space is too sparse 

# to do: make nerf predict rgb also. i failed brilliantly earlier lol 


#author: rmodi
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob 
from pathlib import Path 
import cv2
from einops import rearrange, reduce, repeat
from dataloader import PerceptionFieldDataset
from model import Model
from sklearn.manifold import TSNE
import time

if __name__ == '__main__':
    h,w = 32,32
    d = 1024 
    patch_size = 14
    fwd_chunk_size = 32 
    batch_size = 1
    num_workers = 8
    num_epochs = 100000
    lr = 0.0001
    device = 'cuda'
    n_interpolations = 10
    save_dir = Path('../interpolations')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    model_save_path = Path('checkpoints/model_15.pth')
    
    p1 = './lecunn.png'
    x1 = cv2.imread(p1)
    x1 = cv2.resize(x1, (448,448))
    # x1 = 0*x1 
    # x1[:10,:10] = 250
    orig_x1 = cv2.resize(x1,(32,32))
    min = np.min(x1)
    max = np.max(x1)
    x1 = (x1 - min) / (max - min) #do min max scaling
    x1 = torch.from_numpy(x1).permute(2,0,1).float()
    x1 = x1.unsqueeze(0)
    x1 = x1.to(device)
    
    p2 = './bengio.png'
    x2 = cv2.imread(p2)
    x2 = cv2.resize(x2, (448,448))
    # x2 = 0*x2 
    # x2[-10:,-10:] = 250
    orig_x2 = cv2.resize(x2,(32,32))
    min = np.min(x2)
    max = np.max(x2)
    x2 = (x2 - min) / (max - min) #do min max scaling
    x2 = torch.from_numpy(x2).permute(2,0,1).float()
    x2 = x2.unsqueeze(0)
    x2 = x2.to(device)
    
    #build model
    model = Model(hidden_dim = d, h = h, w = w, fwd_chunk_size = fwd_chunk_size).to(device)
    
    #load model
    model.load_state_dict(torch.load(model_save_path), strict = True)
    print("loaded model")
    
    output_feats,output_rgbs = model.interpolate_function(x1,x2,n_interpolations)
    print("output", len(output_feats))
    
    
    ######## predicting feature islands of hinton###################
    # for done, feat_out in enumerate(output_feats):
    #     print("done", done,feat_out.shape)
    #     x = TSNE(n_components=3, learning_rate='auto',
    #              init='random', perplexity=3).fit_transform(feat_out)
    #     print("x shape", x.shape)
    #     x = rearrange(x, ' (h w) c -> h w c', h = h, w = w)
    #     # x = x.detach().cpu().numpy()
    #     x = (x - np.min(x)) / (np.max(x) - np.min(x))
    #     x = x * 255
    #     x = x.astype(np.uint8)
        
    #     x = np.concatenate([orig_x1,orig_x2,x],axis=1)
    #     save_path = str(save_dir/'{}.png'.format(str(done).zfill(3)))
    #     cv2.imwrite(save_path, x)
    #     time.sleep(2)
    
    ##################################################################
    
    
    
    
    
    ######### rgb readout from the distributed memory ################
    
    for done, rgb_out in enumerate(output_rgbs):
        print("done", done,rgb_out.shape)
        print("rgb_shape", rgb_out.shape)
        rgb_out = rgb_out.cpu().detach().numpy()
        rgb_out = (rgb_out - np.min(rgb_out)) / (np.max(rgb_out) - np.min(rgb_out))
        rgb_out = rearrange(rgb_out, ' (h w) c -> h w c', h = h, w = w)
        # exit(1)
        rgb_out = rgb_out * 255
        rgb_out = rgb_out.astype(np.uint8)
        
        print("decode shape", orig_x1.shape, orig_x2.shape, rgb_out.shape)
        
        
        #x = np.concatenate([orig_x1,orig_x2,rgb_out],axis=1)
        x = rgb_out
        save_path = str(save_dir/'{}.png'.format(str(done).zfill(3)))
        cv2.imwrite(save_path, x)
        time.sleep(2)
    
    
    
    
    
    
    
    
    
    
    
    
    ##################################################################

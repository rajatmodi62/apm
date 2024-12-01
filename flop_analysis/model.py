import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob 
from pathlib import Path 
import cv2
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch.nn as nn
import math 


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.cuda()


class Model(nn.Module):
    # h,w of the image which will be fwd pass. 
    # coordinate based query 
    def __init__(self, hidden_dim = 1024, h = 32, w = 32, fwd_chunk_size = 16):
        
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(2*hidden_dim, 4096)
        
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048,1024)
        
        self.feat_proj_head = nn.Linear(1024, 768)
        # self.rgb_head_1 = nn.Linear(2816, 256)
        # self.rgb_head_2 = nn.Linear(256,256)
        # self.rgb_head_3 = nn.Linear(256,3)

        #initialize positional encoding 
        self.pos = positionalencoding2d(hidden_dim, h,w) #to break input coordinate symmetry
        self.h, self.w = h,w
        self.fwd_chunk_size = fwd_chunk_size
        
        #init a single patch size 
        #will operate on 448 by 448 to get information into the columns
        self.patch_size = 14
        self.stride = 14
        self.conv1 = nn.Conv2d(3, 1, kernel_size=self.patch_size, stride=self.stride)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.little_self_attention = nn.MultiheadAttention(768, 8, batch_first=True) 
        self.little_query = nn.Parameter(torch.randn(768))
        self.pos_output = positionalencoding2d(768, h,w) #loving attention at the output of the cortical stack.
        
    def forward(self, x):
        x_pos = x#contains the whole cortical column stack
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        
        feat = self.feat_proj_head(x)
       
        # rgb = F.relu(self.rgb_head_1(torch.cat([feat,x_pos],1))) #breaks rgb output symmetry
        # rgb = F.relu(self.rgb_head_2(rgb))
        # rgb = F.relu(self.rgb_head_3(rgb))
        return feat#, rgb
        
        
    def forward_chunk(self, x):
        
        # print("slimmed down net..")
        x_pos = x#contains the whole cortical column stack
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        
        feat = self.feat_proj_head(x)
       
        rgb = F.relu(self.rgb_head_1(torch.cat([feat,x_pos],1))) #breaks rgb output symmetry
        rgb = F.relu(self.rgb_head_2(rgb))
        rgb = F.relu(self.rgb_head_3(rgb))
        
        
         
        return feat, rgb # feat is the feature which is produced
    
    #x : b,c,h,w
    # feat: b (h w) d
    
    def forward_wrapper(self,x, x_avg, feat):
        x,x_avg,feat = x.cuda(), x_avg.cuda(),feat.cuda()
        
        b,c,h,w = x.shape
        
        #trigger the copying of the average latent feature like dna 
        summary_feat = self.conv1(x)
        summary_feat = rearrange(summary_feat, 'b c h w -> b (h w) c')#still need to resolve symmetry between locations in the cortical column
        summary_feat = summary_feat.squeeze(-1)
        summary_feat = repeat(summary_feat, 'b d -> b d h w', h = self.h, w = self.w) #squeezed the perceptual information into the column
        
        
        
        pos = self.pos  #d h w
        pos = repeat(pos, 'd h w -> b d h w', b = b)
        
        input_feat = torch.cat([summary_feat, pos], dim=1) #along d dimension, break identity symmetry at same input location
        
        #batchify the entire forward pass 
        input_feat = rearrange(input_feat, 'b d h w -> (b h w) d')
        
        target_feat= feat
        target_rgb = rearrange(x_avg, 'b c h w -> (b h w) c')
        
        chunk_size = self.fwd_chunk_size
        n_chunks = input_feat.shape[0] // chunk_size
        if input_feat.shape[0] % chunk_size != 0:
            n_chunks += 1
        n_forwards = 0
        for i in range(n_chunks):
            
            start = i*chunk_size
            end = min((i+1)*chunk_size, input_feat.shape[0])
            input_feat_chunk = input_feat[start:end]
            print("forwarding through the model", input_feat_chunk.shape)
            feat_chunk, rgb_chunk = self.forward_chunk(input_feat_chunk)
            
            n_forwards+=1
            if i == 0:
                feat_out = feat_chunk
                rgb_out = rgb_chunk
            else:
                feat_out = torch.cat([feat_out, feat_chunk], dim=0)
                rgb_out = torch.cat([rgb_out, rgb_chunk], dim=0)
        
        
        feat_out = rearrange(feat_out, '(b h w) d -> b h w d', b = b, h = self.h, w = self.w)
        orig_feat_out = feat_out
        
        
        pos_out = repeat(self.pos_output, 'd h w -> b d h w', b = b)
        pos_out = rearrange(pos_out, 'b d h w -> b h w d')
        feat_out = feat_out + pos_out
        feat_out = rearrange(feat_out ,'b h w d -> b (h w) d')#(N,L,E q ​ ) when batch_first=True
        
        query = repeat(self.little_query, 'd -> b  d', b = b, )
        query = query.unsqueeze(1) #need one query only

        
        feat_out, _ = self.little_self_attention(query, feat_out, feat_out)
        
        feat_out = feat_out.squeeze(1)
        
        feat_loss = F.mse_loss(feat_out, target_feat)
        rgb_loss = 0
        
        loss = feat_loss + rgb_loss
        
        rgb_out = rearrange(rgb_out, '(b h w) c -> b c h w', b = b, h = self.h, w = self.w)
        
        return loss, feat_loss, rgb_loss, feat_out, rgb_out,orig_feat_out

    
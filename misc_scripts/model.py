#author: rmodi
#i thank geoff hinton for his glom paper, 
# i thank all the amazing people who invented cnns, transformers, and neural fields. 
#i thank the supreme god, for revealing to me the secrets of perception fields.
import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
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
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048,1024)
        
        self.feat_proj_head = nn.Linear(1024, 1024)
        self.rgb_head_1 = nn.Linear(1024*3, 256)
        self.rgb_head_2 = nn.Linear(256,256)
        self.rgb_head_3 = nn.Linear(256,3)

        #initialize positional encoding 
        self.pos = positionalencoding2d(hidden_dim, h,w) #to break input coordinate symmetry
        self.h, self.w = h,w
        self.fwd_chunk_size = fwd_chunk_size
        
        #init a single patch size 
        #will operate on 448 by 448 to get information into the columns
        self.patch_size = 14
        self.stride = 14
        self.conv1 = nn.Conv2d(3, 1, kernel_size=self.patch_size, stride=self.stride)
        
    def forward_chunk(self, x):
        print("this shit is mlp kevin not gonna believe it!!!!")
        x_pos = x#contains the whole cortical column stack
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        
        feat = self.feat_proj_head(x)
        print("believe it this is actual feature forward, i am a simple mlp!!!!!")
        # exit(1)
        # print("head on feat")
        # print(self.pos.shape, feat.shape)
        rgb = F.relu(self.rgb_head_1(torch.cat([feat,x_pos],1))) #breaks rgb output symmetry
        rgb = F.relu(self.rgb_head_2(rgb))
        rgb = F.relu(self.rgb_head_3(rgb))
        
        # rgb= None   
        return feat, rgb # feat is the feature which is produced
    
    #x : b,c,h,w
    # feat: b (h w) d
    
    def forward_wrapper(self,x, x_avg, feat):
        x,x_avg,feat = x.cuda(), x_avg.cuda(),feat.cuda()
        # feat = torch.zeros_like(feat)
        # print("set feat 0")
        b,c,h,w = x.shape
        feat = rearrange(feat, 'b (h w) d -> b d h w', h = self.h, w = self.w)
        
        #trigger the copying of the average latent feature like dna 
        summary_feat = self.conv1(x)
        summary_feat = rearrange(summary_feat, 'b c h w -> b (h w) c')#still need to resolve symmetry between locations in the cortical column
        summary_feat = summary_feat.squeeze(-1)
        summary_feat = repeat(summary_feat, 'b d -> b d h w', h = self.h, w = self.w) #squeezed the perceptual information into the column
        
        # print("summary feat", feat.shape, summary_feat.shape)
        
        pos = self.pos  #d h w
        pos = repeat(pos, 'd h w -> b d h w', b = b)
        
        input_feat = torch.cat([summary_feat, pos], dim=1) #along d dimension, break identity symmetry at same input location
        
        #batchify the entire forward pass 
        input_feat = rearrange(input_feat, 'b d h w -> (b h w) d')
        target_feat = rearrange(feat, 'b d h w -> (b h w) d')
        target_rgb = rearrange(x_avg, 'b c h w -> (b h w) c')
        
        chunk_size = self.fwd_chunk_size
        n_chunks = input_feat.shape[0] // chunk_size
        if input_feat.shape[0] % chunk_size != 0:
            n_chunks += 1
        n_forwards = 0
        for i in range(n_chunks):
            # print("chunk", i, "/", n_chunks)
            start = i*chunk_size
            end = min((i+1)*chunk_size, input_feat.shape[0])
            input_feat_chunk = input_feat[start:end]
            # target_feat_chunk = target_feat[start:end]
            # print("device", input_feat_chunk.device)
            feat_chunk, rgb_chunk = self.forward_chunk(input_feat_chunk)
            # rgb_out = None
            n_forwards+=1
            if i == 0:
                feat_out = feat_chunk
                rgb_out = rgb_chunk
            else:
                feat_out = torch.cat([feat_out, feat_chunk], dim=0)
                rgb_out = torch.cat([rgb_out, rgb_chunk], dim=0)
        # print("feat out", feat_out.shape, target_feat.shape)
        # print("rgb out", rgb_out.shape, target_rgb.shape)
        # print("no of forwards", n_forwards)
        # exit(1)
        feat_loss = F.mse_loss(feat_out, target_feat)
        # rgb_loss = 0
        rgb_loss = F.mse_loss(rgb_out, target_rgb)
        loss = feat_loss + rgb_loss
        # print("loss", loss)
        # target_rgb = rearrange(target_rgb, '(b h w) c -> b c h w', b = b, h = self.h, w = self.w)
        rgb_out = rearrange(rgb_out, '(b h w) c -> b c h w', b = b, h = self.h, w = self.w)
        # rgb_out = None
        # print("in model")
        return loss, feat_loss, rgb_loss, feat_out, rgb_out

    
    def interpolate_function(self,x1,x2, n_interpolations = 100):
        # no gradient flow required
        with torch.no_grad():
            x1,x2 = x1.cuda(), x2.cuda()
            b,c,h,w = x1.shape
            
            # f1,f2 are summary vectors
            f1,f2 = self.conv1(x1), self.conv1(x2)
            f1 = rearrange(f1,'b c h w -> b (h w) c')
            f2 = rearrange(f2,'b c h w -> b (h w) c')
            f1 = f1.squeeze(-1) #single col vector
            f2 = f2.squeeze(-1) #single col vector 
            
            pos = self.pos  #d h w
            pos = repeat(pos, 'd h w -> b d h w', b = b)
            
            #generate n_interpolations between f1 and f2
            interpolation_vectors = []
            for i in range(n_interpolations):
                interpolation_vectors.append(f1 + (f2 - f1) * i / n_interpolations)
            
            output_feats = []
            output_rgbs = []
            
            for i in range(n_interpolations):
                print("interpolation", i, "/", n_interpolations)
                summary_vector = interpolation_vectors[i]
                print("interpolation shape", summary_vector.shape)
                summary_vector = repeat(summary_vector, 'b d -> b d h w', h = self.h, w = self.w) #repeat column vector to all locations 
                summary_vector = summary_vector.cuda()
                input_feat = torch.cat([summary_vector, pos], dim=1) #along d dimension
                #batchify the entire forward pass 
                input_feat = rearrange(input_feat, 'b d h w -> (b h w) d')
                
                chunk_size = self.fwd_chunk_size
                n_chunks = input_feat.shape[0] // chunk_size
                if input_feat.shape[0] % chunk_size != 0:
                    n_chunks += 1
                for j in range(n_chunks):
                    # print("chunk", i, "/", n_chunks)
                    start = j*chunk_size
                    end = min((j+1)*chunk_size, input_feat.shape[0])
                    input_feat_chunk = input_feat[start:end]
                    feat_chunk, rgb_chunk = self.forward_chunk(input_feat_chunk)
                    #rgb is not needed right now 
                    #rgb_out = None
                    if j == 0:
                        feat_out = feat_chunk
                        rgb_out = rgb_chunk
                    else:
                        feat_out = torch.cat([feat_out, feat_chunk], dim=0)
                        rgb_out = torch.cat([rgb_out, rgb_chunk], dim=0)
                        
                feat_out  = feat_out.cpu().detach().numpy()
                output_feats.append(feat_out)
                output_rgbs.append(rgb_out)    
                feat_out = None #next vector interpolation happens here
                rgb_out = None
        
        print("getting ready to return....")
        # exit(1)
        return output_feats,output_rgbs
    
    def predict_image(self, x):
        x = x.cuda()
        b,c,h,w = x.shape
        print("x",x.shape)
        # exit(1)
        #trigger the copying of the average latent feature like dna 
        summary_feat = self.conv1(x)
        summary_feat = rearrange(summary_feat, 'b c h w -> b (h w) c')
        summary_feat = summary_feat.squeeze(-1)
        summary_feat = repeat(summary_feat, 'b d -> b d h w', h = self.h, w = self.w)
        
        pos = self.pos
        pos = repeat(pos, 'd h w -> b d h w', b = b)
        
        input_feat = torch.cat([summary_feat, pos], dim=1)
        input_feat = rearrange(input_feat, 'b d h w -> (b h w) d')
        
        chunk_size = self.fwd_chunk_size
        n_chunks = input_feat.shape[0] // chunk_size
        if input_feat.shape[0] % chunk_size != 0:
            n_chunks += 1
        n_forwards =0 
        for i in range(n_chunks):
            start = i*chunk_size
            end = min((i+1)*chunk_size, input_feat.shape[0])
            input_feat_chunk = input_feat[start:end]
            feat_chunk, rgb_chunk = self.forward_chunk(input_feat_chunk)
            # rgb_out = None
            if i == 0:
                feat_out = feat_chunk
                rgb_out = rgb_chunk
            else:
                feat_out = torch.cat([feat_out, feat_chunk], dim=0)
                rgb_out = torch.cat([rgb_out, rgb_chunk], dim=0)
            n_forwards+=1
        print("no of forwards", n_forwards)
        # exit(1)
        feat_out  = feat_out.cpu().detach().numpy()
        rgb_out = rgb_out.cpu().detach().numpy()
        return feat_out,rgb_out
    
if __name__ == '__main__':
    model = Model().cuda()
    x = torch.randn(4,3,448,448)
    x_avg = torch.randn(4,3,32,32)
    feat = torch.randn(4,1024,1024)
    loss, feat_loss, rgb_loss, feat_out, rgb_out = model.forward_wrapper(x,x_avg, feat)
    print(feat_loss, rgb_loss,feat_out.shape, rgb_out.shape)
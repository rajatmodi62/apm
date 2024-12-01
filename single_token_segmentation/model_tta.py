# i am forever humbled and grateful to geoff hinton for sharing his glom and forward forward paper with all of us.
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

from model import Model
import clip
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ModelTTA(nn.Module):
    def __init__(self, hidden_dim = 1024, h = 32, w = 32, fwd_chunk_size = 16, num_classes= 10, backbone_path = None):
        
        super(ModelTTA, self).__init__()

        device = 'cuda'
        #load the clip model 
        self.clip_teacher, self.preprocess = clip.load('ViT-L/14', device)

        
        print("init model")
        self.backbone = Model(hidden_dim = hidden_dim, h = h, w = w, fwd_chunk_size = fwd_chunk_size)
        print("loaded backbone")
        print("loaded classification")
        self.h = h
        self.w = w
        self.num_classes = num_classes
        
        
        
    def forward(self, img,cls_id, image_features):#, x, x_avg, feat):
        x = transform(img)
        # print("x is", x.shape)
        x = repeat(x, 'c h w -> b c h w', b = 8) #prepare for forward passes. 
        x_avg = F.avg_pool2d(x, kernel_size=(14,14), stride=(14,14))
        
        
        #forward pass through backbone, and force it to align
        loss, feat_loss, rgb_loss, feat_out, rgb_out,orig_feat_out = self.backbone.forward_wrapper(x, x_avg, image_features) #image features of clip are target for our model
        
        return loss, feat_out,orig_feat_out
        
        
if __name__ == '__main__':
      
    # print("testing dataset", clip.available_models())
    # exit(1)
    from torchvision.datasets import CIFAR100
    import os
    dataset = CIFAR100(root=os.path.expanduser("~/data"), download=True, train=False)
    model = ModelTTA().cuda()
    for idx in range(len(dataset)):

        print("doing", idx+1, "of", len(dataset))
        image, class_id = dataset[idx]
        loss, status = model(image,class_id, dataset,)
        # print("image is", image.shape)
        print("loss is", loss,status)
        # exit(1)
    # x = torch.randn(4,3,448,448)
    # x_avg = torch.randn(4,3,32,32)
    # feat = torch.randn(4,1024)
    # model(x,x_avg, feat)
        
    
    
    

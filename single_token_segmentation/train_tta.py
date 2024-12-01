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
from torchvision import transforms
import torchvision
from model_tta import ModelTTA
import clip
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from sklearn.manifold import TSNE


h,w = 32,32
d = 1024 
n_iters_tta = 1200 #number of iterations for test time training for each sample
n_samples_fwd = 4
patch_size = 14
fwd_chunk_size = 512
batch_size = 32*2
num_workers = 8
num_epochs = 15
lr = 1e-4
device = 'cuda'
train_print_freq = 50
is_train = False
target_trues = 1200

clip_teacher, preprocess = clip.load('ViT-L/14', device)

def randomize_weights_student(model):
    for param in model.parameters():
        print("randomizing!!!")
        nn.init.normal_(param.data, mean=0, std=0.01)  # Initialize with random values from a normal distribution


model_tta = ModelTTA(hidden_dim = 1024, h = 32, w = 32, fwd_chunk_size = 16, num_classes= 10, backbone_path = None).cuda()
dataset = torchvision.datasets.CIFAR10(root = '~/data', train=is_train, download=True, transform=None)

import torch.optim.lr_scheduler as lr_scheduler

optimizer = torch.optim.Adam(model_tta.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

correct,n_preds =0,0



def dump_island(orig_feat_out, i, iter):
    orig_feat_out = orig_feat_out.detach().cpu().numpy()
    orig_feat_out = orig_feat_out[0]
    feat_out = rearrange(orig_feat_out, 'h w d -> (h w) d')
    feat = TSNE(n_components=1, learning_rate='auto',
                 init='random', perplexity=3).fit_transform(feat_out)
    feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
    feat = feat.squeeze(-1)
    feat = rearrange(feat, '(h w)  -> h w ', h = 32, w = 32)
    
    feat = repeat(feat, 'h w -> h w c', c = 3)
    feat = feat * 255
    feat = feat.astype(np.uint8)
    feat = np.concatenate([orig_img, feat], axis = 1)
    save_str = '{}_{}.png'.format(i, iter)
    cv2.imwrite(save_str, feat)
    print("write done")
idx = 200
for i in range(idx, len(dataset)):
    # adapt the model everytime
    print("discarded backbone and reloaded weights from scratch")
    randomize_weights_student(model_tta.backbone)
    print("randomized")
    n_trues=0
    is_last_True = False
    # exit(1)
    img, label = dataset[i]
    orig_img = np.array(img)
    
    # print("image shape", orig_img.shape)
    # exit(1)
    optimizer.zero_grad()
    was_correctly_predicted = False  
    break_count =0  
    for iter in range(n_iters_tta):
        
        print("iter", iter, '/', n_iters_tta)
        image_input = preprocess(img).unsqueeze(0).to('cuda')
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to('cuda')
        with autocast():
            with torch.no_grad():
                image_features = clip_teacher.encode_image(image_input)
                text_features = clip_teacher.encode_text(text_inputs)
        
        
            image_features = repeat(image_features, 'b d-> b b1 d', b1 = 8)
            image_features = rearrange(image_features, 'b b1 d -> (b b1) d')
        
        # with autocast():
            loss, feat_out,orig_feat_out = model_tta(img, label, image_features)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # loss.backward()
            # optimizer.step()
            print("loss scaled", loss)
            
            image_features = feat_out[0]
            # text_features = text_features.to(torch.float32)
            # image_features = image_features.to(torch.float32)
            image_features = image_features.unsqueeze(0)
            
            # print("image features", image_features.shape, text_features.shape, image_features.dtype, text_features.dtype)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
            index = indices[0]
        
        
        if iter%50==0:
            print("going for routing")
            dump_island(orig_feat_out, i, iter)
    
    



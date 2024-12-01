#author: rmodi
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

class PerceptionFieldDataset(Dataset):
    def __init__(self,\
            data_root = './data/coco/features',\
            split = 'train2017',
            patch_size=14):
        
        self.data_root = Path(data_root)/split
        self.img_paths = sorted(glob.glob(str(Path(data_root)/split/'*.jpg'), recursive = True))
        self.patch_size = patch_size 
        self.stride = self.patch_size
        print("read", len(self.img_paths), "images")
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        min = np.min(img)
        max = np.max(img)
        img = (img - min) / (max - min) #do min max scaling 
        x = torch.from_numpy(img).permute(2,0,1).float()
        
        
        # print("x shape", x.shape,torch.min(x), torch.max(x))
        #average reconstruction feature 
        avg_feat = x
        x_avg_filtered = F.avg_pool2d(avg_feat, kernel_size=(self.patch_size, self.patch_size), stride=(self.stride, self.stride))
        x_avg_filtered = x_avg_filtered.squeeze(0)
        
        #rearrange and plot
        # x_avg_filtered = rearrange(x_avg_filtered, 'c h w -> h w c')
        # x_avg_filtered = x_avg_filtered.cpu().numpy()
        # cv2.imwrite("avg_feat.png", np.uint8(x_avg_filtered*255))
        
        # dino feat reconstructed
        feat_path = img_path.split('/')[-1].split('.')[0] + '.npy'
        feat_path = str(self.data_root/feat_path)
        feat = np.load(feat_path)

        # print("feat shape", x.shape, feat.shape)
        # print("name", img_path)
        return x, x_avg_filtered, feat
    
    
if __name__ == '__main__':
    dataset = PerceptionFieldDataset(split = 'val2017',\
                data_root = '../data/cluster_folder')

    dataset[1]
    
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    
    # for done, batch in enumerate(dataloader):
    #     print("done", done)
    #     print(batch[0].shape, batch[1].shape)
    #     break
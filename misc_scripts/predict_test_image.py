#author: rmodi 
# performs prediction on any test image

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
import copy

from sklearn.manifold import TSNE


if __name__ == '__main__':
    
    h,w = 32,32
    d = 1024 
    patch_size = 14
    fwd_chunk_size = 1 
    batch_size = 1
    num_workers = 8
    num_epochs = 100000
    lr = 0.0001
    device = 'cuda'
    
    save_path = Path('./checkpoints')
    save_path.mkdir(parents=True, exist_ok=True)
    model_save_path = Path('checkpoints/model_15.pth')

    # img_path = 'cat.jpeg'
    # img_path = 'interpolation_images/yogesh_rawat.jpg'
    # img_path = 'ruk.jpeg'
    img_path = 'geoff.jpeg'
    img = cv2.imread(img_path)
    print("read")
    # exit(1)
    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (448,448))
    orig_img = copy.deepcopy(img)
    min = np.min(img)
    max = np.max(img)
    img = (img - min) / (max - min) #do min max scaling 
    x = torch.from_numpy(img).permute(2,0,1).float()
    x = x.unsqueeze(0)
    x = x.to(device)
    print("x shape", x.shape)
    
    #build model
    model = Model(hidden_dim = d, h = h, w = w, fwd_chunk_size = fwd_chunk_size).to(device)
    
    #load model
    model.load_state_dict(torch.load(model_save_path), strict = True)
    print("loaded model")
    
    #perform prediction
    model.eval()
    #using the mechanism i invented.  
    with torch.no_grad():
        feat_out,rgb_out = model.predict_image(x)
        print("feat out", feat_out.shape)
    x = TSNE(n_components=3, learning_rate='auto',
                 init='random', perplexity=3).fit_transform(feat_out)
        


    x = rearrange(x, '(h w) c -> h w c', h = 32, w = 32)

    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x * 255
    x = x.astype(np.uint8)
        # cv2.imwrite('featout.png', x)
        #print("orig_imf", orig_img.shape, x.shape)
    print("rmodi...")
    
    #rgb output from the model 
    # rgb_out = rgb_out.squeeze(0).cpu().detach().numpy()
    rgb_out = (rgb_out - np.min(rgb_out)) / (np.max(rgb_out) - np.min(rgb_out))
    rgb_out = rgb_out * 255
    rgb_out = rearrange(rgb_out, '(h w) c -> h w c', h = 32, w = 32)
    # print("rgb out shape", rgb_out.shape)
    # exit(1)
    
    orig_img = cv2.resize(orig_img, (32,32))
    to_save =  np.concatenate((orig_img, rgb_out, x), axis = 1) #original image| predicted rgb| predicted feat.
    print("saving....")
    cv2.imwrite('./v2.jpg', to_save)

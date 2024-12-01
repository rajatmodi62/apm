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

if __name__ == '__main__':
    
    data_root = '../data/cluster_folder'
    val_split = 'val2017'
    h,w = 32,32
    d = 1024 
    patch_size = 14
    fwd_chunk_size = 32
    batch_size = 1
    num_workers = 8
    num_epochs = 100000
    lr = 0.0001
    device = 'cuda'
    train_print_freq = 20
    save_path = Path('./checkpoints')
    save_path.mkdir(parents=True, exist_ok=True)
    model_save_path = Path('checkpoints/model_15.pth')

    
    save_dir = Path('../visualizations')
    save_dir.mkdir(parents=True, exist_ok=True)
    #build dataset 
    val_dataset = PerceptionFieldDataset(split = val_split,\
                data_root = data_root, patch_size = patch_size)
    
    #build dataloader
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    #build model
    model = Model(hidden_dim = d, h = h, w = w, fwd_chunk_size = fwd_chunk_size).to(device)

    #load model
    model.load_state_dict(torch.load(model_save_path), strict = True)
    print("loaded model")    
    # exit(1)
    #train loop
    for done, batch in enumerate(val_dataloader):
            # print("done val", done)
        x, x_avg, feat = batch #original image, downsampled image, feature latent for each location
        print("done",done,"/", len(val_dataloader))
        print("x avg", x_avg.shape)
        orig_img = x.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        orig_img = (orig_img - np.min(orig_img)) / (np.max(orig_img) - np.min(orig_img))
        orig_img = orig_img * 255
        orig_img = orig_img.astype(np.uint8)
        # cv2.imwrite('orig.png', orig_img)
        x = x.to(device)
        x_avg = x_avg.to(device)
        feat = feat.to(device)
        loss, feat_loss, rgb_loss, feat_out, rgb_out = model.forward_wrapper(x,x_avg, feat) 
        feat_out = feat_out.squeeze(0).cpu().detach().numpy()
        rgb_out = rgb_out.squeeze(0).cpu().detach().numpy()
        
        
        error = feat_out - feat.cpu().detach().numpy()
        error = np.abs(error)
        #collect error along dimensions
        error = np.sum(error, axis = 1)
        error = (error - np.min(error)) / (np.max(error) - np.min(error))
        error = error * 255
        error = error.astype(np.uint8)
        heatmap = cv2.applyColorMap(error, cv2.COLORMAP_HOT)
        # print("heatmap appied",type(heatmap))
        # exit(1)
        
        error = rearrange(error,'b (h w)-> (b h) w',h=32,w=32)  #b was 1
        error = repeat(error, 'h w -> h w c', c = 3)
        print("error",error.shape)
        # exit(1)
        
        
        
        print("rgb out", rgb_out.shape)
        rgb_out = (rgb_out - np.min(rgb_out)) / (np.max(rgb_out) - np.min(rgb_out))
        rgb_out = rgb_out * 255
        rgb_out = rgb_out.astype(np.uint8)
        rgb_out = rgb_out.transpose(1,2,0)
        
        # cv2.imwrite('reconstruction.png', rgb_out)
        # exit(1)
        # print("feat out", feat_out.shape, rgb_out.shape)
        print("feat out", feat_out.shape)
        
        
        ###### predicted islands ##############################
        x = TSNE(n_components=3, learning_rate='auto',
                 init='random', perplexity=3).fit_transform(feat_out)
        


        x = rearrange(x, '(h w) c -> h w c', h = 32, w = 32)

        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = x * 255
        x = x.astype(np.uint8)
        # cv2.imwrite('featout.png', x)
        ###########################################################
        
        #############tsne for gt_islands###########################
        print("feat shape", feat.shape)
        feat = feat.squeeze(0).cpu().detach().numpy()
        
        feat = TSNE(n_components=3, learning_rate='auto',
                 init='random', perplexity=3).fit_transform(feat)
        feat = rearrange(feat, '(h w) c -> h w c', h = 32, w = 32)
        feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
        feat = feat * 255
        feat = feat.astype(np.uint8)
        # exit(1)
        
        
        #print("orig_imf", orig_img.shape, x.shape)
        
        ###########original downsampled image#######################
        orig_img = cv2.resize(orig_img, (32,32))
        ############################################################
        
        ######################calculate error#########################
        
        
        
        to_save =  np.concatenate((orig_img, rgb_out,feat, x,error), axis = 1) #original downsampled image, reconstruction, gt_features, islands
        save_path = save_dir/'{}.jpg'.format(str(done))
        cv2.imwrite(str(save_path), to_save)
        # break
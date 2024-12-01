#author: rmodi
import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
import glob 
from pathlib import Path 
import cv2
from einops import rearrange, reduce, repeat
from dataloader import PerceptionFieldDataset
from model import Model

if __name__ == '__main__':
    
    data_root = '/home/rmodi/krishna/hinton/morphogenesis/data/coco/features'
    train_split = 'train2017'
    val_split = 'val2017'
    
    # data_root = '/home/rmodi/ssd/krishna/morphogenesis/data/cluster_folder'
    # train_split = 'val2017'
    # val_split = 'val2017'
    
    
    h,w = 32,32
    d = 1024 
    patch_size = 14
    fwd_chunk_size = 512
    batch_size = 32*2
    num_workers = 8
    num_epochs = 50000*2
    lr = 0.0001
    device = 'cuda'
    train_print_freq = 20
    save_dir = Path('./checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)


    #build dataset 
    train_dataset = PerceptionFieldDataset(split = train_split,\
                data_root = data_root, patch_size = patch_size)
    val_dataset = PerceptionFieldDataset(split = val_split,\
                data_root = data_root, patch_size = patch_size)
    
    #build dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    #build model
    model = Model(hidden_dim = d, h = h, w = w, fwd_chunk_size = fwd_chunk_size).to(device)
    state_dict = '/home/rmodi/krishna/hinton/morphogenesis/codebase_3/checkpoints/model_3.pth'
    model.load_state_dict(torch.load(state_dict), strict = True)
    print("loaded model")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    #train loop
    val_loss = []
    for epoch in range(4, num_epochs):
        print("epoch", epoch)
        running_total_loss = []
        running_feat_loss = []
        running_rgb_loss = []
        for done, batch in enumerate(train_dataloader):
            # print("done", done)
            x,x_avg, feat = batch
            x = x.to(device)
            x_avg = x_avg.to(device)
            feat = feat.to(device)
            loss, feat_loss, rgb_loss, feat_out, rgb_out = model.forward_wrapper(x, x_avg, feat) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_total_loss.append(loss.item())
            running_feat_loss.append(feat_loss.item())
            #dummy for this experiment
            # running_rgb_loss.append(0)
            running_rgb_loss.append(rgb_loss.item())  
            # print("loss", loss.item())
            print("done",done)
            if done % train_print_freq == 0:
                print("epoch:{}/{} done:{}/{} rgb_loss:{} feat_loss:{} total_loss:{}".format(epoch, num_epochs, done, len(train_dataloader), np.mean(running_rgb_loss), np.mean(running_feat_loss), np.mean(running_total_loss)))
            # break
        running_val_loss = []
        print("validating")
        for done, batch in enumerate(val_dataloader):
            # print("done val", done)
            x,x_avg, feat = batch
            x = x.to(device)
            x_avg = x_avg.to(device)
            feat = feat.to(device)
            loss, feat_loss, rgb_loss, feat_out, rgb_out = model.forward_wrapper(x,x_avg, feat) 
            running_val_loss.append(loss.item())
            if done % train_print_freq == 0:
                print("epoch:{}/{} done:{}/{} val_loss:{}".format(epoch, num_epochs, done, len(val_dataloader), np.mean(running_val_loss)))
            # break
        val_loss.append(np.mean(running_val_loss))
        
        #save model
        save_path = save_dir/'model_{}.pth'.format(epoch)
        torch.save(model.state_dict(), str(save_path))
        print("saved model to", save_path)
    
        #dump the val loss run                     
        with open('val_loss.txt', 'w') as f:
            for item in val_loss:
                f.write("%s\n" % item)
        
    
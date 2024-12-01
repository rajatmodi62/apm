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
import imageio

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
    model_save_path = Path('/home/rmodi/ssd/krishna/morphogenesis/codebase/checkpoints/model_1.pth')

    
    save_dir = Path('../interpolations')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    #build model
    model = Model(hidden_dim = d, h = h, w = w, fwd_chunk_size = fwd_chunk_size).to(device)

    #load model
    model.load_state_dict(torch.load(model_save_path), strict = True)
    print("loaded vector")
    # start_vector = torch.randn(1, 1024).to(device)
    # end_vector = 100*torch.randn(1, 1024).to(device)
    
    #generate n vectors between start and end
    # n = 100
    # vectors = []
    # for i in range(n):
    #     vectors.append(start_vector + (end_vector - start_vector) * i / n)
    # summary_vector = torch.randn(1,1024).cuda()
    
    def generate_circular_trajectory(radius, dimensions, num_samples):
        # Generate random angles for each dimension
        angles = np.random.uniform(0, 2*np.pi, size=(num_samples, dimensions-1))

        # Convert spherical coordinates to Cartesian coordinates
        coordinates = np.zeros((num_samples, dimensions))
        coordinates[:, 0] = radius  # Set the radius for the first dimension

        for i in range(1, dimensions):
            coordinates[:, i] = coordinates[:, i-1] * np.cos(angles[:, i-1])
            if i < dimensions - 1:
                coordinates[:, i] *= np.sin(angles[:, i-1])

        return coordinates
    
    n = 100
    coordinates = generate_circular_trajectory(100, 1024, n)
    print(coordinates.shape)
    vectors = []
    for i in range(n):
        vectors.append(coordinates[i].reshape(1,1024))
    # exit(1)
    #generate n images from these vectors
    images = []
    import random
    for i in range(n):
        print("vector shape", vectors[i].shape)
        # summary_vector = torch.from_numpy(vectors[i]).cuda()
        summary_vector = random.randint(0,100)*torch.randn(1,1024).cuda() 
        print("summary vector shape",summary_vector.shape)
        feat_out, rgb_out = model.interpolate_wrapper(summary_vector)
        # tensor_from_numpy = torch.tensor(numpy_array, dtype=tensor_randn_like.dtype, device=tensor_randn_like.device)
        
        # rgb_out = (rgb_out - np.min(rgb_out)) / (np.max(rgb_out) - np.min(rgb_out))
        # rgb_out = rgb_out * 255
        # rgb_out = rgb_out.astype(np.uint8)
        # rgb_out = rgb_out.transpose(1,2,0)
        # cv2.imwrite('reconstruction.png', rgb_out)
        print("feat out",i, feat_out.shape, rgb_out.shape)
        feat_out  = feat_out.cpu().detach().numpy()
        x = TSNE(n_components=3, learning_rate='auto',
                 init='random', perplexity=3).fit_transform(feat_out)
        


        x = rearrange(x, '(h w) c -> h w c', h = 32, w = 32)

        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = x * 255
        x = x.astype(np.uint8)
        # cv2.imwrite('featout.png', x)
        
        to_save = x
        save_path = save_dir/'{}.jpg'.format(str(i))
        images.append(to_save)
        cv2.imwrite(str(save_path), to_save)
        # break
    output_gif = 'output.gif'

    # Save the list of images as a GIF
    imageio.mimsave(output_gif, images, duration=2) 
    
#author: rmodi
import os 
import torch
import glob 
import numpy as np 
from sklearn.manifold import TSNE
from einops import rearrange, reduce, repeat
import cv2
from pathlib import Path


model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').cuda()

def get_layer_output_hook( name):
        def hook(module, input, output):
            # 'output' here contains the features from the layer
            layer_outputs[name] = output
        return hook

    
split = 'train2017'
# split = 'val2017'
# split = 'test2017'
data_root = Path('./data/coco/resized')/split 

image_paths = sorted(glob.glob(str(data_root/'*.jpg'), recursive=True))

save_path = Path('./data/coco/features')/split
save_path.mkdir(parents=True, exist_ok=True)

for done, img_path in enumerate(sorted(image_paths)):
    print("done", done, "/", len(image_paths))
    img = cv2.imread(img_path)
    x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()
    
    layer_outputs = {}
    for name, module in model.named_modules():
        layer_outputs[name] = None
        module.register_forward_hook(get_layer_output_hook(name))
    
    y = model(x)
    final = layer_outputs['norm']
    x = final[0][:-5].cpu().detach().numpy()
    img_id = img_path.split('/')[-1].split('.')[0] + '.npy'
    np.save(str(save_path/img_id), x)
    

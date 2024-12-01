#author: rmodi
import os 
import torch 
import torchvision
import numpy as np
from sklearn.manifold import TSNE
from einops import rearrange, reduce, repeat
import cv2
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').cuda()

h,w= 224*2, 224*2


layer_outputs = {}
def get_layer_output_hook( name):
        def hook(module, input, output):
            # 'output' here contains the features from the layer
            layer_outputs[name] = output
        return hook
    
for name, module in model.named_modules():
    layer_outputs[name] = None
    module.register_forward_hook(get_layer_output_hook(name))

x = torch.randn(1, 3, 224, 224).cuda()

img_path = 'istockphoto-1368965646-612x612.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (h,w))
x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()
# exit(1)
y = model(x) # (1, 1000)
print(layer_outputs.keys())

final = layer_outputs['norm']
x = final[0][:-5].cpu().detach().numpy()
print(x.shape)
tsne = TSNE(n_components=3, random_state=42)
print("start tsne")
x = tsne.fit_transform(x)
print(x.shape)

min_val = np.min(x)
max_val = np.max(x)

# Normalize the array to the range [0, 1]
x = (x - min_val) / (max_val - min_val)
x = x * 255
print(np.min(x), np.max(x), x.shape)

x = rearrange(x, '(h w) d ->  h w d ', h = h//14)
# x = cv2.resize(x, (h*16, w*16), interpolation=cv2.INTER_NEAREST)

print(x.shape)
cv2.imwrite('tsne.png', np.uint8(x))
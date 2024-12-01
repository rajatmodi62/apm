import torch
import clip
from PIL import Image
from fvcore.nn import FlopCountAnalysis
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)


def count_layers(model):
    return sum(1 for _ in model.children())


image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# print(image.shape)
image = torch.randn(1, 3, 224, 224).to(device)


texts = ["a photo of" for _ in range(1)]
text = clip.tokenize(texts).to(device)

# output = model(image, text)
# print(type(output), len(output), output[0].shape, output[1].shape)
flops = FlopCountAnalysis(model, (image,text))
print(flops.total())
exit(1)
# print(flops.by_module_and_operator())
# # Count the layers
# num_layers = count_layers(model)
# print("Number of layers:", num_layers)


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# num_parameters = count_parameters(model)
# print("Number of parameters:", num_parameters)
9794582002.0

# with torch.no_grad():
#     t1 = time.time()
#     image_features = model.encode_image(image)
#     t2 = time.time()
#     print(t2-t1)    

#     t1 = time.time()
#     text_features = model.encode_text(text)
#     t2 = time.time()
#     print(t2-t1)
    
    
t1 = time.time()
image_features = model.encode_image(image)
t2 = time.time()
print(t2-t1)    

t1 = time.time()
text_features = model.encode_text(text)
t2 = time.time()
print(t2-t1)

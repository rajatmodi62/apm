#author: rmodi
#resize coco images 
import torch 
import glob 
import numpy as np 
from pathlib import Path 
import cv2 

# stolen from chatgpt
def center_crop(image, target_size):
    """
    Center crops the input image to the specified target size.

    Parameters:
        image (numpy.ndarray): Input image read using OpenCV.
        target_size (tuple): Target size (height, width) for the cropped image.

    Returns:
        numpy.ndarray: Center-cropped image.
    """
    # Get the dimensions of the input image
    height, width = image.shape[:2]

    # Calculate the cropping box
    y1 = int((height - target_size[0]) / 2)
    y2 = y1 + target_size[0]
    x1 = int((width - target_size[1]) / 2)
    x2 = x1 + target_size[1]

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


# split = 'train2017'
# split = 'test2017'
split = 'val2017'
resize_h, resize_w = 224*2, 224*2
data_root = Path('./data/coco/original')/split
img_paths = sorted(glob.glob(str(data_root/'*.jpg'), recursive=True))
to_save = Path('./data/coco/resized')/split
to_save.mkdir(parents=True, exist_ok=True)

for done, img_path in enumerate(img_paths):
    if done % 100 == 0:
        print(f"{done}/{len(img_paths)}")
    img = cv2.imread(img_path)
    img = center_crop(img, (resize_h, resize_w))
    cv2.imwrite(str(to_save/img_path.split('/')[-1]), img)
    # break

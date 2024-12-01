import cv2 
import numpy as np 

x = 'turing.jpg'
img = cv2.imread(x)
img = cv2.resize(img, (32,32))

cv2.imwrite(x, img)

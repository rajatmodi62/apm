#author: rmodi 
#create a movie from interpolation images, 
import os
import imageio
import glob
from pathlib import Path
import cv2



if __name__ == "__main__":

    dir = Path('/home/rmodi/ssd/krishna/morphogenesis/interpolations')
    
    image_paths = sorted(glob.glob(str(dir/'*.png'), recursive=True))
    output_path = './output.gif'
    duration = 1000
    frames = []
    for done, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        text = str(done+1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        font_color = (0, 0, 255)  # Black color

        text_position = (25, 20)  # Coordinates (x, y) of the top-left corner of the text

        cv2.putText(img, text, text_position, font, font_scale, font_color, font_thickness)

        frames.append(img)
    
    imageio.mimsave(output_path, frames, duration=duration)
    print("done!!")
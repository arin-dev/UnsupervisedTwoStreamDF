import os
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from skimage import feature

count = 1

def convert_to_ela_image(filename, quality):
    global count
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    if count % 500 == 0:
        ela_im.show()
        count = 1
    else:
        count += 1

    return ela_im

def visualize_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            ela_image = convert_to_ela_image(image_path, 90)
            ela_image.show()  # Visualize the ELA image

# Sample usage
folder_path = r'specific_folder_path'  # Specify your folder path here
visualize_images_in_folder(folder_path)
"""
##########################################
Image to Numpy Converter
Author: Solam Lee, MD (solam@yonsei.ac.kr)
##########################################

< Information >

This program will convert the dataset composed of

1. the clinical photographs (saved in .jpg format with RGB colorspace. won't work with CYMK format.)
2. the pixelwise annotations for the hair loss (target) (saved in .gif format)
3. the pixelwise annotations for the scalp area (mask) (saved in .gif format)

into numpy files that can work with our main program.

Please note that the three items should have same image size from each other.

"""

# Import Library

import os
import PIL.Image as pilimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize

"""

< Settings >

All images and annotation will be resized into (height, width). The default size is 320 x 320.
Please specify each directory for clinical images and pixelwise annotations for hair loss (target) and scalp area (mask)

"""

# Settings

height = 320
width = 320

image_dir = "dataset/image/"
target_dir = "dataset/target/"
mask_dir = "dataset/imask/"

image_npy_filename = "image.npy"
target_npy_filename = "target.npy"
mask_npy_filename = "mask.npy"


# Main Process

n_img = len(os.listdir(image_dir))

image = np.empty ( (n_img, height, width, 3) )
target = np.empty ( (n_img, height, width) ) 
mask = np.empty ( (n_img, height, width) )

for img_no, file_name in enumerate(os.listdir(image_dir)):
    img_name = file_name.lower()[:-4]
    image[img_no] = pilimg.open ( image_dir + img_name + ".jpg" ).resize((height, width))
    target[img_no] = pilimg.open ( target_dir + img_name + ".gif" ).resize((height, width))
    mask[img_no] = pilimg.open ( mask_dir + img_name + ".gif" ).resize((height, width))


# Convert Datatype and Save Numpy

image = image.astype(np.uint8)
target = target.astype(np.uint8)
mask = mask.astype(np.uint8)

np.save ( image_npy_filename, image )
np.save ( target_npy_filename, target )
np.save ( mask_npy_filename,mask )
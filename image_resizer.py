# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:43:04 2018

@author: 583937
"""

#create list of images
import os
image_list = []
stage1_train = os.getcwd() + '\stage1_train'
image_list = [name for name in os.listdir(stage1_train)]

from PIL import Image
from resizeimage import resizeimage #you will need to 'pip install python-resize-image'

#loop through images, resize to 256x256, and place new image in its respective 'image' folder
for selected_image in image_list:
    image_path = ''
    stage1_train = os.getcwd() + '\stage1_train'
    image_path = stage1_train + '/' + selected_image
    image = image_path + '/images/' + selected_image + '.png'

    with open(image, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_contain(image, [256,256])
            cover.save(image_path + '/images/' + selected_image + '-image-resize.png', image.format)

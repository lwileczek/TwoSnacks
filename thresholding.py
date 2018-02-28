#load packages
import imageio
import os
import numpy as np
from IPython.display import display,clear_output,Image
from skimage.color import rgb2gray
#gray the images and use otsu thresholding

image_list = []
stage1_train = os.getcwd() + '\\stage1_train'
image_list = [name for name in os.listdir(stage1_train)]

from skimage.filters import threshold_otsu

#loop through the images
for selected_image in image_list:
    image_path = ''
    stage1_train = os.getcwd() + '\\stage1_train'
    image_path = stage1_train + '/' + selected_image
    image = image_path + '/images/' + selected_image + '.png'

    #gray out the images using rgb2gray
    with open(image, 'r+b') as f:
        image = imageio.imread(f)
        im_gray = rgb2gray(image)
        
        #apply the otsu thresholding to the new grayed image
        thresh_val = threshold_otsu(im_gray)
        boolean_matrix = im_gray < thresh_val
        
        # Make sure the larger portion of the mask is considered background
        if np.sum(boolean_matrix) < np.sum(~boolean_matrix):
            boolean_matrix = ~boolean_matrix
        im_gray[boolean_matrix] = 0
        mask = im_gray 
    
        if not os.path.exists(image_path + '/gray_otsu/'):
            os.makedirs(image_path + '/gray_otsu/')
            imageio.imwrite(image_path + '/gray_otsu/' + selected_image + '-image-gray-otsu.png', mask)
      
image_list = []
stage1_train = os.getcwd() + '\\stage1_train'
image_list = [name for name in os.listdir(stage1_train)]

#loop through the images
for selected_image in image_list:
    image_path = ''
    stage1_train = os.getcwd() + '\\stage1_train'
    image_path = stage1_train + '/' + selected_image
    image = image_path + '/images/' + selected_image + '.png'
    
    total_mask = []
    #gray out the images using rgb2gray
    with open(image, 'r+b') as f:
        image = imageio.imread(f)
        total_mask = np.zeros(image.shape[0:2])
        
    mask_list = os.listdir(image_path + '/masks/') 
    for mask in mask_list:
        image = image_path + '/masks/' + mask
        with open(image, 'r+b') as f:
            image = imageio.imread(f)
            total_mask = (total_mask > 0) | (image > 0)
    
    if not os.path.exists(image_path + '/combined_mask/'):
        os.makedirs(image_path + '/combined_mask/')
        imageio.imwrite(image_path + '/combined_mask/' + selected_image + '-combined_mask.png', total_mask)
        
    otsu = image_path + '/gray_otsu/' + selected_image + '-image-gray-otsu.png'
    with open(otsu, 'r+b') as f:
        otsu = imageio.imread(f)
        print(1 - (np.count_nonzero((otsu > 0) ^ (total_mask > 0))/(total_mask.shape[0] * total_mask.shape[1])))
        
#Resize images
#Set to [256,256]

from PIL import Image
from resizeimage import resizeimage #you will need to 'pip install python-resize-image'

#loop through each image, create a folder called resize under the path for that image, resize to [256, 256] and output to 
#new folder with the name ending in -image-resize
for selected_image in image_list:
    image_path = ''
    stage1_train = os.getcwd() + '\\stage1_train'
    image_path = stage1_train + '/' + selected_image
    image = image_path + '/gray_otsu/' + selected_image + '-image-gray-otsu.png'

    with open(image, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_contain(image, [256,256])
            if not os.path.exists(image_path + '/resize_otsu/'):
                os.makedirs(image_path + '/resize_otsu/')
                cover.save(image_path + '/resize_otsu/' + selected_image + '-image-resize.png', image.format)


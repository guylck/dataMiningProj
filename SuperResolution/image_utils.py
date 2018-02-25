from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.misc import imsave, imread, imresize
# from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage.filters import gaussian_filter
# import cv

from keras import backend as K

import os
import time


_image_scale_multiplier = 1
img_size = 128 * _image_scale_multiplier
stride = 64 * _image_scale_multiplier

def subimage_generator(img, stride, patch_size, nb_hr_images):
    for _ in range(nb_hr_images):
        for x in range(0, img_size, stride):
            for y in range(0, img_size, stride):
                subimage = img[x : x + patch_size, y : y + patch_size, :]

                yield subimage

def transform_images(directory, output_directory, scaling_factor=2, max_nb_images=-1, true_upscale=False):
    index = 1

    if not os.path.exists(output_directory + "X/"):
        os.makedirs(output_directory + "X/")

    if not os.path.exists(output_directory + "y/"):
        os.makedirs(output_directory + "y/")

    # For each image in input_images directory
    nb_images = len([name for name in os.listdir(directory)])

    if max_nb_images != -1:
        print("Transforming %d images." % max_nb_images)
    else:
        assert max_nb_images <= nb_images, "Max number of images must be less than number of images in path"
        print("Transforming %d images." % (nb_images))

    if nb_images == 0:
        print("Extract the training images or images from imageset_91.zip (found in the releases of the project) "
              "into a directory with the name 'input_images'")
        print("Extract the validation images or images from set5_validation.zip (found in the releases of the project) "
              "into a directory with the name 'val_images'")
        exit()

    for file in os.listdir(directory):
        img = imread(directory + file, mode='RGB')

        # Resize to 256 x 256
        img = imresize(img, (img_size, img_size))

        # Create patches
        hr_patch_size = (16 * scaling_factor * _image_scale_multiplier)
        nb_hr_images = (img_size ** 2) // (stride ** 2)

        hr_samples = np.empty((nb_hr_images, hr_patch_size, hr_patch_size, 3))

        image_subsample_iterator = subimage_generator(img, stride, hr_patch_size, nb_hr_images)

        stride_range = np.sqrt(nb_hr_images).astype(int)

        i = 0
        for j in range(stride_range):
            for k in range(stride_range):
                hr_samples[i, :, :, :] = next(image_subsample_iterator)
                i += 1

        lr_patch_size = 16 * _image_scale_multiplier

        t1 = time.time()
        # Create nb_hr_images 'X' and 'Y' sub-images of size hr_patch_size for each patch
        for i in range(nb_hr_images):
            ip = hr_samples[i]
            # Save ground truth image X
            imsave(output_directory + "/y/" + "%d_%d.png" % (index, i + 1), ip)

            # Apply Gaussian Blur to Y
            op = gaussian_filter(ip, sigma=0.5)

            # Subsample by scaling factor to Y
            op = imresize(op, (lr_patch_size, lr_patch_size), interp='bicubic')

            if not true_upscale:
                # Upscale by scaling factor to Y
                op = imresize(op, (hr_patch_size, hr_patch_size), interp='bicubic')

            # Save Y
            imsave(output_directory + "/X/" + "%d_%d.png" % (index, i+1), op)

        print("Finished image %d in time %0.2f seconds. (%s)" % (index, time.time() - t1, file))
        index += 1

        if max_nb_images > 0 and index >= max_nb_images:
            print("Transformed maximum number of images. ")
            break

    print("Images transformed. Saved at directory : %s" % (output_directory))


transform_images("training_data/JPEG96/", "training_data/", 3)
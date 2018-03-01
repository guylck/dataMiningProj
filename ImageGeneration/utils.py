from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input


def create_inception_embedding(grayscaled_rgb, inception):

    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


def resize_images(srcDir, dstDir, baseWidth=32):
    files = [f for f in listdir(srcDir) if isfile(join(srcDir, f))]

    for index in range(len(files)):
        myImage = Image.open(join(srcDir, files[index]))
        wpercent = (baseWidth / float(myImage.size[0]))
        hsize = int((float(myImage.size[1]) * float(wpercent)))
        myImage = myImage.resize((baseWidth, hsize), PIL.Image.ANTIALIAS)
        myImage.save(join(dstDir, files[index]))

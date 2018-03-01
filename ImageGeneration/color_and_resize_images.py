from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import utils
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf

def color_images(images_dir):

    # Load weights of resnet inception model
    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.load_weights('./inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    inception.graph = tf.get_default_graph()

    # Prepare the test data for using it as input for the model
    testDatasetPath = images_dir #"./Resources/test/original/"
    color_me = []
    testFileNames = os.listdir(testDatasetPath)
    for img_name in testFileNames:
        color_me.append(img_to_array(load_img(os.path.join(testDatasetPath, img_name))))
    color_me = np.array(color_me, dtype=float)
    color_me = 1.0/255*color_me
    color_me = gray2rgb(rgb2gray(color_me))
    color_me_embed = utils.create_inception_embedding(color_me, inception)
    color_me = rgb2lab(color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))

    # Loads the model and its weights
    with open("./saved models/model.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("./saved models/color_tensorflow_real_mode.h5")

    # Test model
    output = model.predict([color_me, color_me_embed], batch_size=188)
    output = output * 128

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((32, 32, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave("./Resources/test/result/img_"+str(i)+".png", lab2rgb(cur))

if __name__=="__main__":

    images_dir = input("Enter the path of the test images: ")
    assert os.path.exists(images_dir), "I did not find the images at, " + str(images_dir)

    # Colors the images and saves them in "./Resources/test/result/"
    color_images(images_dir)

    #TODO: take the images from result and resize them with super resolution
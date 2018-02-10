import tensorflow as tf
import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def resize_image(input_image_path,
                 output_image_path,
                 size):
    original_image = Image.open(input_image_path)

    # converts the image to grayscale and then rezise it
    resized_image = original_image.convert('L').resize(size)

    resized_image.save(output_image_path)


def img_to_input_array(img_path):
    # Reads the image
    img = Image.open(img_path)
    # Converts the image to a numpy array
    img_array = np.array(img)
    # flatten the array to 1 dim
    img_array.flatten()

    # Just to show the image
    plt.imshow(img_array.reshape([28, 28]), cmap='Greys')
    plt.show()

    # reshape the array so it will fit the input of the model
    img_array = img_array.reshape([1, img_array.size])
    return (img_array)


# Sets the path to our saved model
best_model_path = "best_model/model.ckpt" + str(250)

# Running a new session
print("Starting session...")
with tf.Session() as sess:
    # 'Saver' op to save and restore all the variables
    saver = tf.train.import_meta_graph("{0}.meta".format(best_model_path))

    # Restore model weights from previously saved model
    saver.restore(sess, best_model_path)

    # Getting the input and the oputput layer
    net_input = sess.graph.get_tensor_by_name("input:0")
    net_output = sess.graph.get_tensor_by_name("output:0")

    print(net_input)
    print(net_output)

    img_dir = "images"
    img_resize_dir = "img_resized"

    # Gonig over all the images in the directory given
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        output_img_path = os.path.join(img_resize_dir, img)

        # Resizes the image to 28X28 and saves the resized image
        resize_image(input_image_path=img_path,
                     output_image_path=output_img_path,
                     size=(28, 28))

        # prints the name of the image
        print(output_img_path)

        # Converts the image to numpy array so it will fit the input of the model
        input_array = img_to_input_array(output_img_path)

        # Gets the result of the model prediction
        result = sess.run(net_output, feed_dict={net_input: input_array})
        print(result)
        print(sess.run(tf.argmax(result, 1)))
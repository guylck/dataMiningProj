from __future__ import print_function
from keras.models import load_model
import os
from matplotlib import pyplot as plt
from keras.preprocessing import image
import numpy as np

# ----------------------------------Predict my own images--------------------------------------------------------
def resize_to_input_and_save(input_image_path,
                             output_image_path,
                             size):
    # Reads the image and resize it
    resized_image = image.load_img(input_image_path, target_size=size)
    # Saves the resized image in the wnated path
    resized_image.save(output_image_path)

    # Gets the img as a numpy array
    img_array = image.img_to_array(resized_image)

    #   Normalize the pixel to be between [0,1] for the prediction
    img_array /= 255

    # Just to show the image
    plt.imshow(img_array)
    plt.show()

    # reshape the array so it will fit the input of the model
    # (number of samples, rows, cols, dimentions(3 cuz rgb))
    img_array = np.expand_dims(img_array, axis=0)

    return (img_array)

# ----------------------------------Load best model--------------------------------------------------------
path = 'best_model/'

# Gets the name of the best model h5 file
best_model_name = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'best_model' in i]
print(best_model_name)

# returns the best model by the path
model = load_model(os.path.join(path, best_model_name[0]))

wanted_classes = ["dog", "cat", "bird", "horse", "frog"]
img_dir = "images"
img_resize_dir = "img_resized"

for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img)
    output_img_path = os.path.join(img_resize_dir, img)

    input_array = resize_to_input_and_save(input_image_path=img_path,
                                           output_image_path=output_img_path,
                                           size=(32, 32))

    result = model.predict(input_array)
    print(output_img_path)
    print(result)
    # Prints the name of the class, by getting the max prob from the prediction
    # which is returned as a list with 1 item which is the index
    # and using that index in the wanted classes list
    print(wanted_classes[np.argmax(result, axis=-1)[0]])
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Conv2D, UpSampling2D, Input, Reshape, concatenate
from keras.models import Model
from keras.layers.core import RepeatVector
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, rgb2gray, gray2rgb
import numpy as np
import os
import tensorflow as tf
import utils

datasetLimit = 5000

# Prepare train images
trainDatasetPath = './Resources/colored32/'
X = []
grayscaleFileList = os.listdir(trainDatasetPath)
for index in range(datasetLimit):
    X.append(img_to_array(load_img(trainDatasetPath + grayscaleFileList[index])))
X = np.array(X, dtype=float)
Xtrain = (1.0/255) * X

#Load weights of resnet inception model
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('./inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = utils.create_inception_embedding(grayscaled_rgb, inception)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, embed], Y_batch)


# The model architecture
embed_input = Input(shape=(1000,))

#Encoder
encoder_input = Input(shape=(32, 32, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
fusion_output = RepeatVector(4 * 4)(embed_input)
fusion_output = Reshape(([4, 4, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3)
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)


#Train model
batch_size = 100
model.compile(optimizer='adam', loss='mse')
model.fit_generator(image_a_b_gen(batch_size), epochs=50, steps_per_epoch=(datasetLimit / batch_size))

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("color_tensorflow_real_mode.h5")
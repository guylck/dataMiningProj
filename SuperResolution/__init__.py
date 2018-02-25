from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
from scipy.misc import imsave, imread, imresize
import keras.callbacks as callbacks
import keras.optimizers as optimizers
from keras.preprocessing import image

import numpy as np
import os
import time
import warnings

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


# The factor to scale the image resolution
# in our case - 96 / 32 = 3
scale_factor = 3

# Will be determined after training
# self.weight_path = "weights/SR Weights %dX.h5" % (self.scale_factor)
weights_path = "weights/SR Weights %dX.h5" % (scale_factor)

training_path = "training_data/"
validate_path = "training_data/"

# members from BaseSuperResolution
f1 = 9
f2 = 1
f3 = 5
n1 = 64
n2 = 32
# parameters
image_scale_multiplier = 1
model_width = model_height = 32
channels = 3

def train():
    # creating shape
    shape = (model_width * image_scale_multiplier, model_height * image_scale_multiplier, channels)

    # creating model
    initial_model = Input(shape=shape)

    x = Convolution2D(n1, (f1, f1), activation='relu', padding='same', name='level1')(initial_model)
    x = Convolution2D(n2, (f2, f2), activation='relu', padding='same', name='level2')(x)

    out = Convolution2D(channels, (f3, f3), padding='same', name='output')(x)

    model = Model(initial_model, out)

    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
    # if load_weights: model.load_weights(self.weight_path)
    return model


def image_lib_to_arrays(path, size):
    img_list = []
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        # Reads the image and resize it
        img = image.load_img(img_path, target_size=(size, size))
        img_list.append(np.array(img))

    img_list = np.asarray(img_list)
    img_list = img_list.astype('float32')
    img_list /= 255

    return img_list

def _index_generator(N, batch_size=32, shuffle=True, seed=None):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)

        if batch_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)

        current_index = (batch_index * batch_size) % N

        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)


def image_generator(directory, scale_factor=2, target_shape=None, channels=3, small_train_images=False, shuffle=True,
                    batch_size=32, nb_inputs=1, seed=None):
    if not target_shape:
        if small_train_images:
            if K.image_dim_ordering() == "th":
                image_shape = (channels, 16 * image_scale_multiplier, 16 * image_scale_multiplier)
                y_image_shape = (channels, 16 * scale_factor * image_scale_multiplier,
                                 16 * scale_factor * image_scale_multiplier)
            else:
                # image_shape = (16 * image_scale_multiplier, 16 * image_scale_multiplier, channels)
                # y_image_shape = (16 * scale_factor * image_scale_multiplier,
                #                  16 * scale_factor * image_scale_multiplier, channels)
                image_shape = (32 * image_scale_multiplier, 32 * image_scale_multiplier, channels)
                y_image_shape = (32 * scale_factor * image_scale_multiplier,
                                 32 * scale_factor * image_scale_multiplier, channels)
        else:
            if K.image_dim_ordering() == "th":
                image_shape = (channels, 32 * scale_factor * image_scale_multiplier, 32 * scale_factor * image_scale_multiplier)
                y_image_shape = image_shape
            else:
                image_shape = (32 * scale_factor * image_scale_multiplier, 32 * scale_factor * image_scale_multiplier,
                               channels)
                y_image_shape = image_shape
    else:
        if small_train_images:
            if K.image_dim_ordering() == "th":
                y_image_shape = (3,) + target_shape

                target_shape = (target_shape[0] * image_scale_multiplier // scale_factor,
                                target_shape[1] * image_scale_multiplier // scale_factor)
                image_shape = (3,) + target_shape
            else:
                y_image_shape = target_shape + (channels,)

                target_shape = (target_shape[0] * image_scale_multiplier // scale_factor,
                                target_shape[1] * image_scale_multiplier // scale_factor)
                image_shape = target_shape + (channels,)
        else:
            if K.image_dim_ordering() == "th":
                image_shape = (channels,) + target_shape
                y_image_shape = image_shape
            else:
                image_shape = target_shape + (channels,)
                y_image_shape = image_shape

    file_names = [f for f in sorted(os.listdir(directory + "X/"))]
    X_filenames = [os.path.join(directory, "X", f) for f in file_names]
    y_filenames = [os.path.join(directory, "y", f) for f in file_names]

    nb_images = len(file_names)
    print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)
        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            x_fn = X_filenames[j]
            img = imread(x_fn, mode='RGB')
            if small_train_images:
                img = imresize(img, (32 * image_scale_multiplier, 32 * image_scale_multiplier))
            img = img.astype('float32') / 255.

            if K.image_dim_ordering() == "th":
                batch_x[i] = img.transpose((2, 0, 1))
            else:
                batch_x[i] = img

            y_fn = y_filenames[j]
            img = imread(y_fn, mode="RGB")
            img = img.astype('float32') / 255.

            if K.image_dim_ordering() == "th":
                batch_y[i] = img.transpose((2, 0, 1))
            else:
                batch_y[i] = img

        if nb_inputs == 1:
            yield (batch_x, batch_y)
        else:
            batch_x = [batch_x for i in range(nb_inputs)]
            yield batch_x, batch_y


def fit(model, weight_path, batch_size=128, nb_epochs=100, save_history=True, history_fn="Model History.txt"):
    """
    Standard method to train any of the models.
    """

    samples_per_epoch = len([name for name in os.listdir(training_path + "X/")])
    val_count = len([name for name in os.listdir(validate_path + "X/")])

    callback_list = [callbacks.ModelCheckpoint(weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                               mode='max', save_weights_only=True, verbose=2)]
    # if save_history:
    #     callback_list.append(HistoryCheckpoint(history_fn))
    #
    #     if K.backend() == 'tensorflow':
    #         log_dir = './%s_logs/' % self.model_name
    #         tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
    #         callback_list.append(tensorboard)

    # print("Training model : %s" % (self.__class__.__name__))
    model.fit_generator(image_generator(training_path, scale_factor=scale_factor, small_train_images=True,
                                        batch_size=batch_size),
                             steps_per_epoch=samples_per_epoch // batch_size + 1,
                             epochs=nb_epochs, callbacks=callback_list,
                             validation_data=image_generator(validate_path, scale_factor=scale_factor, small_train_images=False, batch_size=batch_size),
                             validation_steps=val_count // batch_size + 1)

    # data = image_lib_to_arrays("training_data/JPEG32", 32)
    # label = image_lib_to_arrays("training_data/JPEG96", 96)
    #
    # model.fit(data, label, epochs=nb_epochs, batch_size=batch_size, callbacks=callback_list,
    #           validation_steps=val_count)

    return model


sr = train()
sr = fit(sr, weights_path)




import cv2
from cv2 import cvtColor, COLOR_BGR2GRAY, imread
from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras import backend as K
from scipy.misc import imsave, imresize
import keras.callbacks as callbacks
import keras.optimizers as optimizers
from keras.preprocessing import image
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
import numpy as np
import os

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
weights_path = "weights/SR Weights %dX.h5" % (scale_factor)

training_path = "./Resources/training_data/colored/"
validate_path = "./Resources/validation_data/colored/"

# parameters
model_width = model_height = 32
channels = 3
batch_size = 100

def get_model(load_weights=False, image_scale_multiplier=1):
    # creating shape
    shape = (model_width * image_scale_multiplier * scale_factor, model_height * image_scale_multiplier * scale_factor, channels)

    # creating model
    initial_model = Input(shape=shape)

    x = Convolution2D(64, (9, 9), activation='relu', padding='same', name='level1')(initial_model)
    x = Convolution2D(32, (1, 1), activation='relu', padding='same', name='level2')(x)

    out = Convolution2D(channels, (5, 5), padding='same', name='output')(x)

    model = Model(initial_model, out)

    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])

    if load_weights:
        model.load_weights(weights_path)

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


def _index_generator(N, batch_size, shuffle=True, seed=None):
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


def image_generator(directory, batch_size, shuffle=True, seed=None, image_scale_multiplier=1):
    x_image_shape = y_image_shape = \
        (model_width * scale_factor * image_scale_multiplier,
         model_height * scale_factor * image_scale_multiplier, channels)

    training_file_names = [f for f in sorted(os.listdir(directory + "X/"))]
    X_filenames = [os.path.join(directory, "X", f) for f in training_file_names]
    y_filenames = [os.path.join(directory, "Y", f) for f in training_file_names]

    nb_images = len(training_file_names)
    print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + x_image_shape)
        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            x_fn = X_filenames[j]
            img = imread(x_fn)
            img = img[..., ::-1]
            img = imresize(img, (x_image_shape[0], x_image_shape[1]))
            img = img.astype('float32') / 255.
            batch_x[i] = img

            y_fn = y_filenames[j]
            img = imread(y_fn)
            img = img[..., ::-1]
            img = img.astype('float32') / 255.
            batch_y[i] = img
        yield (batch_x, batch_y)


def fit(model, weight_path, epochs=100, save_history=True, history_fn="Model History.txt"):
    """
    method to train the model.
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

    model.fit_generator(image_generator(training_path, batch_size=batch_size),
                             steps_per_epoch=samples_per_epoch // batch_size + 1,
                             epochs=epochs, callbacks=callback_list,
                             validation_data=image_generator(validate_path, batch_size=batch_size),
                             validation_steps=val_count // batch_size + 1)

    return model


def upscale(model, img_path, results_folder_path="./", intermediate_folder_path="./", save_intermediate=True, suffix="scaled", verbose=True):

    # Destination path
    path = os.path.splitext(img_path)
    original_filename = os.path.splitext(os.path.basename(img_path))[0];
    filename = results_folder_path + original_filename + "_" + suffix + "(%dx)" % (scale_factor) + path[1]

    # Read image
    true_img = imread(img_path)
    true_img = true_img[..., ::-1]
    init_dim_1, init_dim_2 = true_img.shape[0], true_img.shape[1]

    if verbose:
        print("Old Size : ", true_img.shape)
        print("New Size : (%d, %d, 3)" % (init_dim_1 * scale_factor, init_dim_2 * scale_factor))

    # Use full image for super resolution
    img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor

    images = imresize(true_img, (img_dim_1, img_dim_2))
    images = np.expand_dims(images, axis=0)
    print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))

    # Save intermediate bilinear scaled image is needed for comparison.
    if save_intermediate:
        if verbose:
            print("Saving intermediate image.")
        fn = intermediate_folder_path + original_filename + "_intermediate" + path[1]
        intermediate_img = imresize(true_img, (init_dim_1 * scale_factor, init_dim_2 * scale_factor))
        imsave(fn, intermediate_img)

    # Process images
    img_conv = images.astype(np.float32) / 255.

    # Create prediction for image
    result = model.predict(img_conv, batch_size=batch_size, verbose=verbose)

    if verbose:
        print("De-processing images.")

    # Deprocess image
    result = result.astype(np.float32) * 255.

    # Output shape is (original_width * scale, original_height * scale, channels)
    result = result[0, :, :, :] # Access the 3 Dimensional image vector
    result = np.clip(result, 0, 255).astype('uint8')

    """
    # Used to remove noisy edges
    result = cv2.pyrUp(result)
    result = cv2.medianBlur(result, 3)
    result = cv2.pyrDown(result)
    """

    if verbose:
        print("\nCompleted De-processing image.")
        print("Saving image.")

    imsave(filename, result)


def predict_folder(images_path, result_path, intermediate_path):

    # Prepare the test data for using it as input for the model
    testDatasetPath = images_path
    testFileNames = os.listdir(testDatasetPath)
    for img_name in testFileNames:
        upscale(get_model(load_weights=True, image_scale_multiplier=1),
                os.path.join(testDatasetPath, img_name), result_path, intermediate_path)


if __name__ == "__main__":
    sr = get_model()
    sr = fit(sr, weights_path)
    predict_folder("./Resources/test/test/", "./Resources/test/result/", "./Resources/test/intermediate/")



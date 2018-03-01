from keras.models import Model
from keras.layers import Input
from keras.layers.merge import add
import EED
import EES
import cv2
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2ycbcr, ycbcr2rgb
import numpy
import math
import os

scale = 3
batch_size = 100

# Get train images data and labels

datasetLimit = 6400
trainDatasetPath = './Resources/train/'

Y = []
X = []
trainFileList = os.listdir(trainDatasetPath)
for index in range(datasetLimit):
    img = cv2.imread(trainDatasetPath + trainFileList[index])
    resized_img = cv2.resize(img, (int(img.shape[0] / scale), int(img.shape[1] / scale)), cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YCrCb)

    Y.append(img)
    X.append(resized_img)

Y = numpy.array(Y, dtype=float)
X = numpy.array(X, dtype=float)
Ytrain = (1.0/255) * Y
Xtrain = (1.0/255) * X


# Get validation images data and labels

validationsetLimit = 1600
validateDatasetPath = './Resources/validate/'

Y = []
X = []
validateFileList = os.listdir(validateDatasetPath)
for i in range(validationsetLimit):
    img = cv2.imread(validateDatasetPath + validateFileList[i])
    resized_img = cv2.resize(img, (int(img.shape[0] / scale), int(img.shape[1] / scale)), cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YCrCb)

    Y.append(img)
    X.append(resized_img)

Y = numpy.array(Y, dtype=float)
X = numpy.array(X, dtype=float)
Yvalidate = (1.0/255) * Y
Xvalidate = (1.0/255) * X


def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def model_EEDS():
    _input = Input(shape=(None, None, 1), name='input')
    _EES = EES.model_EES()(_input)
    _EED = EED.model_EED()(_input)
    _EEDS = add(inputs=[_EED, _EES])

    model = Model(input=_input, output=_EEDS)
    Adam = adam(lr=0.0003)
    model.compile(optimizer=Adam, loss='mse')
    return model


# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

def image_YCrCB_gen(batch_size):
    for x_batch, y_batch in datagen.flow(Xtrain, Ytrain, batch_size=batch_size):
        X_batch = x_batch[:, :, :, 0]
        X_batch = X_batch.reshape(X_batch.shape + (1,))
        Y_batch = y_batch[:, :, :, 0]
        Y_batch = Y_batch.reshape(Y_batch.shape + (1,))

        yield (X_batch, Y_batch)

def validation_YCrCb_gen(batch_size):
    for x_batch, y_batch in datagen.flow(Xvalidate, Yvalidate, batch_size=batch_size):
        X_batch = x_batch[:, :, :, 0]
        X_batch = X_batch.reshape(X_batch.shape + (1,))
        Y_batch = y_batch[:, :, :, 0]
        Y_batch = Y_batch.reshape(Y_batch.shape + (1,))

        yield (X_batch, Y_batch)

def EEDS_train():
    _EEDS = model_EEDS()
    print(_EEDS.summary())
    # data, label = pd.read_training_data("./myTrain/train.h5")
    # val_data, val_label = pd.read_training_data("./myTrain/val.h5")

    checkpoint = ModelCheckpoint("./myTrain/EEDS_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='min')
    callbacks_list = [checkpoint]

    # _EEDS.fit(data, label, batch_size=100, validation_data=(val_data, val_label), callbacks=callbacks_list, shuffle=True, epochs=50, verbose=1)

    _EEDS.fit_generator(image_YCrCB_gen(batch_size), epochs=10, steps_per_epoch=(datasetLimit / batch_size), callbacks=callbacks_list, verbose=1, validation_data=validation_YCrCb_gen(batch_size), validation_steps=(validationsetLimit / batch_size))

    _EEDS.save_weights("./myTrain/EEDS_final.h5")


def EEDS_predict():
    testSrcPath = "./Resources/test/test/"
    testsInputsPath = "./Resources/test/inputs/"
    testDstPath = "./Resources/test/result/"
    testsLimit = 100

    bicubicPSNRList = []
    EEDSPSNRList = []

    EEDS = model_EEDS()
    EEDS.load_weights("./myTrain/EEDS_final.h5")

    testFileNames = os.listdir(testSrcPath)
    for index in range(testsLimit):
        fileName = testFileNames[index]
        label = cv2.imread(testSrcPath + fileName)
        shape = label.shape

        img = cv2.resize(label, (int(shape[0] / scale), int(shape[1] / scale)), cv2.INTER_CUBIC)
        cv2.imwrite(testsInputsPath + fileName, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        X = numpy.zeros((1, img.shape[0], img.shape[1], 1))
        X[0, :, :, 0] = img[:, :, 0].astype(float) / 255.
        img = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

        pre = EEDS.predict(X, batch_size=1) * 255.
        pre[pre[:] > 255] = 255
        pre = numpy.uint8(pre)
        img[:, :, 0] = pre[0, :, :, 0]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(testDstPath + fileName, img)

        # psnr calculation:
        im1 = cv2.imread(testSrcPath + fileName, cv2.IMREAD_COLOR)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
        im2 = cv2.imread(testsInputsPath + fileName, cv2.IMREAD_COLOR)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
        im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
        # cv2.imwrite("Bicubic.jpg", cv2.cvtColor(im2, cv2.COLOR_YCrCb2BGR))
        im3 = cv2.imread(testDstPath + fileName, cv2.IMREAD_COLOR)
        im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

        bicubicPSNRList.append(cv2.PSNR(im1[:, :, 0], im2[:, :, 0]))
        EEDSPSNRList.append(cv2.PSNR(im1[:, :, 0], im3[:, :, 0]))

    print("Bicubic Average:")
    print(numpy.mean(numpy.array(bicubicPSNRList, dtype=float)))
    print("EEDS Average:")
    print(numpy.mean(numpy.array(EEDSPSNRList, dtype=float)))

    """
    print("Bicubic:")
    print(cv2.PSNR(im1[:, :, 0], im2[:, :, 0]))
    print("EEDS:")
    print(cv2.PSNR(im1[:, :, 0], im3[:, :, 0]))
    """


if __name__ == "__main__":
    EEDS_train()
    EEDS_predict()

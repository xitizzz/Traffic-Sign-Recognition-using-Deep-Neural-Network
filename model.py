from keras.models import Sequential
from keras.applications import vgg16
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, Input
from keras.models import  Model
from keras.utils import layer_utils

from settings import IMG_SIZE, NUM_CLASSES
import keras.backend as K


def baseline_model():

    model = Sequential()

    # Block 1 Convolution layer 1,2
    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 2 Convolution layer 3,4
    model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 3 Convolution layer 5,6
    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 4 Fully-connected layer 7, output layer 8
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.summary()
    return model


def vgg16_model(weights_path='./imagenet/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'):

    model = Sequential()

    # Block 1 Convolution layer 1,2
    model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu', data_format='channels_last'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2 Convolution layer 3,4
    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3 Convolution layer 5,6,7
    model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4 Convolution layer 8,9,10
    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5 Convolution layer 11,12,13
    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='final_pool'))

    # Load the weights
    if weights_path:
        model.load_weights(weights_path)

    # Block 5 Fully-connected layer 14,15 , Output layer 16
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.summary()

    return model


def baseline_model_normalization():

    model = Sequential()
    BatchNormalization(epsilon=1e-06, momentum=0.99, weights=None)

    # Block 1 Convolution layer 1,2  Normalization layer 3
    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 1 Convolution layer 4,5  Normalization layer 6
    model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 1 Convolution layer 7,8  Normalization layer 9
    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 5 Fully-connected layer 10, Normalization layer 11,  Output layer 12
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.summary()
    return model

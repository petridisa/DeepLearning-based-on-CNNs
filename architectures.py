import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Input, Reshape, Activation, MaxPooling2D, \
    UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.keras.regularizers import l2


def createInitialModel():
    inputShape = (48, 48, 1)
    model = Sequential()
    # Mostly when we go deeper to the encoding part we want to increase the number of channels
    # Construction of ConvolutionLayer & MaxPooling set :
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))

    return model


def createAlternativeModel():
    inputShape = (48, 48, 1)
    # create sequential model ()
    model = Sequential()
    stride = 1

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    # Construction of ConvolutionLayer & MaxPooling set :
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    return model


def createFixedArch():
    inputShape = (48, 48, 1)
    # create sequential model ()
    model = Sequential()
    kernelSize = 3
    poolSize = 2
    stride = 1

    inputs = Input(shape=inputShape)
    model = inputs
    model = Conv2D(32, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)
    model = Dropout(0.1)(model)
    model = MaxPooling2D(padding='same', pool_size=poolSize)(model)
    model = Conv2D(64, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)
    model = Dropout(0.1)(model)
    model = MaxPooling2D(padding='same', pool_size=poolSize)(model)
    model = Conv2D(128, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)
    model = Dropout(0.1)(model)
    model = MaxPooling2D(padding='same', pool_size=poolSize)(model)
    model = Conv2D(256, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)
    model = Dropout(0.1)(model)
    model = MaxPooling2D(padding='same', pool_size=poolSize)(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(7, activation='sigmoid')(model)

    cnn = Model(inputs, model, name='CNN_Final_Architecture')

    return cnn


def createDoubleConvolutionArc():
    inputShape = (48, 48, 1)
    # create sequential model ()
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(7, activation='softmax'))

    return model


def createSampleModel():
    inputShape = (48, 48, 1)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model
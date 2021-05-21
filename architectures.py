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


def createInitialModel():
    inputShape = (48, 48, 1)
    model = Sequential()
    # Mostly when we go deeper to the encoding part we want to increase the number of channels
    # Construction of ConvolutionLayer & MaxPooling set :
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(256,(3,3),activation='relu',input_shape = inputShape))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model


def createAlternativeModel():
    inputShape = (48, 48, 1)
    # create sequential model ()
    model = Sequential()

    # Construction of ConvolutionLayer & MaxPooling set :
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=inputShape))
    model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import os
import h5py
import math
import json
import numpy as np

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Concatenate, Layer
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Lambda, LSTM, Bidirectional
from tensorflow.keras import callbacks
import tensorflow.keras.models as models
from tf_keras_kervolution_2d import KernelConv2D, PolynomialKernel, GaussianKernel, LinearKernel, L1Kernel, L2Kernel

x_train = np.load('./train_x_mel40.npy')
x_test = np.load('./test_x_mel40.npy')

Y_TRAIN = np.load('./Y_train.npy')
Y_TEST = np.load('./Y_test.npy')

# Number of mel-bins of the Magnitude Spectrogram
melSize = x_train.shape[1]

# Sub-Spectrogram Size
splitSize = 20

# Mel-bins overlap
overlap = 10

# Time Indices
timeInd = 500

# Channels used
channels = 2

####### Generate the model ###########
inputLayer = Input((melSize,timeInd,channels))
y_test = []
y_train = []
y_test.append(Y_TEST)
y_train.append(Y_TRAIN)

# First conv-layer -- 32 kernels
CONV = KernelConv2D(filters=32, kernel_size=7, padding='same', 
                    kernel_function=GaussianKernel(gamma=0.5))(inputLayer)
CONV = BatchNormalization(axis=1)(CONV)
CONV = Activation('relu')(CONV)
CONV = MaxPooling2D((5,5))(CONV)
CONV = Dropout(0.3)(CONV)

# Second conv-layer -- 64 kernels
CONV = KernelConv2D(filters=64, kernel_size=7, padding='same',
                    kernel_function=GaussianKernel(gamma=0.5))(CONV)
CONV = BatchNormalization(axis=1)(CONV)
CONV = Activation('relu')(CONV)
CONV = MaxPooling2D((4,100))(CONV)
CONV = Dropout(0.30)(CONV)

# Flatten
FLATTEN = Flatten()(CONV)
DENSE = Dense(100, activation='relu')(FLATTEN)
DROPOUT = Dropout(0.30)(DENSE)
    
# Classifier Layer
outputLayer = Dense(10, activation='softmax')(DROPOUT)

classification_model = Model(inputLayer, outputLayer)

# Summary
print(classification_model.summary())

import tensorflow as tf
tf.keras.utils.plot_model(classification_model, show_shapes=True)

type = 'Baseline-Gaussian'
log = callbacks.CSVLogger('./log_' + type + '.csv')
tb = callbacks.TensorBoard(log_dir='./tensorboard-logs' + type + '.csv')
checkpoint = callbacks.ModelCheckpoint(type + '.h5', monitor='val_accuracy',verbose=1, save_best_only=True)

# Compile the model
classification_model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(), 
              metrics=['accuracy']) 

classification_model.fit(x_train, y_train, batch_size=32, epochs=200,
                         callbacks=[log,tb,checkpoint],verbose=1,
                         validation_data=(x_test, y_test), shuffle=True)

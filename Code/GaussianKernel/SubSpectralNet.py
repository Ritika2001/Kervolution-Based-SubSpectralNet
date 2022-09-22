from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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
subSize = int(splitSize/10)
i = 0
inputs = []
outputs = []
toconcat = []
y_test = []
y_train = []
y_test.append(Y_TEST)
y_train.append(Y_TRAIN)
while(overlap*i <= melSize - splitSize):

	# Create Sub-Spectrogram
    if i==0:
        INPUT = Lambda(lambda inputLayer: inputLayer[:,0:20,:,:],
                   output_shape=(splitSize,timeInd,channels))(inputLayer)
    elif i==1:
        INPUT = Lambda(lambda inputLayer: inputLayer[:,10:30,:,:],
                   output_shape=(splitSize,timeInd,channels))(inputLayer)
    elif i==2:
        INPUT = Lambda(lambda inputLayer: inputLayer[:,20:40,:,:],
                   output_shape=(splitSize,timeInd,channels))(inputLayer)

    # First conv-layer -- 32 kernels
    CONV = KernelConv2D(filters=32, 
                        kernel_size=7, 
                        padding='same',
                        kernel_function=GaussianKernel(gamma=0.5))(INPUT)
    CONV = BatchNormalization(axis=1,
                             gamma_regularizer=l2(0.0001),
                             beta_regularizer=l2(0.0001))(CONV)
    CONV = Activation('relu')(CONV)

    CONV = MaxPooling2D((subSize,5))(CONV)
    CONV = Dropout(0.3)(CONV)

    # Second conv-layer -- 64 kernels
    CONV = KernelConv2D(filters=64, 
                        kernel_size=7, 
                        padding='same',
                        kernel_function=GaussianKernel(gamma=0.5)(CONV)
    CONV = BatchNormalization(axis=1,
                             gamma_regularizer=l2(0.0001),
                             beta_regularizer=l2(0.0001))(CONV)
    CONV = Activation('relu')(CONV)

    # Max pool
    CONV = MaxPooling2D((4,100))(CONV)
    CONV = Dropout(0.30)(CONV)

    # Flatten
    FLATTEN = Flatten()(CONV)
    
    OUTLAYER = Dense(32, activation='relu')(FLATTEN)

    DROPOUT = Dropout(0.30)(OUTLAYER)
    
    # Sub-Classifier Layer
    FINALOUTPUT = Dense(10, activation='softmax')(DROPOUT)

    # to be used for model output
    outputs.append(FINALOUTPUT)

    # to be used for global classifier
    toconcat.append(OUTLAYER)

    y_test.append(Y_TEST)
    y_train.append(Y_TRAIN)

    i = i+1

x = Concatenate()(toconcat)

numFCs = int(math.log(i*32, 2))
print(i*32)
print(numFCs)
print(math.pow(2, numFCs))
neurons = math.pow(2, numFCs)

while(neurons >= 64):
    x = Dense(int(neurons), activation='relu')(x)
    x = Dropout(0.30)(x)
    neurons = neurons / 2

# softmax -- GLOBAL CLASSIFIER
out = Dense(10, activation='softmax')(x)
outputs.append(out)

classification_model = Model(inputLayer, outputs)

print(classification_model.summary())

import tensorflow as tf
tf.keras.utils.plot_model(classification_model, show_shapes=True)

type = 'SubSpectralNet-Gaussian-' + str(melSize) + '_' + str(splitSize) + '_' + str(overlap)
log = callbacks.CSVLogger('./log_' + type + '.csv')
tb = callbacks.TensorBoard(log_dir='./tensorboard-logs' + type + '.csv')
checkpoint = callbacks.ModelCheckpoint(type + '.h5', monitor='val_dense_15_accuracy',verbose=1, save_best_only=True)

classification_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001), 
              metrics=['accuracy']) 
classification_model.fit(x_train, y_train, batch_size=16, epochs=2000,
                         callbacks=[log,tb,checkpoint],verbose=1,
                         validation_data=(x_test, y_test), shuffle=True)

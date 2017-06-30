#!/usr/bin/python


import warnings
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import np_utils
from matplotlib import cm
from keras import metrics
import keras
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.models import Sequential
from keras.optimizers import SGD
import scipy
from keras.layers.core import Dense, Activation, Lambda, Reshape,Flatten
from keras.layers import Conv1D,Conv2D,MaxPooling2D, MaxPooling1D, Reshape
#from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.layers.merge import Concatenate
from keras.models import load_model
from keras.utils import plot_model
#from skimage.util.shape import view_as_blocks
#from skimage.util.shape import view_as_windows
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as K
from keras.layers.local import LocallyConnected1D
from keras.datasets import cifar10



WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

batch_size = 128
num_classes = 10
epochs = 200

# input image dimensions
img_rows, img_cols = 32, 32

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def fws():
    #print "Inside"
    #   Params:
    #   batch ,  lr, decay , momentum, epochs
    #
    #Input shape: (batch_size,40,45,3)
    #output shape: (1,15,50)
    # number of unit in conv_feature_map = splitd
    input = Input(shape=(img_rows,img_cols,3))
    zero_padded_section = keras.layers.convolutional.ZeroPadding2D(padding=(20,17), data_format='channels_last')(input)
    model = keras.applications.vgg16.VGG16(include_top = False,
                    weights = 'imagenet',
                    input_shape = (48,84,3),
                    pooling = 'max',
                    classes = 10)

    model_output = model(input)


    #FC
    dense1 = Dense(units = 512, activation = 'relu',    name = "dense_1")(model_output)
    dense2 = Dense(units = 256, activation = 'relu',    name = "dense_2")(dense1)
    dense3 = Dense(units = 10 , activation = 'softmax', name = "dense_3")(dense2)


    model = Model(inputs = input , outputs = dense3)
    #sgd = SGD(lr=0.08,decay=0.025,momentum = 0.99,nesterov = True)
    model.compile(loss="categorical_crossentropy", optimizer='adam' , metrics = [metrics.categorical_accuracy])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


fws()






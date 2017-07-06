"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from scipy.misc import toimage, imresize
import numpy as np
#import resnet
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
from keras.callbacks import ModelCheckpoint


from keras import backend as K
#K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=20)
csv_logger = CSVLogger('./results/vgg16imagenetpretrained_upsampleimage_cifar10_data_argumentation.csv')

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 197, 197
I_R = 64
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train_original, y_train), (X_test_original, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train_original = X_train_original.astype('float32')
X_test_original = X_test_original.astype('float32')


# upsample it to size 64X64X3


X_train = np.zeros((X_train_original.shape[0],I_R,I_R,3))
for i in range(X_train_original.shape[0]):
    X_train[i] = imresize(X_train_original[i], (I_R,I_R,3), interp='bilinear', mode=None)


X_test = np.zeros((X_test_original.shape[0],I_R,I_R,3))
for i in range(X_test_original.shape[0]):
    X_test[i] = imresize(X_test_original[i], (I_R,I_R,3), interp='bilinear', mode=None)


# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.


print(X_train.shape)




#model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
#model =get_vgg_pretrained_model()


 #Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(input_shape=(I_R,I_R,3),weights='imagenet', include_top=False,pooling=max)
model_vgg16_conv.summary()
#print("ss")
    #Create your own input format (here 3x200x200)
input = Input(shape=(I_R,I_R,3),name = 'image_input')
#print("ss2")
    #Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)
print("ss3")
    #Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(10, activation='softmax', name='predictions')(x)

    #Create your own model 
my_model = Model(input=input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()



my_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# serialize model to JSON
model_json = my_model.to_json()
with open("./results/model_data_argumentation.json", "w") as json_file:
    json_file.write(model_json)


print(my_model.summary())
if not data_augmentation:
    print('Not using data augmentation.')
    # checkpoint
   # filepath="./results/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
   # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
   # callbacks_list = [checkpoint]
    # Fit the model
    #model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)

    my_model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    my_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[lr_reducer, early_stopper, csv_logger])
my_model.save_weights("./results/vgg16_pretrained_upsample_model_data_argumentation.h5")
print("Saved model to disk")

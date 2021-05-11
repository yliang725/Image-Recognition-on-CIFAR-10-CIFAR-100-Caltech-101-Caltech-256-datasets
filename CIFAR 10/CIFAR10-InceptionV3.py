'''
    File name: CIFAR10-InceptionV3.py
    Author: Yue Liang
    Date last modified: 4/25/2020
    Python Version: 3.8
    TensorFlow 2.4
'''

# %%

from keras.datasets import cifar10
import numpy as np
import scipy.misc
# %%
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, Input, Dropout, Activation, Dense, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential, Model

# %%
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

# %%
import keras

print('x reshape start')
train_data = train_data.astype('float32')
train_data /= 255
x = np.asarray([scipy.misc.imresize(x_img, [299, 299]) for x_img in train_data])


print('x reshape done')

print('y reshape start')
test_data = test_data.astype('float32')
test_data /= 255
y = np.asarray([scipy.misc.imresize(x_img, [299, 299]) for x_img in test_data])


print('y reshape done')

# %%
train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

# %%
from keras.preprocessing.image import ImageDataGenerator

class_num = 10

base_model = InceptionV3(weights='imagenet', include_top=False)

transfer_learning_arch = base_model.output
transfer_learning_arch = GlobalAveragePooling2D()(transfer_learning_arch)
transfer_learning_arch = Dense(1024, activation='relu')(transfer_learning_arch)
transfer_learning_arch = Dropout(0.4)(transfer_learning_arch)
transfer_learning_arch = Dense(512, activation='relu')(transfer_learning_arch)
transfer_learning_arch = Dropout(0.4)(transfer_learning_arch)
predictions = Dense(class_num, activation='softmax')(transfer_learning_arch)

transfer_learning_model = Model(inputs=base_model.input, outputs=predictions)
transfer_learning_model.summary()

opt = Adadelta(lr=0.3, rho=0.95, epsilon=1e-08, decay=1e-2 / 100)
transfer_learning_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

datagen = ImageDataGenerator(
        zca_epsilon=1e-06,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False,
        validation_split=0.0
    )

print('start11')
callbacks = [ModelCheckpoint('Cifar10_InceptionV3.h5', monitor='val_acc', save_best_only=True)]

datagen.fit(x)

transfer_learning_model.fit_generator(datagen.flow(x,train_labels,batch_size=1),
                                      epochs=500,
                                      validation_data=(y,test_labels),
                                    workers=4)

print('finished')

# %%

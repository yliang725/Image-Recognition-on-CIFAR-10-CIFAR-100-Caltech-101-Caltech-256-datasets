#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
    File name: Caltech256-InceptionV3.py
    Author: Yue Liang
    Date last modified: 4/25/2020
    Python Version: 3.8
    TensorFlow 2.4
'''


# In[ ]:


from keras.models import Sequential, Model
from keras.layers import Conv2D, Input, Dropout, Activation, Dense, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

from keras.applications.inception_v3 import InceptionV3


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   )

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    '256_ObjectCategories/256_ObjectCategories_train',
    target_size=(299, 299),
    batch_size=20,
    class_mode='categorical',
    shuffle=True
    )

validation_generator = validation_datagen.flow_from_directory(
    '256_ObjectCategories/256_ObjectCategories_test',  
    target_size=(299, 299),
    batch_size=20,
    class_mode='categorical',
    )


# In[ ]:


print(train_datagen.dtype)


# In[ ]:


class_num = 257

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

print('start11')
callbacks = [ModelCheckpoint('Caltech256.h5', monitor='val_accuracy', save_best_only=True)]

history = transfer_learning_model.fit_generator(train_generator,
                                                epochs=60,
                                                validation_data=validation_generator,
                                                callbacks=callbacks)


# In[ ]:


from matplotlib import pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:











# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 03:58:07 2020

@author: kingslayer
"""

#CONVOLUTIONAL NEURAL NETWORK

#PART 1 (Making the CNN)

#importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

#Initialising the CNN
classifier=Sequential()

#Step 1-Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(256,256,3),activation="relu"))

#Step 2-Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3-Flatten
classifier.add(Flatten())

#Step 4-Full Connection
classifier.add(Dense(output_dim=128,activation="relu"))
classifier.add(Dense(output_dim=1,activation="sigmoid"))

#Compiling the CNN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


#PART 2(Fitting CNN
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(256,256),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(256,256),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,        
        validation_data=validation_generator,
        validation_steps=800)
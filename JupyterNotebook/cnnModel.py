# -*- coding: utf-8 -*-
"""
@author: karnika
"""
#importing keras libraries and packages
#initialize (layers/graph)
from keras.layers import Sequential 
#image type+basic step
from keras.layers import Convolution2D
#adding pooling layers, feature maps
from keras.layers import MaxPooling2D
#input for fully connected layers 
from keras.layers import Flatten
#add fully connected layers
from keras.layers import Dense 
#choosing no of feature detectors we create (hence feature maps)
#initialising
classifier = Sequential()

#step1 : convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) 
#(no of feature maps = filter use) (stride 1)
#increase filters, input shape when on gpu, 3x3 dimensions
#converting all images in the same format: input_shape argument; expected format
#b&w : 2d array, colored: 3d array(blue, green, red)

#step 2: max pooling : reduced featured map
#(stride 2)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#step 3: flattening (pooled feature map into one single vector)
#mid steps imp to provide info about how pixels are connected to each other along w individual info
classifier.add(Flatten())

#step 4: full connection(ANN)
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #because binary outcome; else softmax

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_cross_entropy', metrics = ['accuracy']) 
#more than 2 outcomes: categorical entropy

#image augmentation: avoid overfitting
#fitting CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) #so same image identified in different batches 

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#fit + test performance 
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        nb_epochs=25,
        validation_data=test_set,
        nb_val_samples=2000)

#increasing accuracy (add another convolutional/fully connected layer: experimentation basis/ high targer_size(higher pixels))
#(after step 2), change input_shape 
















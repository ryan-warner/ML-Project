import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation='relu'))

classifier.add(MaxPool2D(pool_size = (2,2)))

classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation ='relu'))

classifier.add(Dense(1, activation= 'sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/train', target_size=(64,64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test', target_size=(64,64), batch_size = 32, class_mode = 'binary')

classifier.fit(train_set, 
              epochs=5,
              steps_per_epoch=3125,
              validation_data=(test_set), 
              validation_steps=625
              )

result = classifier.predict(test_set)

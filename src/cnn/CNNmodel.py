import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

classifier = Sequential()

# First Convolution and Pooling layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Second Convolution and Pooling layer with larger filters
classifier.add(Conv2D(64, (3, 3)))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Third Convolution and Pooling layer
classifier.add(Conv2D(128, (3, 3)))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Dense layer with more neurons and Dropout
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))  # Adding dropout to reduce overfitting

# Output layer
classifier.add(Dense(1, activation='sigmoid'))

# Compiling the model with an advanced optimizer
optimizer = Adam(learning_rate=0.001)
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/train', target_size=(64,64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test', target_size=(64,64), batch_size = 32, class_mode = 'binary', shuffle=False)

#steps = datapoints/batch_size
r = classifier.fit(train_set, 
              epochs=100,
              steps_per_epoch=len(train_set)//32,
              validation_data=(test_set), 
              validation_steps=len(test_set)//32,
              verbose=2,
              )

# Predicting the Test set results
test_steps_per_epoch = math.ceil(test_set.samples / test_set.batch_size)
predictions = classifier.predict(test_set, steps=test_steps_per_epoch, verbose=2)
predictions = np.round(predictions).tolist()
#predicted_classes = np.where(predictions > 0.5, 1, 0)

print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, predictions)
print(cm)

# roc auc score
roc_auc = roc_auc_score(test_set.classes, predictions)
print('ROC AUC: ', roc_auc)

# Calculate roc auc
print(classification_report(test_set.classes, predictions))

plt.figure()
class_labels = ["FAKE", "REAL"]
ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(values_format='d')
plt.title('Confusion Matrix')
plt.show()



#visualization
plt.figure(figsize=(12, 16))

plt.subplot(4, 2, 1)
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='val_Loss')
plt.title('Loss Function Evolution')
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy Function Evolution')
plt.legend()


plt.subplot(4, 2, 3)
plt.plot(r.history['precision'], label='precision')
plt.plot(r.history['val_precision'], label='val_precision')
plt.title('Precision Function Evolution')
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(r.history['recall'], label='recall')
plt.plot(r.history['val_recall'], label='val_recall')
plt.title('Recall Function Evolution')
plt.legend()

# Adjust the spacing between plots
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.9)

plt.show()
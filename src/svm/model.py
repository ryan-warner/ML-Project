import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from PIL import Image, ImageOps
import requests

from src import logger

import pandas as pd

from dataset.fake_dataset import FakeDataset
from dataset.real_dataset import RealDataset
import sys

def model_iterative():
    # load real ds from TSV file
    ds_real = pd.read_csv('dataset/Train_GCC-training.tsv', sep='\t', header=0, names=['caption', 'url'])

    # Preprocess the data

    # Train the model

    # Evaluate the model



def preprocessing(images):
    # Batch image preprocessing on the fly
    # Get images from url paths, resize, and extract features

    # Resize images, standardize, and grayscale
    features = np.array([])
    for image in images:
        image = image.resize((128, 128))
        image = ImageOps.grayscale(image)
        image = np.array(image)
        image = image.flatten()
        if features.size == 0:
            features = image
        else:
            features = np.vstack((features, image))

    return features

def batch_iterator(real_ds_iter, fake_ds_iter, batch_size=32, batch_limit=1000):
    batches = 0
    while batches < batch_limit or batch_limit == None:
        logger.info(f"Batch {batches + 1}/{batch_limit}")
        # Pull batch_size images in order from the real and fake datasets, randomly order and return with labels
        real_batch = []
        fake_batch = []

        try:
            for i in range(batch_size):
                real_batch.append(next(real_ds_iter))
                fake_batch.append(next(fake_ds_iter))
        except StopIteration:
            break

        # Combine and label
        batch = real_batch + fake_batch
        labels = [1] * batch_size + [0] * batch_size

        # Shuffle
        zipped = list(zip(batch, labels))   
        np.random.shuffle(zipped)

        batch, labels = zip(*zipped)

        yield batch, labels
        batches += 1

def partial_train(preprocess_function, real_ds, fake_ds, batch_size):
    svm_model = SGDClassifier(loss='hinge')

    # Example loop for iterating over the dataset in batches
    for batch, batch_labels in batch_iterator(real_ds, fake_ds, batch_size, batch_limit=100):

        features = preprocess_function(batch)

        # Standardize features
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)

        # Partially fit the SVM on the current batch
        svm_model.partial_fit(features, batch_labels, classes=np.unique(batch_labels))

    return svm_model

def validate(model, real_ds, fake_ds, preprocess_function):
    for batch, batch_labels in batch_iterator(real_ds, fake_ds, batch_size=1000, batch_limit=1):
        features = preprocess_function(batch)

        # Standardize features
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)

        # Make predictions
        predictions = model.predict(features)

        # Evaluate the model
        print(classification_report(batch_labels, predictions))

        # Confusion matrix
        cm = confusion_matrix(batch_labels, predictions)
        print(cm)

        # Accuracy
        accuracy = accuracy_score(batch_labels, predictions)
        print(accuracy)

def model():
    ds_real = pd.read_csv('dataset/Train_GCC-training.tsv', sep='\t', header=0, names=['caption', 'url'])
    ds_real_test = pd.read_csv('dataset/Validation_GCC-1.1.0-Validation.tsv', sep='\t', header=0, names=['caption', 'url'])
    ds_fake = FakeDataset("InfImagine/FakeImageDataset")
    ds_real = RealDataset(ds_real)
    ds_real_test = RealDataset(ds_real_test)
    ds_fake_iter = iter(ds_fake)
    ds_real_iter = iter(ds_real)

    model = partial_train(preprocessing, ds_real_iter, ds_fake_iter, 1000)

    ds_real_test.train = False
    ds_fake.train = False

    ds_fake_iter = iter(ds_fake)
    ds_real_iter = iter(ds_real_test)

    logger.info("Model Validation")
    validate(model, ds_real_iter, ds_fake_iter, preprocessing)


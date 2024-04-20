import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from PIL import Image, ImageOps
import requests
import torch
import torchgan
from torchgan.models import Generator,Discriminator
from torchgan.losses import (GeneratorLoss,DiscriminatorLoss,least_squares_discriminator_loss,least_squares_generator_loss)
from src import logger
from torch.optim import Adam
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import pandas as pd

from dataset.fake_dataset import FakeDataset
from dataset.real_dataset import RealDataset
import sys

# General Imports
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Torchgan Imports
import torchgan
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import Trainer

def model_iterative():
    # load real ds from TSV file
    ds_real = pd.read_csv('dataset/Train_GCC-training.tsv', sep='\t', header=0, names=['caption', 'url'])

    # Preprocess the data

    # Train the model

    # Evaluate the model



def preprocessing(images, augment=True):
    """
    Preprocesses images while preserving spatial dimensions.
    """
    processed_images = []
    for image in images:
        image = image.resize((128, 128))
        # image = ImageOps.grayscale(image)  # Uncomment for grayscale

        # Convert to NumPy array with channels first format
        image = np.array(image).transpose((2, 0, 1)) 

        # Augmentation (optional)
        if augment:
            # You can add various augmentation techniques here, such as:
            # Random cropping, horizontal flipping, color jittering, etc.
            # Example: Horizontal flip
            flipped = np.flip(image, axis=2)
            # Rotation
            rotated_left = np.rot90(image, k=1, axes=(1, 2))  # Rotate 90 degrees left
            rotated_right = np.rot90(image, k=-1, axes=(1, 2))  # Rotate 90 degrees right

            processed_images.extend([image, flipped, rotated_left, rotated_right])
        else:
            processed_images.append(image)
    return processed_images

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

class BatchDataset(dsets):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_dataset_from_iterator(real_ds_iter, fake_ds_iter, batch_size=32, batch_limit=1000):
    """
    Creates a PyTorch Dataset from our batch iterator, so we can use existing GAN architecture
    """
    images = []
    labels = []
    image_count = 0
    num_images = batch_size*batch_limit
    
    for batch, batch_labels in batch_iterator(real_ds_iter, fake_ds_iter):
        images.extend(batch)
        labels.extend(batch_labels)
        image_count += len(batch)
        if image_count >= num_images:
            break
    
    images = preprocessing(images,augment=False) #don't augment for now, if time to train later then sure
    return BatchDataset(images, labels)

      
def classify_image(discriminator, image, device):
    """
    Classifies an image as real or fake using the trained discriminator.
    """
    # Preprocess the image (assuming preprocessing function is defined)
    image = preprocessing([image],augment = False)[0]  # Preprocessing expects a list

    # Convert to PyTorch tensor and add batch dimension
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)  

    # Pass the image through the discriminator
    with torch.no_grad():  # No need for gradients during inference
        output = discriminator(image_tensor)

    # Interpret the output
    # The specific interpretation depends on the final layer and activation
    # of your discriminator. Assuming it outputs a single value:
    if output.item() > 0.5:
        return 1
    else:
        return 0

    
minimax_losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
wgangp_losses = [
    WassersteinGeneratorLoss(),
    WassersteinDiscriminatorLoss(),
    WassersteinGradientPenalty(),
]
lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]

def model():
    ds_real = pd.read_csv('dataset/Train_GCC-training.tsv', sep='\t', header=0, names=['caption', 'url'])
    ds_real_test = pd.read_csv('dataset/Validation_GCC-1.1.0-Validation.tsv', sep='\t', header=0, names=['caption', 'url'])
    ds_fake = FakeDataset("InfImagine/FakeImageDataset")
    ds_real = RealDataset(ds_real)
    ds_real_test = RealDataset(ds_real_test)
    ds_fake_iter = iter(ds_fake)
    ds_real_iter = iter(ds_real)
    dataset = create_dataset_from_iterator(ds_real_iter, ds_fake_iter, num_images=1000)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    cgan_network = {
        "generator": {
            "name": ConditionalGANGenerator,
            "args": {
                "encoding_dims": 100,
                "num_classes": 10,  # MNIST digits range from 0 to 9
                "out_channels": 1,
                "step_channels": 32,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            },
            "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
        },
        "discriminator": {
            "name": ConditionalGANDiscriminator,
            "args": {
                "num_classes": 10,
                "in_channels": 1,
                "step_channels": 32,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            },
            "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
        },
    }

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
        epochs = 10
    else:
        device = torch.device("cpu")
        epochs = 5

    print("Device: {}".format(device))
    print("Epochs: {}".format(epochs))
    trainer_cgan = Trainer(
        cgan_network, lsgan_losses, sample_size=64, epochs=epochs, device=device
    )
    trainer_cgan(dataloader)

    discriminator = trainer_cgan.networks['discriminator']

    testReals = []
    i = 0
    for img in ds_real_test:

        testReals.append(classify_image, img, device)
        if i%10 == 0:
            print(i,"th test")
            print("Current accuracy = ",sum(testReals)/len(testReals))

    print("Final accuracy = ",sum(testReals)/len(testReals))
    

'''
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(plt.imread("{}/epoch{}_generator.png".format(trainer_cgan.recon, trainer_cgan.epochs)))
    plt.show()'''


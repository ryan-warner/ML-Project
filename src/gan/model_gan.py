import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import os

from src import logger
from PIL import Image, ImageOps
import requests
import torch
import torchgan
from torchgan.models import Generator,Discriminator
from torchgan.losses import (GeneratorLoss,DiscriminatorLoss,least_squares_discriminator_loss,least_squares_generator_loss)

from torch.optim import Adam
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import pandas as pd

from dataset.fake_dataset import FakeDataset
from dataset.real_dataset import RealDataset


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


def preprocessing(images, augment=True):
    """
    Preprocesses images while preserving spatial dimensions.
    """
    print("preprocessing")
    processed_images = []
    for image in images:
        #print(image.shape(image))
        image = image.resize((128, 128))
        image = ImageOps.grayscale(image)  # Uncomment for grayscale
        print(np.shape(image))
        # Convert to NumPy array with channels first format
        image = np.array(image)

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

def batch_iterator(real_ds_iter, fake_ds_iter, batch_size=64, batch_limit=10000):
    print("batch iteratoring")
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

def create_dataset_from_iterator(real_ds_iter, fake_ds_iter, batch_size=64, batch_limit=10000):
    """
    Creates a PyTorch Dataset from our batch iterator, 
    and returns the data as NumPy arrays.
    """
    print("attempting to make dataset from iterator")
    images = []
    labels = []
    image_count = 0
    num_images = batch_size * batch_limit

    for batch, batch_labels in batch_iterator(real_ds_iter, fake_ds_iter,batch_limit=batch_limit):
        images.extend(batch)
        labels.extend(batch_labels)
        image_count += len(batch)
        if image_count >= num_images:
            break
        else:
            print("creating dataset from batching progress:",image_count*100/num_images,"%")

    images = preprocessing(images, augment=False)  # Preprocess without augmentation

    # Convert to NumPy arrays
    images_np = np.array(images)
    labels_np = np.array(labels)
    return images_np, labels_np

def SaveDataToFile(images_np, labels_np):
    print("attempting to save to file")
    if not(os.path.exists("images.npy") & os.path.exists("labels.npy")):
        # Save the NumPy arrays
        np.save("images.npy", images_np)
        np.save("labels.npy", labels_np)
    else:
        print("Files already exist. To Overwrite, please delete or move existing files")


def LoadDataFromFile():
    print("attempting to load from file")
    try:
        images_np = np.load("images.npy")
        labels_np = np.load("labels.npy")
    except Exception:
        print("Data Could Not be loaded")

    return images_np, labels_np

def Data2Dataset(images_np, labels_np):
    print("Moving data into Torch TensorDataset")
    # Create a TensorFlow dataset with images and labels
    images_torch = torch.from_numpy(images_np)
    labels_torch = torch.from_numpy(labels_np)
    return torch.utils.data.TensorDataset(images_torch,labels_torch)

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
def InitializeData(batch_limit=10000):
    print("Initializing data file is beginning")
    ds_real = pd.read_csv('dataset/Train_GCC-training.tsv', sep='\t', header=0, names=['caption', 'url'])
    ds_real_test = pd.read_csv('dataset/Validation_GCC-1.1.0-Validation.tsv', sep='\t', header=0, names=['caption', 'url'])
    ds_fake = FakeDataset("InfImagine/FakeImageDataset")
    ds_real = RealDataset(ds_real)
    ds_real_test = RealDataset(ds_real_test)
    ds_fake_iter = iter(ds_fake)
    ds_real_iter = iter(ds_real)
    images_np, labels_np = create_dataset_from_iterator(ds_real_iter, ds_fake_iter,batch_limit=batch_limit)
    print(np.shape(images_np))
    SaveDataToFile(images_np, labels_np)
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
<<<<<<< HEAD
        image = Image.fromarray(self.images[idx])  # Convert NumPy to PIL Image
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

=======
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
>>>>>>> Weiting/CNN
        return image, label
    def __call__(self, idx=None, transform=None):
        if idx:
            image = self.images[idx]
            label = self.labels[idx]
        if transform:
            self.images = transform(self.images)
        return image, label
def model():
    print("model start")
    images_np, labels_np = LoadDataFromFile()
    #dataset = Data2Dataset(images_np,labels_np)
    dataset = CustomImageDataset(images=images_np,labels=labels_np)
    transform=transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    transformed_dataset = dataset(transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)
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
    print("torch check")
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
    print("training complete check")
    discriminator = trainer_cgan.networks['discriminator']

    testReals = []
    i = 0
    ds_real_test = pd.read_csv('dataset/Validation_GCC-1.1.0-Validation.tsv', sep='\t', header=0, names=['caption', 'url'])
    ds_real_test = RealDataset(ds_real_test)
    for img in ds_real_test:

        testReals.append(classify_image, img, device)
        if i%10 == 0:
            print(i,"th test")
            print("Current accuracy = ",sum(testReals)/len(testReals))

    print("Final accuracy = ",sum(testReals)/len(testReals))


#InitializeData(batch_limit=1)



#print(np.shape(images_np[0]))
#data = Image.fromarray(images_np[0]) 
#data.save('gfg_dummy_pic.png') 

model()
# General Imports
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import os
#from src import logger
print(os.path.isdir("./cifakedata/"))
# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
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
#help(torchvision.utils.make_grid)
# Define data path


cuda_epochs = 20
minimax_losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
wgangp_losses = [
    WassersteinGeneratorLoss(),
    WassersteinDiscriminatorLoss(),
    WassersteinGradientPenalty(),
]
lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]

cgan_network = {
        "generator": {
            "name": ConditionalGANGenerator,
            "args": {
                "encoding_dims": 100,
                "num_classes": 2,  # Data is either real or fake
                "out_channels": 3, #RGB
                "step_channels": 16,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            },
            "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
        },
        "discriminator": {
            "name": ConditionalGANDiscriminator,
            "args": {
                "num_classes": 2, 
                "in_channels": 3, #RGB
                "step_channels": 16,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            },
            "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
        },
    }
def model():
    #print("model start")
    # Create CIFAKE dataset
    data_path = "./cifakedata/train"
    cifake_dataset = datasets.ImageFolder(root=data_path, transform=transforms.Compose(
        [
            #transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    ))
    # Create dataloader
    dataloader = DataLoader(cifake_dataset, batch_size=64, shuffle=True)
    print("data loaded")
    #print("torch check")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
        epochs = cuda_epochs
    else:
        device = torch.device("cpu")
        epochs = 1

    print("Device: {}".format(device))
    print("Epochs: {}".format(epochs))
    trainer_cgan = Trainer(
        cgan_network, lsgan_losses, sample_size=64, epochs=epochs, device=device
    )
    trainer_cgan(dataloader)
    #print("training complete check")

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    torchvision.utils.make_grid
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
    plt.show()

    print("testing trained gan begins:")

    trainedGAN =trainer_cgan.model_names[1]
    trainedGAN = getattr(trainer_cgan,trainedGAN)
    
    # Example usage:
    total_accuracy = 0
    num_batches = 0

    data_path_test = "./cifakedata/test"

    cifake_dataset_test = datasets.ImageFolder(root=data_path_test, transform=transforms.Compose(
        [
            #transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    ))
    # Create dataloader
    print("train complete")
    dataloaderNew = DataLoader(cifake_dataset_test, batch_size=100, shuffle=True)
    print("test data loading complete complete")
    # Iterate through the dataloader
    for real_batch, labels in dataloaderNew:
        print("batch testing next start")
        # Test the batch and accumulate accuracy
        accuracy, confusion = test_batch(trainedGAN, real_batch, labels)
        total_accuracy += accuracy
        num_batches += 1
        print("current accuracy is ,",accuracy)
        print("confusion matrix")
        print(confusion)

    # Calculate average accuracy across batches
    average_accuracy = total_accuracy / num_batches

    print(f"Average discriminator accuracy: {average_accuracy}")
    


def test_batch(discriminator, images, labels):
    """
    Tests a batch of images using the discriminator and calculates accuracy.

    Args:
        image_batch (torch.Tensor): A batch of image tensors with shape (B, C, H, W).
        labels (torch.Tensor): A batch of labels (real/fake) with shape (B,).

    Returns:
        float: The accuracy of the discriminator on the batch.
    """
    # Preprocess the image (if necessary)
    # This should match the preprocessing used during training
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
    # Get discriminator's predictions
    with torch.no_grad():
        predictions = discriminator.forward(images.to('cuda'), labels.to('cuda'))

    # Threshold predictions to get class labels (0 or 1)
    predicted_labels = (predictions > 0.5).float()

    # Calculate accuracy
    accuracy = accuracy_score(labels.cpu(), predicted_labels.cpu())
    confusion = confusion_matrix(labels.cpu(),predicted_labels.cpu())
    return accuracy, confusion

model()
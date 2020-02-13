# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:12:38 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

# define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# download and load the training data
train_set = datasets.MNIST(
    "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
data_iter = iter(trainloader)

images, labels = data_iter.next()
#plt.imshow(images[1].numpy().squeeze(), cmap="Greys_r")


# -----------------------------------------------------------------------------
# Class Network
# -----------------------------------------------------------------------------

class Network(nn.Module):
    """
    Class Network.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        # hidden layers
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # output layer
        self.fc3 = nn.Linear(64, 10)
        
        
    def forward(self, x):
        """
        Performs forward pass.
        
        :param x:           network input
        :return:            network output
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    
model = Network()
print(model)
    

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def view_classify(img, act):
    """
    Shows the activations for an instance.
    
    :param img:             mnist image
    :param act:             class activations
    """
    act = act.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    # left image
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    # right image
    ax2.barh(np.arange(10), act)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    
    
# resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)

# forward pass through the network
img = images[0,:]
act = model.forward(img)

view_classify(img.view(1, 28, 28), act)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        # zero gradients
        optimizer.zero_grad()
        # get prediction
        output = model(images)
        
        # parameter optimization
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

act = model.forward(img)
view_classify(img.view(1, 28, 28), act)

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:15:39 2020

@author: Daniel Wehner
@see: https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import torch
import numpy as np
import torchvision.utils as vutils

from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from IPython import display
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# Class DiscriminatorNet
# -----------------------------------------------------------------------------

class DiscriminatorNet(torch.nn.Module):
    """
    Class DiscriminatorNet.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        super(DiscriminatorNet, self).__init__()
        self.n_features = 784
        self.n_out = 1
        
        self.__model_fn()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        
        
    def __model_fn(self):
        """
        Specifies the network.
        """
        self.hidden0 = nn.Sequential( 
            nn.Linear(self.n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, self.n_out),
            torch.nn.Sigmoid()
        )


    def forward(self, X):
        """
        Performs a forward-pass on the data.
        
        :param X:           network input
        """
        X = self.hidden0(X)
        X = self.hidden1(X)
        X = self.hidden2(X)
        X = self.out(X)
        
        return X
    
    
# -----------------------------------------------------------------------------
# Class GeneratorNet
# -----------------------------------------------------------------------------
    
class GeneratorNet(torch.nn.Module):
    """
    Class GeneratorNet.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        super(GeneratorNet, self).__init__()
        self.n_features = 100
        self.n_out = 784
        
        self.__model_fn()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        
        
    def __model_fn(self):
        """
        Specifies the network.
        """
        self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, self.n_out),
            nn.Tanh()
        )


    def forward(self, X):
        """
        Performs a forward-pass on the data.
        
        :param X:           network input
        """
        X = self.hidden0(X)
        X = self.hidden1(X)
        X = self.hidden2(X)
        X = self.out(X)
        
        return X


# -----------------------------------------------------------------------------
# Class GAN
# -----------------------------------------------------------------------------

class GAN():
    """
    Class GAN (Generative Adversarial Network).
    """
    
    def __init__(self):
        """
        Constructor.
        """
        self.discriminator = DiscriminatorNet()
        self.generator = GeneratorNet()
        
        self.loss = nn.BCELoss()
    
    
    def train(self, X, b_size=128, n_epochs=200):
        """
        Trains the GAN network.
        """
        self.n_epochs = n_epochs
        # create loader with data, so that we can iterate over it
        data_loader = torch.utils.data.DataLoader(X, batch_size=b_size, shuffle=True)
        # num batches
        num_batches = len(data_loader)
        
        # generate some samples to see progress
        num_test_samples = 16
        test_noise = self.__noise(num_test_samples)
        
        # perform training iterations
        for epoch in range(self.n_epochs):
            # go over all batches
            for i, (real_batch, _) in enumerate(data_loader):
                # train discriminator
                X_real = Variable(self.__images_to_vectors(real_batch))
                # generate fake data and detach 
                # (so that gradients are not calculated for generator)
                X_fake = self.generator(self.__noise(real_batch.size(0))).detach()
                
                # train discriminator
                # -------------------------------------------------------------
                d_error = self.__train_discriminator(X_real, X_fake)
        
                # train generator
                # -------------------------------------------------------------
                # generate fake data
                X_fake = self.generator(self.__noise(real_batch.size(0)))
                # train generator
                g_error = self.__train_generator(X_fake)
                
                # print progress
                # -------------------------------------------------------------
                # display progress every few batches
                if i % 100 == 0: 
                    test_images = self.__vectors_to_images(
                        self.generator(test_noise))
                    test_images = test_images.data
                    
                    self.__log_images(test_images, num_test_samples)
                    # display status logs
                    self.__display_status(
                        epoch, self.n_epochs, i, num_batches,
                        d_error, g_error)


    def __train_discriminator(self, X_real, X_fake):
        """
        Trains the disciminator network.
        
        :param X_real:      real data
        :param X_fake:      fake data
        :return:            discriminator error
        """
        # reset gradients
        self.discriminator.optimizer.zero_grad()
        
        # train on real data
        prediction_real = self.discriminator(X_real)
        # calculate error and backpropagate
        error_real = self.loss(prediction_real, self.__ones_target(X_real.size(0)))
        error_real.backward()
    
        # train on fake data
        prediction_fake = self.discriminator(X_fake)
        # calculate error and backpropagate
        error_fake = self.loss(prediction_fake, self.__zeros_target(X_real.size(0)))
        error_fake.backward()
        
        # update weights with gradients
        self.discriminator.optimizer.step()
        
        # return error and predictions for real and fake inputs
        return error_real + error_fake


    def __train_generator(self, X_fake):
        """
        Trains the generator network.
        
        :param X_fake:      fake data
        :return:            generator error
        """
        # reset gradients
        self.generator.optimizer.zero_grad()
        
        # sample noise and generate fake data
        prediction = self.discriminator(X_fake)
        # calculate error and backpropagate
        error = self.loss(prediction, self.__ones_target(X_fake.size(0)))
        error.backward()
        
        # update weights with gradients
        self.generator.optimizer.step()
        
        return error
    
    
    def __images_to_vectors(self, images):
        """
        Converts images to vectors.
        
        :param images:      images to be converted
        :return:            vectors of images
        """
        return images.view(images.size(0), 784)
    
    
    def __vectors_to_images(self, vectors):
        """
        Converts vectors to images.
        
        :param vectors:     vectors to be converted
        :return:            images
        """
        return vectors.view(vectors.size(0), 1, 28, 28)
    
    
    def __noise(self, size):
        """
        Generates a 1d vector of gaussian sampled random values.
        
        :param size:        size of the random vector
        :return:            random vector
        """
        n = Variable(torch.randn(size, 100))
        
        return n
    
    
    def __ones_target(self, size):
        """
        Generates a tensor containing ones.
        
        :param size:        size of the vector
        :return:            vector of ones
        """
        data = Variable(torch.ones(size, 1))
        
        return data
    
    
    def __zeros_target(self, size):
        """
        Generates a tensor containing zeroes.
        
        :param size:        size of the vector
        :return:            vector of zeroes
        """
        data = Variable(torch.zeros(size, 1))
        
        return data
    
    
    def __log_images(self, images, n_images, format="NCHW"):
        """
        Displays the images.
        
        :param images:      images
        :param n_images:    number of images
        :param format:      format of the images
        """
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format == "NHWC":
            images = images.transpose(1,3)
        
        # make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=True, scale_each=True)
        # make vertical grid from image tensor
        nrows = int(np.sqrt(n_images))
        vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

        # plot and save horizontal
        plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis("off")
        display.display(plt.gcf())
        plt.close()


    def __display_status(self, epoch, n_epochs, batch, n_batches, d_error, g_error):
        """
        Prints the current status of the GAN network.
        
        :param epoch:       current epoch
        :param n_epochs:    total number of epochs
        :param batch:       current batch
        :param n_batches:   total number of batches
        :param d_error:     discriminator error
        :param g_error:     generator error
        """
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        
        print("Epoch: [{}/{}], Batch Num: [{}/{}]"\
            .format(epoch, n_epochs, batch, n_batches))
        print("Discriminator Loss: {:.4f}, Generator Loss: {:.4f}" \
            .format(d_error, g_error))
        
   
# -----------------------------------------------------------------------------
# Download data
# -----------------------------------------------------------------------------
        
def get_data():
    """
    Downloads the data.
    
    :return:                data set
    """
    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    out_dir = "./dataset"
    
    return datasets.MNIST(root=out_dir, train=True,
        transform=transform, download=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
if __name__ == "__main__":
    X = get_data()
    
    gan = GAN()
    gan.train(X, b_size=128, n_epochs=200)

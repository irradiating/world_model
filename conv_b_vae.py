# coding: utf8

import numpy as np
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, input_shape, vector_size):
        super(ConvVAE,self).__init__()

        self.channel = input_shape[0]
        self.h = input_shape[1]
        self.w = input_shape[2]

        ## Encoder
        self.conv1 = nn.Conv2d(in_channels=self.channel, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)

        ## latent vector
        ## representation of mean(mu) and std(logvar) # 256*6*6
        self.mean_variation = nn.Linear(in_features=9216, out_features=vector_size)
        self.std_variation = nn.Linear(in_features=9216, out_features=vector_size)
        self.vector = nn.Linear(in_features=vector_size, out_features=9216)

        ## Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=9216, out_channels=128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=6, stride=2)
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=self.channel, kernel_size=6, stride=2)


    def encode(self, x):
        x = F.relu( self.conv1(x) ) # [1, 32, 63, 63]
        x = F.relu( self.conv2(x) ) # [1, 64, 30, 30]
        x = F.relu( self.conv3(x) ) # [1, 128, 14, 14]
        x = F.relu( self.conv4(x) ) # [1, 256, 6, 6]

        # flatten
        x = x.view(-1, 9216) # [1, 9216]

        return self.mean_variation(x), self.std_variation(x) # mu, logvar


    def reparametrize(self, mu, logvar):
        std = torch.exp( 0.5 * logvar ) # [1, 4608]
        eps = torch.randn_like(std) # [1, 4608]

        return eps * std + mu


    def decode(self, x):
        x = self.vector(x).view(-1, 9216, 1, 1) # [1, 9216, 1, 1]
        x = F.relu( self.deconv1(x) ) # [1, 128, 5, 5]
        x = F.relu( self.deconv2(x) ) # [1, 64, 13, 13]
        x = F.relu( self.deconv3(x) ) # [1, 32, 29, 29]
        x = F.relu( self.deconv4(x) ) # [1, 16, 62, 62]
        x = torch.sigmoid( self.deconv5(x) ) # [1, 3, 128, 128]

        return x

    def forward(self, x, encode=False, mean=False):
        mu, logvar = self.encode(x) # [1, 4608] [1, 4608]

        z = self.reparametrize(mu, logvar) # [1, 4608]

        if encode:
            if mean:
                return mu
            return z

        return self.decode(z), mu, logvar, z


    def conv_vae_loss(self, deconv_img, img, mu, logvar):
        BETA = 4.0

        batch_size = img.size(0) # 1

        loss = F.binary_cross_entropy( deconv_img, img, size_average=False )
        # loss = F.binary_cross_entropy(deconv_img, img)

        kld = -0.5 * torch.sum( 1 + logvar - mu.pow(2) - logvar.exp() )

        loss /= batch_size
        kld /= batch_size

        return loss + BETA * kld.sum()





if __name__ == "__main__":
    convvae = ConvVAE(input_shape=(3,128,128), vector_size=4608)
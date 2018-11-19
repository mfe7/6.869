# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - CycleGenerator    --> Used in the CycleGAN
#   - DCDiscriminator   --> Used in CycleGAN
#
# For the assignment, you are asked to create the architecture of the CycleGAN network by
# filling in the __init__ methods in the CycleGenerator class. We have filled in the architechure
# for the DCDiscriminator class for you.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)



class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=4)

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim=conv_dim*2)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(in_channels=conv_dim*2, out_channels=conv_dim, kernel_size=4)
        self.deconv2 = deconv(in_channels=conv_dim, out_channels=3, kernel_size=4, batch_norm=False)

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        out = F.relu(self.conv1(x))            # BS x 32 x 16 x 16
        out = F.relu(self.conv2(out))          # BS x 64 x 8 x 8

        out = F.relu(self.resnet_block(out))   # BS x 64 x 8 x 8

        out = F.relu(self.deconv1(out))        # BS x 32 x 16 x 16
        out = F.tanh(self.deconv2(out))        # BS x 3 x 32 x 32

        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64):
        super(DCDiscriminator, self).__init__()

        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4)
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4)
        self.conv4 = conv(in_channels=conv_dim * 4, out_channels=1, kernel_size=4, stride=1, padding=0,
                          batch_norm=False)

    def forward(self, x):

        out = F.relu(self.conv1(x))    # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))  # BS x 128 x 8 x 8
        out = F.relu(self.conv3(out))  # BS x 256 x 4 x 4

        out = self.conv4(out).squeeze()
        out = F.sigmoid(out)
        return out

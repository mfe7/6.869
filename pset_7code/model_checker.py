# CSC 321, Assignment 4
#
# This is a script to check whether the outputs of your CycleGenerator, DCDiscriminator, and
# CycleGenerator models produce the expected outputs.
#
# NOTE THAT THIS MODEL CHECKER IS PROVIDED FOR CONVENIENCE ONLY, AND MAY PRODUCE FALSE NEGATIVES.
# DO NOT USE THIS AS THE ONLY WAY TO CHECK THAT YOUR MODEL IS CORRECT.
#
# Usage:
# ======
#
#    python model_checker.py
#

import os
import pdb
import pickle
import argparse

import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

# Numpy
import numpy as np

# Local imports
import utils
from models import DCGenerator, DCDiscriminator, CycleGenerator


RANDOM_SEED = 11


def get_emoji_loader(emoji_type):
    """Creates training and test data loaders.
    """
    transform = transforms.Compose([
                    transforms.Scale(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_path = os.path.join('./emojis', emoji_type)
    test_path = os.path.join('./emojis', 'Test_{}'.format(emoji_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    return train_dloader, test_dloader


def count_parameters(model):
    """Finds the total number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seeds(seed):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def sample_noise(dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (1, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return Variable(torch.rand(1, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def check_dc_generator():
    """Checks the output and number of parameters of the DCGenerator class.
    """
    set_random_seeds(RANDOM_SEED)
    G = DCGenerator(noise_size=100, conv_dim=32)
    noise = sample_noise(100)
    output = G(noise)
    output_np = output.data.cpu().numpy()

    dc_generator_expected = np.load('checker_files/dc_generator.npy')

    if np.allclose(output_np, dc_generator_expected):
        print('DCGenerator output: EQUAL')
    else:
        print('DCGenerator output: NOT EQUAL')

    num_params = count_parameters(G)
    expected_params = 370624

    print('DCGenerator #params = {}, expected #params = {}, {}'.format(
          num_params, expected_params, 'EQUAL' if num_params == expected_params else 'NOT EQUAL'))

    print('-' * 80)


def check_dc_discriminator():
    """Checks the output and number of parameters of the DCDiscriminator class.
    """
    set_random_seeds(RANDOM_SEED)
    D = DCDiscriminator(conv_dim=32)

    dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type='Apple')
    images, labels = iter(dataloader_X).next()
    images = Variable(images)

    output = D(images)
    output_np = output.data.cpu().numpy()

    dc_discriminator_expected = np.load('checker_files/dc_discriminator.npy')

    if np.allclose(output_np, dc_discriminator_expected):
        print('DCDiscriminator output: EQUAL')
    else:
        print('DCDiscriminator output: NOT EQUAL')

    num_params = count_parameters(D)
    expected_params = 167872

    print('DCDiscriminator #params = {}, expected #params = {}, {}'.format(
          num_params, expected_params, 'EQUAL' if num_params == expected_params else 'NOT EQUAL'))

    print('-' * 80)


def check_cycle_generator():
    """Checks the output and number of parameters of the CycleGenerator class.
    """
    set_random_seeds(RANDOM_SEED)
    G_XtoY = CycleGenerator(conv_dim=32, init_zero_weights=False)

    dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type='Apple')
    images, labels = iter(dataloader_X).next()
    images = Variable(images)

    output = G_XtoY(images)
    output_np = output.data.cpu().numpy()

    # np.save('checker_files/cycle_generator.npy', output_np)
    cycle_generator_expected = np.load('checker_files/cycle_generator.npy')

    if np.allclose(output_np, cycle_generator_expected):
        print('CycleGenerator output: EQUAL')
    else:
        print('CycleGenerator output: NOT EQUAL')

    num_params = count_parameters(G_XtoY)
    expected_params = 105856

    print('CycleGenerator #params = {}, expected #params = {}, {}'.format(
          num_params, expected_params, 'EQUAL' if num_params == expected_params else 'NOT EQUAL'))

    print('-' * 80)



if __name__ == '__main__':

    try:
        check_dc_generator()
    except:
        print('Crashed while checking DCGenerator. Maybe not implemented yet?')

    try:
        check_dc_discriminator()
    except:
        print('Crashed while checking DCDiscriminator. Maybe not implemented yet?')

    try:
        check_cycle_generator()
    except:
        print('Crashed while checking CycleGenerator. Maybe not implemented yet?')

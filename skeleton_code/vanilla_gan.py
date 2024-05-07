#  COMP 6211D & ELEC 6910T , Assignment 3
#
# This is the main training file for the vanilla GAN part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


# Local imports
from data_loader import get_emoji_loader
from models import CycleGenerator, DCDiscriminator
from vanilla_utils import create_dir, create_model, checkpoint, sample_noise, save_samples

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(train_loader, opts, device):
    
    G, D = create_model(opts)
    
    G.to(device)
    D.to(device)
    
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    # d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.SGD(D.parameters(), opts.lr, momentum=0.5)

    fixed_noise = sample_noise(opts.batch_size, opts.noise_size).to(device)
    
    iteration = 1
    
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()
    total_train_iters = opts.num_epochs * len(train_loader)
    
    D_real_losses=[]
    D_fake_losses=[]
    D_total_losses=[]
    G_losses=[]
    iters=[]

    for epoch in range(opts.num_epochs):

        for batch in train_loader:

            real_images = batch[0].to(device)

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################

            d_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Compute the discriminator loss on real images
            # fill ground_truth tensor with all 1's since all images are real. real_images.size(0) accesses the batch size.
            ground_truth = torch.ones(real_images.size(0), device = device);
            # calculate the loss between real images and ground truth. D(real_images) should output a tensor of all 1's ideally
            D_real_loss = bce_loss(D(real_images), ground_truth);
            

            # 2. Sample noise
            # sample same number of noise as the batch size
            noise = sample_noise(opts.batch_size, opts.noise_size).to(device)
            

            # 3. Generate fake images from the noise
            fake_images = G(noise);
            

            # 4. Compute the discriminator loss on the fake images
            # fill fake_ground_truth tensor with all 0's since all images are generated. real_images.size(0) accesses the batch size.
            fake_ground_truth = torch.zeros(real_images.size(0), device = device);
            # calculate the loss between generated images and the ground truth. D(fake_images) should output a tensor of all 0's ideally
            # use detach() to avoid back propagation affecting the generator during discriminator training phase
            D_fake_loss = bce_loss(D(fake_images.detach()), fake_ground_truth);
            
            # 5. Compute the total discriminator loss
            D_total_loss = (D_fake_loss + D_real_loss)/2;
    
            D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Sample noise
            noise = sample_noise(opts.batch_size, opts.noise_size).to(device)
            
            # 2. Generate fake images from the noise
            fake_images = G(noise);
            
            # 3. Compute the generator loss
            fake_real_labels = torch.ones(fake_images.size(0), device=device);
            # allow the gradient to flow through both generator and discriminator
            # generator needs to know how to adjust its param to produce realistic images based on gradient info from the discriminator
            # therefore, no detach() needed here
            G_loss = bce_loss(D(fake_images), fake_real_labels);

            # the weight update here will not affect discriminator since only generator is targeted
            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))
                D_real_losses.append(D_real_loss.item());
                D_fake_losses.append(D_fake_loss.item());
                D_total_losses.append(D_total_loss.item());
                G_losses.append(G_loss.item());
                iters.append(iteration);

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1

    # Plotting the losses after the training loop
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(iters, D_real_losses, label="D Real Loss")
    plt.plot(iters, D_fake_losses, label="D Fake Loss")
    plt.plot(iters, D_total_losses, label="D Total Loss")
    plt.plot(iters, G_losses, label="G Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('GANLossRes.png')


    
    
    
def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    train_loader, _ = get_emoji_loader(opts.emoji, opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)
    
    if torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    train(train_loader, opts, device)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=100) # default 50
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.') # default was 16
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--emoji', type=str, default='Apple', choices=['Apple', 'Facebook', 'Windows'], help='Choose the type of emojis to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./samples_vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=20) #default = 200
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size

    print(opts)
    main(opts)


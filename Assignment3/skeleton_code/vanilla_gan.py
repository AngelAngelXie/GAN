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
import torch.nn.functional as F
import torchvision

# visualization
import matplotlib.pyplot as plt

# Local imports
from data_loader import get_emoji_loader
from models import CycleGenerator, DCDiscriminator
from vanilla_utils import create_dir, create_model, checkpoint, sample_noise, save_samples

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(train_loader, opt, device):

    D_real_losses=[]
    D_fake_losses=[]
    D_total_losses=[]
    G_losses=[]
    iters=[]
    genRes=[]
    
    G, D = create_model(opts)
    
    G.to(device)
    D.to(device)
    
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])
    
    fixed_noise = torch.randn(opt.batch_size, opt.noise_size, 1, 1, device=device)
    
    iteration = 1
    
    mse_loss = torch.nn.MSELoss()
#     bce_loss = torch.nn.BCELoss()
    loss_fn = F.binary_cross_entropy
    total_train_iters = opts.num_epochs * len(train_loader)
    
    for epoch in range(opts.num_epochs):

        for batch in train_loader:

            real_images = batch[0].to(device)

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################

            d_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Compute the discriminator loss on real images
            # real label = 1
            real_labels = torch.ones(opt.batch_size, device=device)
            D_real_loss = loss_fn(D(real_images), real_labels)

            # 2. Sample noise
            noise = torch.randn(opt.batch_size, opt.noise_size, 1, 1, device=device)
            
            # 3. Generate fake images from the noise
            fake_images = G(noise)
            fake_labels = torch.zeros(opt.batch_size, device=device)
            flipped_fake_labels = real_labels

            # 4. Compute the discriminator loss on the fake images
            D_fake_loss = loss_fn(D(fake_images.detach()), fake_labels)
            

            # 5. Compute the total discriminator loss
            D_total_loss = 0.5 * (D_fake_loss + D_real_loss)
    
            D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Sample noise
            noise = torch.randn(opt.batch_size, opt.noise_size, 1, 1, device=device)

            # 2. Generate fake images from the noise
            fake_images = G(noise)
            
            # 3. Compute the generator loss
            G_loss = loss_fn(D(fake_images), flipped_fake_labels)

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
            # if iteration % opts.sample_every == 0:
                

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1

        with torch.no_grad():
            fake_images = G(fixed_noise).detach().cpu()
            genRes.append(
                torchvision.utils.make_grid(fake_images, padding=2, normalize=True))


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

    for i in range(0, opts.num_epochs, 5):
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.title(f'Generated images at epoch {i}')
        plt.imshow(np.transpose(genRes[i], (1, 2, 0)))
        plt.show()
        
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.title(f'Generated images after last epoch')
        plt.imshow(np.transpose(genRes[-1], (1, 2, 0)))
        plt.savefig('{i}.png')
    
    
    
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
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128, help='The number of images in a batch.')
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
    parser.add_argument('--sample_every', type=int , default=50)
    parser.add_argument('--checkpoint_every', type=int , default=50)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size

    print(opts)
    main(opts)


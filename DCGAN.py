# Adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from datetime import datetime

seed = 42069  # Word?

batch_size = 16
image_size = 32
# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 32

# Size of feature maps in discriminator
ndf = 32

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# dataset = dset.MNIST(root="", train=True, download=True,
#                     transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]))
#
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                          shuffle=True, num_workers=0)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# real_batch = next(iter(dataloader))
# example = real_batch[0][0]
# save_image(example, "training_example.png")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class DCGAN(object):
    def __init__(self, modelG=Generator(0), modelD=Discriminator(0)):
        self.generator = modelG
        self.discriminator = modelD
        self.optimizerG = optim.Adam(modelG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerD = optim.Adam(modelD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.G_losses = []
        self.D_losses = []
        self.img_list = []
        self.criterion = nn.BCELoss()

    def train(self, epochs, dataloader):
        real_label = 1.
        fake_label = 0.
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        iters = 0

        for epoch in range(epochs):
            start = time.time()
            for i, data in enumerate(dataloader, 0):
                ## Train with all-real batch
                self.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,),   real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                self.generator.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    self.img_list.append(fake)

                iters += 1
            print("time for epoch: " + str(time.time() - start))

    def save(self, fp=None):
        if fp is None:
            fp = ""
        else:
            fp += "/dcgan_epochs{0}".format(datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))

        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_generator_state_dict': self.optimizerG.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizerD.state_dict(),
            'generator_loss': self.G_losses,
            'discriminator_loss': self.D_losses,
            'img_list': self.img_list
        }, fp)

    def load(self, fp):
        state = torch.load(fp)
        self.generator.load_state_dict(state['generator_state_dict'])
        self.discriminator.load_state_dict(state["discriminator_state_dict"])
        self.optimizerD.load_state_dict(state['optimizer_discriminator_state_dict'])
        self.optimizerG.load_state_dict(state['optimizer_generator_state_dict'])
        self.G_losses = state['generator_loss']
        self.D_losses = state['discriminator_loss']

        if 'img_list' in state.keys():
            self.img_list = state['img_list']
        else:
            self.img_list = []



if __name__ == "__main__":
    dataset = dset.MNIST(root="", train=True, download=True,
                         transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    dcgan = DCGAN(netG, netD)

    for i in range(5):
        dcgan.train(10, dataloader)
        dcgan.save("MNIST_playground_output")

    print(dcgan.D_losses)
    print(dcgan.G_losses)

    # for img in (dcgan.img_list):
    #     save_image(img, "MNIST_playground_output/epoch_" + str(i) + ".png")
    #     i += 1
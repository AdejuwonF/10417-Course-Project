import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as dset
from PIL import Image
import util
import time
from torch import optim
from datetime import datetime

nz = 100
ngf = 32
ndf = 32
nc = 1
batch_size = 128
image_size = 32
workers = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        """self.fc = nn.Sequential(
           nn.Linear(100, ngf*4*4*4)
        )"""
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
        self.main.apply(weights_init)
    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # # inputs is 3 x 32 x 32 image
            # nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ndf),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size is ndf x 16 x 16
            # nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(2*ndf),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size is 2*ndf x 8 x 8
            # nn.Conv2d(in_channels=2*ndf, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(4*ndf),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size is 4*ndf x 4 x 4
            # nn.AvgPool2d(4)
            # # state size is 4*ndf x 1 x 1
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
            # nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            nn.AvgPool2d(4)
        )
        self.fc = nn.Sequential(
            nn.Linear(4*ndf, 1),
        )
        self.main.apply(weights_init)
        self.fc.apply(weights_init)


    def forward(self, x):
        out = self.main(x)
        out = self.fc.forward(out.view(out.shape[0],-1))
        return out



class DiscriminatorWP(nn.Module):

    def __init__(self):
        super(DiscriminatorWP, self).__init__()
        self.main = nn.Sequential(
            # inputs is 3 x 32 x 32 image
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(normalized_shape=[ndf,image_size//2,image_size//2]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is ndf x 16 x 16
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([2*ndf,image_size//4,image_size//4]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is 2*ndf x 8 x 8
            nn.Conv2d(in_channels=2*ndf, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([4*ndf,image_size//8,image_size//8]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is 4*ndf x 4 x 4
            nn.AvgPool2d(4)
            # state size is 4*ndf x 1 x 1
        )
        self.fc = nn.Sequential(
            nn.Linear(4*ndf, 1),
        )
        self.main.apply(weights_init)
        self.fc.apply(weights_init)


    def forward(self, x):
        out = self.main(x)
        out = self.fc.forward(out.view(out.shape[0],-1))
        return out

class WGAN(object):
    def __init__(self, modelG=Generator(), modelD=Discriminator(), clamp=.01):
        self.generator = modelG
        self.discriminator = modelD
        self.optimG = optim.RMSprop(modelG.parameters(), .0005)
        self.optimD = optim.RMSprop(modelD.parameters(), .0005)
        self.clamp = clamp
        self.G_training_loss = []
        self.D_training_loss = []
        self.img_list = []
        self.epochs = 0

    def train(self, num_epochs, dataloader):
        self.generator.train()
        self.discriminator.train()
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        for epoch in range(num_epochs):
            start = time.time()
            self.epochs += 1
            for i, data in enumerate(dataloader):
                real_samples = data[0]
                self.optimD.zero_grad()
                fake_samples = self.generator.forward(torch.randn(real_samples.shape[0], nz, 1, 1))
                fake = self.discriminator.forward(fake_samples)
                real = self.discriminator.forward(real_samples)
                d_loss = torch.mean(fake) - torch.mean(real)
                d_loss.backward()
                self.optimD.step()

                # Weight clipping
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clamp, self.clamp)

                fake_samples = self.generator.forward(torch.randn(batch_size, nz, 1, 1))
                g_loss = -torch.mean(self.discriminator.forward(fake_samples))
                if i % 5 == 0:
                    # print("Batch:{0} of Epoch:{1}".format(i, epoch))
                    self.optimG.zero_grad()
                    self.optimD.zero_grad()
                    g_loss.backward()
                    self.optimG.step()

            print(time.time() - start)
            self.G_training_loss.append(g_loss.item())
            self.D_training_loss.append(d_loss.item())
            print("Epoch:{0}\nGenerator Loss:{1}\nDiscriminator Loss:{2}".format(epoch, self.G_training_loss[-1],
                                                                                 self.D_training_loss[-1]))
            with torch.no_grad():
                fake = self.generator(fixed_noise).detach().cpu()
            self.img_list.append(fake)

    def save(self, fp=None):
        if fp is None:
            fp = "model_checkpoints/poke_gan_wc_{0}_epochs_{1}"\
                .format(self.epochs, datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))
        else:
            fp += "/gan_wc_{0}_epochs{1}".format(self.epochs, datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))

        torch.save({
            "epochs_trained": self.epochs,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_generator_state_dict': self.optimG.state_dict(),
            'optimizer_discriminator_state_dict': self.optimD.state_dict(),
            'generator_loss': self.G_training_loss,
            'discriminator_loss': self.D_training_loss,
            'clamp': self.clamp
        }, fp)

    def load(self, fp):
        state = torch.load(fp)
        self.epochs = state['epochs_trained']
        self.generator.load_state_dict(state['generator_state_dict'])
        self.discriminator.load_state_dict(state["discriminator_state_dict"])
        self.optimD.load_state_dict(state['optimizer_discriminator_state_dict'])
        self.optimG.load_state_dict(state['optimizer_generator_state_dict'])
        self.G_training_loss = state['generator_loss']
        self.D_training_loss = state['discriminator_loss']
        self.clamp = state['clamp']


class WGAN_GP(object):
    def __init__(self, modelG=Generator(), modelD=DiscriminatorWP(), lamb=10):
        self.generator = modelG
        self.discriminator = modelD
        self.optimG = optim.RMSprop(modelG.parameters(), .0005)
        self.optimD = optim.RMSprop(modelD.parameters(), .0005)
        self.G_training_loss = []
        self.D_training_loss = []
        self.epochs = 0
        self.lamb = lamb

    def train(self, num_epochs, dataloader):
        self.generator.train()
        self.discriminator.train()
        for epoch in range(num_epochs):
            start = time.time()
            self.epochs += 1
            for i, data in enumerate(dataloader):
                real_samples = data[0]
                self.optimD.zero_grad()
                fake_samples = self.generator.forward(torch.randn(real_samples.shape[0], nz, 1, 1))
                d_loss = torch.mean(self.discriminator.forward(fake_samples)) - torch.mean(self.discriminator.forward(real_samples))

                # gradient Penalty
                epsilon = torch.rand(real_samples.shape[0], 1, 1, 1)
                x_hat = (epsilon*real_samples + (1-epsilon)*fake_samples)
                out = self.discriminator.forward(x_hat)
                x_grad = torch.autograd.grad(outputs=out, inputs=x_hat, grad_outputs=torch.ones(real_samples.shape[0],1),
                                             create_graph=True, retain_graph=True, only_inputs=True)[0]
                gp = self.lamb*torch.mean(torch.pow((torch.sqrt(torch.sum(torch.pow(x_grad, 2), dim=[1,2,3])) - 1),2))
                d_loss += gp

                self.optimD.zero_grad()
                d_loss.backward()
                self.optimD.step()

                fake_samples = self.generator.forward(torch.randn(batch_size, nz, 1, 1))
                g_loss = -torch.mean(self.discriminator.forward(fake_samples))
                if i % 5 == 0:
                    print("Batch:{0} of Epoch:{1}".format(i, epoch))
                    self.optimG.zero_grad()
                    self.optimD.zero_grad()
                    g_loss.backward()
                    self.optimG.step()

            print(time.time() - start)
            self.G_training_loss.append(g_loss.item())
            self.D_training_loss.append(d_loss.item())
            print("Epoch:{0}\nGenerator Loss:{1}\nDiscriminator Loss:{2}".format(epoch, self.G_training_loss[-1],
                                                                                 self.D_training_loss[-1]))

    def save(self, fp=None):
        if fp is None:
            fp = "model_checkpoints/poke_gan_gp_{0}_epochs_{1}"\
                .format(self.epochs, datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))
        else:
            fp += "/gan_gp_{0}_epochs{1}".format(self.epochs, datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))

        torch.save({
            "epochs_trained": self.epochs,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_generator_state_dict': self.optimG.state_dict(),
            'optimizer_discriminator_state_dict': self.optimD.state_dict(),
            'generator_loss': self.G_training_loss,
            'discriminator_loss': self.D_training_loss,
            'GP weight': self.lamb
        }, fp)

    def load(self, fp):
        state = torch.load(fp)
        self.epochs = state['epochs_trained']
        self.generator.load_state_dict(state['generator_state_dict'])
        self.discriminator.load_state_dict(state["discriminator_state_dict"])
        self.optimD.load_state_dict(state['optimizer_discriminator_state_dict'])
        self.optimG.load_state_dict(state['optimizer_generator_state_dict'])
        self.G_training_loss = state['generator_loss']
        self.D_training_loss = state['discriminator_loss']
        self.lamb = state['GP weight']

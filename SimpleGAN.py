"""
Simple GAN using fully connected layers

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

import wandb

wandb.init(entity='fabacha22', project='GAN')

os.makedirs("images_wgangpcifar5", exist_ok=True)
#Hyperparameters

cuda = True if torch.cuda.is_available() else False
lr =3e-4
z_dim =64
image_dim = 3072 #32*32*3
batch_size =32
num_epochs =50

class Generator(nn.Module):
    def __init__(self, z_dim,img_dim):
        super().__init__()
        self.gen= nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim),
            nn.Tanh()
        )

    def forward(self,x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc=nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward (self,x):
        return self.disc(x)

# Initialize generator and discriminator
gen = Generator(z_dim,image_dim).cuda()
disc = Discriminator(image_dim).cuda()


fixed_noise = torch.randn((batch_size, z_dim)).cuda()

# Configure data loader
os.makedirs("data/cifar10", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])))


# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=lr)
opt_disc = optim.Adam(disc.parameters(), lr=lr)

#Loss
criterion =nn.BCELoss()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#  Training
# ----------


for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):

    # Configure input
        real = real.view(-1,3072).cuda()
        batch_size = real.shape[0]

    # ---------------------
    #  Train Discriminator: max log (D(real)) + log (1-D(G(z))
    # ---------------------
        noise =torch.randn(batch_size,z_dim).cuda()
        fake =gen(noise)
        disc_real=disc(real).view(-1)
        lossD_real =criterion(disc_real, torch.ones_like(disc_real) )
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_real, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) /2 #Instead of detach above, we could use lossD.backward(retrain_graph =True)
        wandb.log({'Discriminator Loss': lossD.item()})
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()


        #Train Generator min log(1-D(G(z))) #This brings the problem of saturated gradients so instead we will do max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        wandb.log({'Generator Loss': lossG.item()})
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 3, 32, 32)
                data = real.reshape(-1, 3, 32, 32)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                # Log generated and real images using WandB
                wandb.log({"Cifar10 Fake Images": [wandb.Image(img_grid_fake, caption="Fake Images")],
                           "Cifar10 Real Images": [wandb.Image(img_grid_real, caption="Real Images")]})

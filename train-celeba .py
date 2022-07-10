import argparse
import os
import torch.nn as nn
import torch.cuda
from torch import autograd
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.utils import make_grid, save_image

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)

opt = parser.parse_args()

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

if not os.path.exists('./results/'):
    os.makedirs('./results/')
if not os.path.exists('./logs/'):
    os.makedirs('./logs/')
if not os.path.exists('./checkpoint/'):
    os.makedirs('./checkpoint/')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(

            nn.ConvTranspose2d(opt.latent_dim, opt.ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 2, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.init_weight()

    def forward(self, x):
        return self.model(x)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


netG = Generator().to(device)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(opt.channels, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(opt.ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(opt.ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(opt.ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()  # WGAN论文提出去掉该层
        )

        self.init_weight()

    def forward(self, x):
        return self.model(x)
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


netD = Discriminator().to(device)


# Root directory for dataset
dataroot = "./data/celeba"
dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(opt.img_size),
                                transforms.CenterCrop(opt.img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=opt.n_cpu)

optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).repeat(1, opt.channels, opt.img_size, opt.img_size).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)  # 插值图像
    d_interpolates = D(interpolates)  # 判别器对插值图像的输出
    d_interpolates = d_interpolates.view(real_samples.size(0), -1)
    gradients = autograd.grad(outputs=d_interpolates,
                                inputs=interpolates,
                                grad_outputs=torch.ones_like(d_interpolates),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Loss weight for gradient penalty
lambda_gp = 10

iters = 0
print("starting training loop...")

for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):

        # Configure input
        data = data[0].to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        netD.train()
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(data.size(0), opt.latent_dim, 1, 1).to(device)

        # Generate a batch of images
        fake = netG(z)

        # Real images
        real_validity = netD(data)
        real_validity = real_validity.view(data.size(0), -1)
        # Fake images
        fake_validity = netD(fake)
        fake_validity = fake_validity.view(data.size(0), -1)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(netD, data.data, fake.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        netG.train()
        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake = netG(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = netD(fake)
            fake_validity = fake_validity.view(data.size(0), -1)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )


            iters += opt.n_critic

        if iters % 500 == 0:
            noise = torch.randn(64, opt.latent_dim, 1, 1, device=device)
            with torch.no_grad():
                netG.eval()
                fake = netG(noise).detach().cpu()
                fake = make_grid((fake * 0.5 + 0.5), 8, 0)
                save_image(fake, './results/' + 'gen_{}.jpg'.format(iters))

        

    if epoch % 10 == 0:
        torch.save({'G': netG.state_dict(),
                    'D': netD.state_dict(),
                    'optimizerG': optimizer_G.state_dict(),
                    'optimizerD': optimizer_D.state_dict(),
                    'epoch': epoch,
                    'total_step': iters}, './checkpoint/' + 'param_{}.pth'.format(epoch))



























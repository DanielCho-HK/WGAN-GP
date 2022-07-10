import torch.nn as nn
import torch
import argparse
from torchvision.utils import make_grid, save_image

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--ngf", type=int, default=64)


opt = parser.parse_args()

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

    def forward(self, x):
        return self.model(x)




netG = Generator().cuda()
params = torch.load("./checkpoint/param_190.pth")
netG.load_state_dict(params['G'])

netG.eval()
z = torch.randn(4, opt.latent_dim, 1, 1).cuda()
fake = netG(z).detach().cpu()
fake = make_grid((fake * 0.5 + 0.5), 2, 0)
save_image(fake, 'gen.jpg')



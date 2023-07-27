import torch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

class VAEBEV(nn.Module):
    def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
        super(VAEBEV, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z)
        self.fc2 = nn.Linear(h_dim, z)
        self.fc3 = nn.Linear(z, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch * 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 2, channel_in, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def recon(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.recon(z), mu, logvar


class BEVLSTM(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size, batch_size, num_layers, vae=None):
        super().__init__()
        self.vae = vae
        self.lstm = nn.LSTM(latent_size + action_size, hidden_size, batch_first=True)
        self.h_size = (num_layers, batch_size, hidden_size)
        self.init_hs()

    def init_hs(self):
        self.h_0 = Variable(torch.randn(self.h_size)).to(device)
        self.c_0 = Variable(torch.randn(self.h_size)).to(device)

    def encode(self, image):
        x = torch.reshape(image, (-1,) + image.shape[-3:])
        _, mu, logvar = self.vae(x)
        z = self.vae.reparameterize(mu, logvar)
        z = torch.reshape(z, image.shape[:2] + (-1,))
        return z, mu, logvar

    def decode(self, z):
        z_f = torch.reshape(z,  (-1,) + (z.shape[-1],))
        img = self.vae.recon(z_f)
        return torch.reshape(img, z.shape[:2] + img.shape[-3:])

    def forward(self, action, latent):
        in_al = torch.cat([torch.Tensor(action), latent], dim=-1)
        outs, _ = self.lstm(in_al.float(), (self.h_0, self.c_0))
        return outs
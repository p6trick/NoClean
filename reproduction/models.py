# Adapted from Joey Litalien noise2noise-pytorch (MIT), with archived project modifications.
# See LICENSES/noise2noise-pytorch-MIT.txt and THIRD_PARTY_NOTICES.md.
"""U-Net matching the archived noise2noise-finger/src/unet.py topology.

The unusual shared encoder/decoder blocks and first pooling stride of one are
intentional: these are the MNIST modifications found in the legacy code.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Denoiser(nn.Module):
    def __init__(self, first_pool=1):
        super().__init__()
        self._block1 = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(first_pool))
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        self._block4 = self.decoder_block(96)
        self._block5 = self.decoder_block(144)
        self._block6 = nn.Sequential(
            nn.Conv2d(99, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1), nn.LeakyReLU(0.1))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def decoder_block(channels):
        return nn.Sequential(
            nn.Conv2d(channels, 96, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

    def forward(self, x):
        p1 = self._block1(x)
        p2 = self._block2(p1)
        p3 = self._block2(p2)
        p4 = self._block2(p3)
        p5 = self._block2(p4)
        def match(value, reference):
            return F.interpolate(value, size=reference.shape[-2:], mode='nearest')
        u5 = match(self._block3(p5), p4)
        u4 = match(self._block4(torch.cat((u5, p4), dim=1)), p3)
        u3 = match(self._block5(torch.cat((u4, p3), dim=1)), p2)
        u2 = match(self._block5(torch.cat((u3, p2), dim=1)), p1)
        u1 = match(self._block5(torch.cat((u2, p1), dim=1)), x)
        return self._block6(torch.cat((u1, x), dim=1))


def grayscale(rgb):
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True).clamp(0, 1)

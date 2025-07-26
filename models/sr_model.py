import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class SuperResolutionNet(nn.Module):
    def __init__(self, scale_factor=2, channels=3, features=64, res_blocks=8):
        super().__init__()
        self.entry = nn.Conv2d(channels, features, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(features) for _ in range(res_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU()
        )
        self.exit = nn.Conv2d(features, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        x = self.exit(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, n_in, n_out, depth=4, n_base_channels=64):

        super().__init__()

        self.inc = DoubleConv(n_in, n_base_channels)
        self.downs = nn.ModuleList([Down(n_base_channels * (2 ** i), n_base_channels * (2 ** (i+1))) for i in range(depth)])
        self.ups = nn.ModuleList([Up(n_base_channels * (2 ** (i+1)), n_base_channels * (2 ** i)) for i in range(depth)][::-1])
        self.outc = nn.Conv2d(n_base_channels, n_out, kernel_size=1)

    def forward(self, x, PAN):
        
        if PAN == None:
            x = x
        else:
            x = torch.cat([x,PAN],dim = 1)
        xs = []
        x = self.inc(x)
        xs.append(x)
        for down in self.downs:
            x = down(x)
            xs.append(x)
        xs = xs[-2::-1]
        for i, up in enumerate(self.ups):
            x = up(x, xs[i])
        x = self.outc(x)

        return x

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.pool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.linear = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_res):

        x = self.linear(x)
        x_up = F.interpolate(x, scale_factor=2)
        x = torch.cat([x_res, x_up], dim=1)
        x = self.double_conv(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, in_channels, n_base_channels=64): 

        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, n_base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_base_channels, n_base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_base_channels * 2, n_base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_base_channels * 4, 1, 4, 1, 1)
        )

    def forward(self, x, y):

        inp = torch.cat([x, y], dim=1)
        
        return self.model(inp)
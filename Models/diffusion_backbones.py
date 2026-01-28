
import torch
import torch.nn as nn
import torch.nn.functional as F

## UNet

class UNet(nn.Module):

    def __init__(self, n_in, n_out, depth=4, n_base_channels=64):

        super().__init__()

        self.inc = DoubleConv(n_in, n_base_channels)
        self.downs = nn.ModuleList([Down(n_base_channels * (2 ** i), n_base_channels * (2 ** (i+1))) for i in range(depth)])
        self.ups = nn.ModuleList([Up(n_base_channels * (2 ** (i+1)), n_base_channels * (2 ** i)) for i in range(depth)][::-1])
        self.outc = nn.Conv2d(n_base_channels, n_out, kernel_size=1)

    def forward(self, x):
        
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):

        return self.conv(self.pool(x))

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
    
## UNet with CBN

class UNet_CBN(nn.Module):

    def __init__(self, n_in, n_out, n_conditions, depth=4, n_base_channels=64):

        super().__init__()

        self.inc = DoubleConv_CBN(n_in, n_base_channels, n_conditions)
        self.downs = nn.ModuleList([Down_CBN(n_base_channels * (2 ** i), n_base_channels * (2 ** (i+1)), n_conditions) for i in range(depth)])
        self.ups = nn.ModuleList([Up_CBN(n_base_channels * (2 ** (i+1)), n_base_channels * (2 ** i), n_conditions) for i in range(depth)][::-1])
        self.outc = nn.Conv2d(n_base_channels, n_out, kernel_size=1)

    def forward(self, x, c):
        
        xs = []
        x = self.inc(x, c)
        xs.append(x)
        for down in self.downs:
            x = down(x, c)
            xs.append(x)
        xs = xs[-2::-1]
        for i, up in enumerate(self.ups):
            x = up(x, xs[i], c)
        x = self.outc(x)

        return x

class DoubleConv_CBN(nn.Module):

    def __init__(self, in_channels, out_channels, n_conditions):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = MultiConditionalBatchNorm2d(out_channels, n_conditions)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = MultiConditionalBatchNorm2d(out_channels, n_conditions)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, c):

        x = self.relu(self.bn1(self.conv1(x), c))
        x = self.relu(self.bn2(self.conv2(x), c))

        return x

class Down_CBN(nn.Module):

    def __init__(self, in_channels, out_channels, n_conditions):

        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = DoubleConv_CBN(in_channels, out_channels, n_conditions)

    def forward(self, x, c):

        return self.conv(self.pool(x), c)

class Up_CBN(nn.Module):

    def __init__(self, in_channels, out_channels, n_conditions):

        super().__init__()
        self.linear = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
        self.double_conv = DoubleConv_CBN(in_channels, out_channels, n_conditions)

    def forward(self, x, x_res, c):

        x = self.linear(x)
        x_up = F.interpolate(x, scale_factor=2)
        x = torch.cat([x_res, x_up], dim=1)
        x = self.double_conv(x, c)

        return x

class MultiConditionalBatchNorm2d(nn.Module):
  
    def __init__(self, n_channels, n_conditions):

        super().__init__()
        self.n = n_channels
        self.bn = nn.BatchNorm2d(n_channels, affine=False)
        self.linear_beta = nn.Conv2d(n_conditions, n_channels, kernel_size=1)
        self.linear_gamma = nn.Conv2d(n_conditions, n_channels, kernel_size=1)

    def forward(self, x, c):

        x = self.bn(x)
        c = F.avg_pool2d(c, kernel_size=c.shape[2] // x.shape[2])
        gamma = self.linear_gamma(c)
        beta = self.linear_beta(c)
        x = gamma * x + beta
        
        return x
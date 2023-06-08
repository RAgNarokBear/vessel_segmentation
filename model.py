from torch import nn
import torch
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, drop_rate=0):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, padding='same', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class VesselSeg(nn.Module):
    def __init__(self):
        super(VesselSeg, self).__init__()

        self.c1 = ConvBlock(3, 64)
        self.d1 = DownSample(64)
        self.c2 = ConvBlock(64, 128)
        self.d2 = DownSample(128)
        self.c3 = ConvBlock(128, 256)
        self.d3 = DownSample(256)
        self.c4 = ConvBlock(256, 512)
        # self.d4 = DownSample(512)
        # self.c5 = ConvBlock(512, 1024)
        # self.u1 = UpSample(1024)
        # self.c6 = ConvBlock(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = ConvBlock(512, 256)
        self.u3 = UpSample(256)
        self.c8 = ConvBlock(256, 128)
        self.u4 = UpSample(128)
        self.c9 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        # R5 = self.c5(self.d4(R4))
        #
        # o1 = self.c6(self.u1(R5, R4))
        o2 = self.c7(self.u2(R4, R3))
        o3 = self.c8(self.u3(o2, R2))
        o4 = self.c9(self.u4(o3, R1))

        return self.Th(self.out(o4))

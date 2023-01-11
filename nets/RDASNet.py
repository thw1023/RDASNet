import torch
from torch import nn

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Dense(nn.Module):
    '''
    Dense：Conv+LeakyReLU
    '''

    def __init__(self, in_channels, out_channels):
        super(Dense, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.prelu = nn.PReLU(num_parameters=out_channels)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # ret = torch.cat([x, self.prelu(self.conv(x))], 1)
        ret = torch.cat([x, self.leakyrelu(self.conv(x))], 1)
        return ret

class Attention(nn.Module):
    class ChannelAttention(nn.Module):
        def __init__(self, channel, ratio=16):
            super(Attention.ChannelAttention, self).__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel // ratio, 1, bias=False, dilation=2),
                nn.LeakyReLU(),
                nn.Conv2d(channel // ratio, channel, 1, bias=False, dilation=2)
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # print(self.mlp)
            avgout = self.mlp(self.avgpool(x))
            return self.sigmoid(avgout)

    class SpatialAttention(nn.Module):
        def __init__(self):
            super(Attention.SpatialAttention, self).__init__()
            self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=2, dilation=2)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avgout = torch.mean(x, dim=1, keepdim=True)
            out = self.sigmoid(self.conv2d(avgout))
            return out

    def __init__(self, channel):
        super(Attention, self).__init__()
        self.channel_attention = Attention.ChannelAttention(channel)
        self.spatial_attention = Attention.SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

class RDABlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):  # 64,64,8
        super(RDABlock, self).__init__()
        self.layers = nn.Sequential(
            *[Dense(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)]
        )
        self.lff = nn.Conv2d(in_channels * 2 + growth_rate * num_layers, growth_rate, kernel_size=1)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.attention = Attention(in_channels)

    def forward(self, x):
        out = self.attention(x)
        dense = self.layers(x)
        return x + self.lff(torch.cat([dense, out], 1))

class RDASNet(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDASNet, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        # self.prelu = nn.PReLU(num_parameters=64)

        self.rdabs = nn.ModuleList([RDABlock(self.G0, self.G, self.C)])  # RDABlocks
        for _ in range(self.D - 1):
            rdab = RDABlock(self.G, self.G, self.C)
            self.rdabs.append(rdab)

        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
    def forward(self, x):
        sfe1 = self.leakyrelu(self.sfe1(x))
        sfe2 = self.leakyrelu(self.sfe2(sfe1))

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdabs[i](x)
            local_features.append(x)

        x = self.leakyrelu(self.gff(torch.cat(local_features, 1))) + sfe1
        x = self.output(x)
        return x

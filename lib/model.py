import torch.nn as nn
from torchvision.models import mobilenet_v2


class ResidualBlock(nn.Module):
    def __init__(self, inp, oup):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp, oup, 1, 1, 0),
            nn.BatchNorm2d(oup),
        )
            # self.conv = nn.Sequential(
            #     # pw
            #     nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(hidden_dim),
            #     nn.ReLU6(inplace=True),
            #     # dw
            #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            #     nn.BatchNorm2d(hidden_dim),
            #     nn.ReLU6(inplace=True),
            #     # pw-linear
            #     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(oup),
            # )

    def forward(self, x):
        return self.conv(x)



class SegModel(nn.Module):
    def __init__(self, ncl=2):
        super(SegModel, self).__init__()

        self.backbone = mobilenet_v2(pretrained=True).features
        
        self.decode1 = nn.Sequential(nn.Upsample(scale_factor=2), ResidualBlock(1280, 96))
        self.decode2 = nn.Sequential(ResidualBlock(96, 32), nn.Upsample(scale_factor=2))
        self.decode3 = nn.Sequential(ResidualBlock(32, 24), nn.Upsample(scale_factor=2))
        self.decode4 = nn.Sequential(ResidualBlock(24, 16), nn.Upsample(scale_factor=2))
        
        self.decode5 = nn.Sequential(ResidualBlock(16, 16), nn.Upsample(scale_factor=2))
        self.outp =  nn.Conv2d(16, ncl, 3, padding=1)


    def forward(self, x):
        for i in range(0, 2):
            x = self.backbone[i](x)
            # print(i, x.shape)
        encode1 = x

        for i in range(2, 4):
            x = self.backbone[i](x)
            # print(i, x.shape)
        encode2 = x

        for i in range(4, 7):
            x = self.backbone[i](x)
            # print(i, x.shape)
        encode3 = x

        for i in range(7, 14):
            x = self.backbone[i](x)
            # print(i, x.shape)
        encode4 = x

        for i in range(14, 19):
            x = self.backbone[i](x)
            # print(i, x.shape)

        x = self.decode1(x) + encode4
        x = self.decode2(x) + encode3
        x = self.decode3(x) + encode2
        x = self.decode4(x) + encode1
        x = self.decode5(x)
        x = self.outp(x)
        # print(x.shape)
        return x
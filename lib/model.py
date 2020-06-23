import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class SegModel(nn.Module):
    def __init__(self, ncl=2):
        super(SegModel, self).__init__()

        self.backbone = mobilenet_v2(pretrained=True).features
        
        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.dconv5 = nn.ConvTranspose2d(16, 8, 4, padding=1, stride=2)

        self.conv_last = nn.Conv2d(8, 4, 1)
        self.outp =  nn.Conv2d(4, ncl-1, 1)


    def forward(self, x):
        for i in range(0, 2):
            x = self.backbone[i](x)
        encode1 = x

        for i in range(2, 4):
            x = self.backbone[i](x)
        encode2 = x

        for i in range(4, 7):
            x = self.backbone[i](x)
        encode3 = x

        for i in range(7, 14):
            x = self.backbone[i](x)
        encode4 = x

        for i in range(14, 19):
            x = self.backbone[i](x)

        x = torch.cat([
            encode4,
            self.dconv1(x)
        ], dim=1)

        x = self.invres1(x)

        x = torch.cat([
            encode3,
            self.dconv2(x)
        ], dim=1)
        x = self.invres2(x)

        x = torch.cat([
            encode2,
            self.dconv3(x)
        ], dim=1)
        x = self.invres3(x)

        x = torch.cat([
            encode1,
            self.dconv4(x)
        ], dim=1)
        x = self.invres4(x)

        x = self.dconv5(x)
        x = self.conv_last(x)
        x = self.outp(x)
        x = torch.sigmoid(x)
        
        return x
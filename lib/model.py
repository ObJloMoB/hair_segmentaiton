import torch.nn as nn
from torchvision.models import mobilenet_v2


class ResidualBlock(nn.Module):
    def __init__(self, inp, oup):
        super(ResidualBlock, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp),
            nn.BatchNorm2d(hidden_dim),
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
        return x + self.conv(x)



class SegModel(nn.Module):
    def __init__(self, num_cl=2):
        super(SegModel, self).__init__()

        self.backbone = mobilenet_v2(pretrained=True).features
        
        self.decode1 = ResidualBlock(1280, 96)
        self.decode2 = ResidualBlock(96, 64)
        self.decode3 = ResidualBlock(64, 32)
        self.decode4 = ResidualBlock(32, 32)
        self.decode5 = ResidualBlock(32, 16)
        self.outp =  nn.Conv2d(16, num_cl, 3, padding=1)


    def forward(self, x):
        for i in range(0, 1):
            x = self.backbone[i](x)
        encode1 = x

        for i in range(1, 3):
            x = self.backbone[i](x)
        encode2 = x

        for i in range(3, 5):
            x = self.backbone[i](x)
        encode3 = x

        for i in range(5, 8):
            x = self.backbone[i](x)
        encode4 = x

        for i in range(8, 15):
            x = self.backbone[i](x)
        encode5 = x

        for i in range(15, 19):
            x = self.backbone[i](x)






        return x
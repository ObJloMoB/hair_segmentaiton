import torch

from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

import cv2
import numpy as np


class ImageGradientLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(ImageGradientLoss, self).__init__(size_average, reduce, reduction)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, gray_image):
        gradient_tensor_x = torch.Tensor([[0.0, 0.0, 0.0],
                                          [1.0, 0.0, -1.0],
                                          [0.0, 0.0, 0.0]]).to(self.device).view((1, 1, 3, 3))

        gradient_tensor_y = torch.Tensor([[0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0],
                                          [0.0, -1.0, 0.0]]).to(self.device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_image, gradient_tensor_x)
        G_x = F.conv2d(pred, gradient_tensor_x)

        I_y = F.conv2d(gray_image, gradient_tensor_y)
        G_y = F.conv2d(pred, gradient_tensor_y)

        G_x[G_x == torch.clamp(G_x, -1e-2, 1e-2)] = 0
        G_y[G_y == torch.clamp(G_y, -1e-2, 1e-2)] = 0

        gx_pow = G_x**2
        gy_pow = G_y**2

        dist = gx_pow + gy_pow
        G = torch.sqrt(dist)

        gradient = 1 - torch.pow(torch.mul(I_x, G_x) + torch.mul(I_y, G_y), 2)
        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / torch.sum(G)

        print('[FradLossF]', image_gradient_loss)

        return image_gradient_loss

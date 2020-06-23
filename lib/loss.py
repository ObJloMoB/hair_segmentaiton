import torch

from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F


def FradLossF(device):

    def loss(pred, gray_image):
        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(device).view((1, 1, 3, 3))

        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_image, gradient_tensor_x)
        G_x = F.conv2d(pred, gradient_tensor_x)

        I_y = F.conv2d(gray_image, gradient_tensor_y)
        G_y = F.conv2d(pred, gradient_tensor_y)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        G_x_norm = G_x / G
        G_y_norm = G_y / G
        print('G', G.unique())
        print('Gx', G_x.unique())
        print('Gy', G_y.unique())

        
        
        gradient = 1 - torch.pow(I_x * G_x + I_y * G_y, 2)

        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / torch.sum(G)

        print('[FradLossF]', image_gradient_loss)
        image_gradient_loss = image_gradient_loss if image_gradient_loss > 0 else 0

        return image_gradient_loss
    return loss

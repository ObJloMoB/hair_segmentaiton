import torch

from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F

import cv2
import numpy as np


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
        


        # I_x = (I_x / 8) + 0.5
        # I_y = (I_y / 8) + 0.5

        G_x = (G_x / 8) + 0.5
        G_y = (G_y / 8) + 0.5

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        

        # gr = gray_image.permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # gri = (gr*255).astype(np.uint8)
        # cv2.imshow('gri', gri)

        # print('PRPRPR', pred.unique())
        # pr = pred.detach().permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # pri = (pr*255).astype(np.uint8)
        # cv2.imshow('pri', pri)
        # gx = G_x.detach().permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # gxi = (gx*255).astype(np.uint8)
        # cv2.imshow('gxi', gxi)

        # gy = G_y.detach().permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # gyi = (gy*255).astype(np.uint8)
        # cv2.imshow('gyi', gyi)
        # cv2.waitKey(1)

        # print('G', G.unique())
        # print('Gx', G_x.unique())
        # print('Gy', G_y.unique())

        gradient = 1 - torch.pow(I_x * G_x + I_y * G_y, 2)
        G_sum = torch.sum(G)
        print('GGG', G_sum)
        if G_sum < 1e-3:
            return 0
        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / G_sum

        print('[FradLossF]', image_gradient_loss)
        image_gradient_loss = image_gradient_loss if image_gradient_loss > 0 else 0
        return image_gradient_loss
    return loss


class ImageGradientLoss(_WeightedLoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(ImageGradientLoss, self).__init__(size_average, reduce, reduction)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, gray_image):
        # gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
        #                                   [2.0, 0.0, -2.0],
        #                                   [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        # gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
        #                                   [0.0, 0.0, 0.0],
        #                                   [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))


        gradient_tensor_y = torch.Tensor([[1.0],
                                          [0.0],
                                          [-1.0]]).to(self.device).view((1, 1, 3, 1))

        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 1, 3))




        

        I_x = F.conv2d(gray_image, gradient_tensor_x)[:, :, 1:223, :]
        G_x = F.conv2d(pred, gradient_tensor_x)[:, :, 1:223, :]

        I_y = F.conv2d(gray_image, gradient_tensor_y)[:, :, :, 1:223]
        G_y = F.conv2d(pred, gradient_tensor_y)[:, :, :, 1:223]

        # print('Gx', G_x.shape)
        # print('Gy', G_y.shape)

        # print('Gx', G_x)
        # print('Gy', G_y)

        # print('Ix', I_x)
        # print('Iy', I_y)

        # print('Ix uni', I_x.unique())
        
        # I_x = (I_x / 2) + 0.5
        # I_y = (I_y / 2) + 0.5

        # G_x = (G_x / 2) + 0.5
        # G_y = (G_y / 2) + 0.5

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        # gr = gray_image.permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # gri = (gr*255).astype(np.uint8)
        # cv2.imshow('gri', gri)

        # pr = pred.detach().permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # pri = (pr*255).astype(np.uint8)
        # cv2.imshow('pri', pri)
        # gx = G_x.detach().permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # gxi = (gx*255).astype(np.uint8)
        # cv2.imshow('gxi', gxi)

        # gy = G_y.detach().permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # gyi = (gy*255).astype(np.uint8)
        # cv2.imshow('gyi', gyi)

        # ix = I_x.detach().permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # ixi = (ix*255).astype(np.uint8)
        # cv2.imshow('ixi', ixi)

        # iy = I_y.detach().permute((0, 2, 3, 1)).cpu().numpy()[0] 
        # iyi = (iy*255).astype(np.uint8)
        # cv2.imshow('iyi', iyi)
        # cv2.waitKey(1)

        # print('G', G.unique())
        # print('Gx', G_x.unique())
        # print('Gy', G_y.unique())

        gradient = 1 - torch.pow(I_x * G_x + I_y * G_y, 2)
        # print('Grad', gradient)
        G_sum = torch.sum(G)
        # print('GGG', G_sum)
        if G_sum < 1e-3:
            return 0
        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / G_sum

        print('[FradLossF]', image_gradient_loss)
        image_gradient_loss = image_gradient_loss if image_gradient_loss > 0 else 0
        return image_gradient_loss

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import numpy as np
import cv2
import argparse
import os

from time import time, sleep

from lib.model import SegModel
from lib.dataset import MaskDataset
from lib.data_utils import get_all_data, split_train_val
from lib.loss import ImageGradientLoss
from utils.metric import iou

from torch import autograd


def train(opts):
    # Select device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Define model
    model = SegModel().to(device)
    
    # Define dataloaders
    train_transform = [transforms.Compose([#transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                                              transforms.RandomResizedCrop(opts.size, scale=(0.8, 1.2)),
                                              #transforms.RandomAffine(10.),
                                              #transforms.RandomRotation(13.),
                                              transforms.RandomHorizontalFlip()]),
                            transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std= [0.229, 0.224, 0.225])])]

    train_mask_transform = transforms.Compose([transforms.RandomResizedCrop(opts.size, scale=(0.8, 1.2)),
                                               #transforms.RandomAffine(10.),
                                               #transforms.RandomRotation(13.),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()])

    val_transform = [transforms.Compose([transforms.Resize(opts.size),
                                         transforms.CenterCrop(opts.size)]),
                     transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std= [0.229, 0.224, 0.225])])]
    val_mask_transform = transforms.Compose([transforms.Resize(opts.size),
                                             transforms.CenterCrop(opts.size),
                                             transforms.ToTensor()])
    data = get_all_data(opts.data)
    train_data, val_data = split_train_val(*data)

    train_loader = DataLoader(MaskDataset(*train_data, train_transform, train_mask_transform),
                              batch_size=opts.bs,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=1)

    val_loader = DataLoader(MaskDataset(*val_data, val_transform, val_mask_transform),
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=1)
    # Define loss
    loss_criter = nn.BCELoss().to(device)
    edge_criter = ImageGradientLoss().to(device)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=opts.lr)
    scheduler = StepLR(optimizer, step_size=int(opts.epoch/3), gamma=0.1)

    # Add visuzalizer
    if opts.vis:
        raise NotImplementedError('Boi o boi')

    # Training loop
    for epoch in range(opts.epoch):
        # Train cycle
        running_loss = 0.0
        model.train()
        # print(model)
       
        for batch_num, (inputs, gray, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            gray = gray.to(device)
            # with autograd.detect_anomaly():
            outputs = model(inputs)

            # outputs_f = outputs.permute(0, 2, 3, 1).contiguous().view(-1, opts.ncl)
            # labels_f = labels.view(-1).long()

            outputs_f = outputs.view(-1)
            labels_f = labels.view(-1)

            loss = loss_criter(outputs_f, labels_f)
            # print('BCE loss', loss.shape)
            
            edge_loss = edge_criter(outputs, labels)
            total_loss = loss +  opts.edge_w * edge_loss
            # total_loss = edge_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print('loss val ab',  total_loss)
            # print('Outp layer grad', model.outp.weight.grad)

            running_loss += loss.item() * inputs.size(0)

            print(f'epoch num {epoch:02d} batch num {batch_num:04d} train loss {running_loss/((batch_num+1)*inputs.size(0)):02.04f}', end='\n')

        epoch_loss = running_loss / len(train_loader.dataset)

        # Val cycle
        running_loss = 0.0
        runing_iou = 0.0
        model.eval()
        for inputs, gray, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                outputs_f = outputs.view(-1).float()
                labels_f = labels.view(-1).float()
                loss = 0
                # loss = loss_criter(outputs_f, labels_f)
                val_iou = iou(outputs, labels)
                
            runing_iou = val_iou.item() * inputs.size(0)
            # running_loss += loss.item() * inputs.size(0)

        epoch_val_iou = runing_iou / len(val_loader.dataset)
        epoch_val_loss = running_loss / len(val_loader.dataset)
        print(f'\n\nepoch num {epoch:02d} train loss {epoch_loss:02.04f} val loss {epoch_val_loss:02.04f} val iou {runing_iou:02.04f}')

        scheduler.step()

        if (epoch + 1) % opts.save_every == 0:
            torch.save(model.state_dict(), os.path.join(opts.output, f'checkpoint_size{opts.size}_e{epoch}of{opts.epoch}_lr{opts.lr:.01E}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='DATA',
                        type=str, required=True)
    parser.add_argument('--val_split', help='validation data proportion',
                        default=0.2, type=float)
    parser.add_argument('--lr', help='LR',
                        default=1e-4, type=float)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)
    parser.add_argument('--epoch', help='Train duration',
                        default=30, type=int)
    parser.add_argument('--bs', help='BS',
                        default=64, type=int)
    parser.add_argument('--edge_w', help='EDGE W',
                        default=0.5, type=float)
    parser.add_argument('--save_every', help='Save every N epoch',
                        default=10, type=int)
    parser.add_argument('--output', help='Save every N epoch',
                        default='.', type=str)
    parser.add_argument('--vis', help='Visdom visuzalizer',
                        action="store_true")

    args = parser.parse_args()
    train(args)

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
from utils.metric import iou


def train(opts):
    # Select device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Define model
    model = SegModel(opts.ncl).to(device)
    
    # Define dataloaders
    train_transform = transforms.Compose([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                                          transforms.RandomResizedCrop(opts.size, scale=(0.8, 1.2)),
                                          transforms.RandomAffine(10.),
                                          transforms.RandomRotation(13.),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std= [0.229, 0.224, 0.225])])

    train_mask_transform = transforms.Compose([transforms.RandomResizedCrop(opts.size, scale=(0.8, 1.2)),
                                               transforms.RandomAffine(10.),
                                               transforms.RandomRotation(13.),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()])

    val_transform = transforms.Compose([transforms.Resize((opts.size, opts.size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std= [0.229, 0.224, 0.225])])
    val_mask_transform = transforms.Compose([transforms.Resize((opts.size, opts.size)),
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
    def loss_func(input, target):
        smooth = 1.0
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    loss_criter = nn.CrossEntropyLoss().to(device)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=opts.lr)
    scheduler = StepLR(optimizer, step_size=int(opts.epoch/2), gamma=0.1)

    # Add visuzalizer
    if opts.vis:
        raise NotImplementedError('Boi o boi')

    # Training loop
    for epoch in range(opts.epoch):
        # Train cycle
        running_loss = 0.0
        model.train()
       
        for batch_num, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            outputs_f = outputs.permute(0, 2, 3, 1).contiguous().view(-1, opts.ncl)
            labels_f = labels.view(-1).long()
            loss = loss_criter(outputs_f, labels_f)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            print(f'epoch num {epoch:02d} batch num {batch_num:04d} train loss {running_loss/((batch_num+1)*inputs.size(0)):02.04f}', end='\r')

        epoch_loss = running_loss / len(train_loader.dataset)

        # Val cycle
        running_loss = 0.0
        runing_iou = 0.0
        model.eval()
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                outputs_f = outputs.permute(0, 2, 3, 1).contiguous().view(-1, opts.ncl)
                labels_f = labels.view(-1).long()
                loss = loss_criter(outputs_f, labels_f)
                val_iou = iou(outputs, labels)
                
            runing_iou = val_iou.item() * inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

        epoch_val_iou = runing_iou / len(val_loader.dataset)
        epoch_val_loss = running_loss / len(val_loader.dataset)
        print(f'\n\nepoch num {epoch:02d} train loss {epoch_loss:02.04f} val loss {epoch_val_loss:02.04f} val iou {runing_iou:02.04f}')

        scheduler.step()

        if (epoch + 1) % opts.save_every == 0:
            torch.save(model, os.path.join(opts.output, f'checkpoint_e{epoch}of{opts.epoch}_lr{opts.lr:.01E}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data',
                        type=str, required=True)
    parser.add_argument('--val_split', help='validation data proportion',
                        default=0.2, type=float)
    parser.add_argument('--ncl', help='number os classes',
                        default=2, type=int)
    parser.add_argument('--lr', help='LR',
                        default=1e-4, type=float)
    parser.add_argument('--size', help='Input image size',
                        default=256, type=int)
    parser.add_argument('--epoch', help='Train duration',
                        default=30, type=int)
    parser.add_argument('--bs', help='BS',
                        default=64, type=int)
    parser.add_argument('--accum', help='Accumulated batches',
                        default=3, type=int)
    parser.add_argument('--save_every', help='Save every N epoch',
                        default=5, type=int)
    parser.add_argument('--output', help='Save every N epoch',
                        default='.', type=str)
    parser.add_argument('--vis', help='Visdom visuzalizer',
                        action="store_true")

    args = parser.parse_args()
    train(args)

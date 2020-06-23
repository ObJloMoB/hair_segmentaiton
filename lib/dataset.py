import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class MaskDataset(Dataset):
    def __init__(self, img_files, mask_files, transform, mask_transform):
        self.img_files = img_files
        self.mask_files = mask_files
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_files[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[:, :, 0]

        seed = random.randint(0, 2 ** 32)

        random.seed(seed)
        img = Image.fromarray(img)
        img = self.transform[0](img)

        # Rework this somehow PLS
        gray = F.to_grayscale(img)
        gray  = F.to_tensor(gray)

        tensor = self.transform[1](img)

        random.seed(seed)
        mask = Image.fromarray(mask, 'L')
        mask = self.mask_transform(mask)
        return tensor, gray, mask

    def __len__(self):
        return len(self.img_files)

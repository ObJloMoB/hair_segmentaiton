import os
import numpy as np


def get_all_data(data_path):
    mask_dir = os.path.join(data_path, 'masks')
    mask_names = os.listdir(mask_dir)
    img_paths = [os.path.join(data_path, 'images', x.replace('.ppm', '.jpg')) for x in mask_names]
    mask_paths = [os.path.join(data_path, 'masks', x) for x in mask_names]

    return img_paths, mask_paths


def split_train_val(img_paths, mask_paths, val_split=0.2):
    for data in (img_paths, mask_paths):
        np.random.seed(322)
        np.random.shuffle(data)

    split_idx = int(len(img_paths)*val_split)

    val_data = img_paths[:split_idx], mask_paths[:split_idx]
    train_data = img_paths[split_idx:], mask_paths[split_idx:]

    return train_data, val_data




import os
import torch
from torch import nn
import config as cfg
import classes
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


# Setting up dataset & dataloader
train_dataset = classes.ImageDataset(
    data_folder=cfg.data_folder,
    input_dir=cfg.train_input_dir,
    target_dir=cfg.train_input_dir,
    transform=cfg.filter_transform)

test_dataset = classes.ImageDataset(
    data_folder=cfg.data_folder,
    input_dir=cfg.test_input_dir,
    target_dir=cfg.test_input_dir,
    transform=cfg.filter_transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=cfg.train_shuffle)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    shuffle=cfg.test_shuffle)

filter = classes.Filter()
fig = plt.figure(figsize=(18, 9))
for ite, datapoint in enumerate(tqdm(train_loader, desc='Filter')):
    input_image = datapoint['input_image']
    output = filter(input_image)
    im_save = output.numpy()
    im = Image.fromarray(im_save[0, 0, :, :])
    print(np.max(im))
    print(np.min(im))
    im.save(
        os.path.join(cfg.data_folder, cfg.train_target_dir, str(ite) + '.jpg'))

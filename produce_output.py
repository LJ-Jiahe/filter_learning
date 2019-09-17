import os
import torch
from torch import nn
import config as cfg
import classes
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torchvision


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
    torchvision.utils.save_image(
        input_image[0, 0, :, :], os.path.join(cfg.data_folder, cfg.train_gs_dir, str(ite) + '.jpg'))
    torchvision.utils.save_image(
        output[0, 0, :, :], os.path.join(cfg.data_folder, cfg.train_gs_filtered_dir, str(ite) + '.jpg'))

for ite, datapoint in enumerate(tqdm(test_loader, desc='Filter')):
    input_image = datapoint['input_image']
    output = filter(input_image)
    torchvision.utils.save_image(
        input_image[0, 0, :, :], os.path.join(cfg.data_folder, cfg.test_gs_dir, str(ite) + '.jpg'))
    torchvision.utils.save_image(
        output[0, 0, :, :], os.path.join(cfg.data_folder, cfg.test_gs_filtered_dir, str(ite) + '.jpg'))


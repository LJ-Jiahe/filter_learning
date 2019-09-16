import os
from torch import nn
import config as cfg
import classes
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


# Setting up dataset & dataloader
train_dataset = classes.ImageDataset(
    data_folder=cfg.data_folder,
    input_dir=cfg.train_input_dir,
    transform=cfg.filter_transform)

test_dataset = classes.ImageDataset(
    data_folder=cfg.data_folder,
    input_dir=cfg.test_input_dir,
    transform=cfg.filter_transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=cfg.train_shuffle)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    shuffle=cfg.test_shuffle)

filter = classes.Filter()

for ite, datapoint in enumerate(tqdm(test_loader, desc='Filter')):
    input_image = datapoint['input_image']
    output = filter(input_image)
    plt.imshow(output)

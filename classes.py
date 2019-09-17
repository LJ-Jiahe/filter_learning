import torch
import torch.nn as nn
from torch.utils.data import Dataset
import config as cfg
import os
from PIL import Image
import numpy as np
import scipy
import scipy.ndimage


class One_Filter_Net(nn.Module):

    def __init__(self):
        super(One_Filter_Net, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            bias=cfg.filter_bias)

        nn.init.xavier_uniform_(self.conv.weight)
        self.ac_fun = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        return(out)


class ImageDataset(Dataset):

    def __init__(self, data_folder, input_dir, target_dir, transform=None):
        self.data_folder = data_folder
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        self.input_contents = os.listdir(os.path.join(data_folder, input_dir))
        self.target_contents = os.listdir(os.path.join(data_folder, target_dir))

        self.input_contents.sort(key= lambda x:int(x.split('.')[0]))
        self.target_contents.sort(key= lambda x:int(x.split('.')[0]))

    def __len__ (self):
        return len(self.input_contents)

    def __getitem__(self, idx):
        input_image = Image.open(os.path.join(self.data_folder, 
                                             self.input_dir, 
                                             self.input_contents[idx]))
        target_image = Image.open(os.path.join(self.data_folder, 
                                              self.target_dir, 
                                              self.target_contents[idx]))

        in_tar_images = {'input_image': self.transform(input_image),
                         'target_image': self.transform(target_image)}

        return in_tar_images


class Filter(nn.Module):

    def __init__(self):
        super(Filter, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=cfg.filter_in_channels,
            out_channels=cfg.filter_out_channels,
            kernel_size=cfg.filter_kernel_size,
            stride=cfg.filter_stride,
            padding=cfg.filter_padding,
            bias=cfg.filter_bias)

        self.weights_init()
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        out = self.conv(x)
        return(out)

    def weights_init(self):
        n= np.zeros(self.conv.kernel_size)
        t = tuple(ti // 2 for ti in self.conv.kernel_size)
        n[t] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=3)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
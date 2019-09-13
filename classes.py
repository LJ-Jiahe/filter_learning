import torch
import torch.nn as nn
import config as cfg

class One_Filter_Net(nn.Module):

    def __init__(self):
        super(One_Filter_Net, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=cfg.in_channels
            out_channels=cfg.out_channels
            kernel_size=cfg.kernal_size
            stride=cfg.stride
            padding=cfg.padding)

        nn.init.xavier_uniform_(self.conv.weight)
        self.ac_fun = nn.sigmoid()

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

    def __getitem__(self, idx):
        input_image = io.imread(os.path.join(self.data_folder, 
                                             self.input_dir, 
                                             self.input_contents[idx])
        target_image = io.imread(os.path.join(self.data_folder, 
                                              self.target_dir, 
                                              self.target_contents[idx])

        in_out_images = {'input_image': self.transform(input_image),
                         'target_image': self.transform(target_image)}

        return in_out_images



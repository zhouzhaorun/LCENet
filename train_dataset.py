import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import random

class LowLightDataset(Dataset):
    def __init__(self, low_dir, real_dir, is_train=False, img_size=320, transform=None):
        self.data_info = self.get_img_info(low_dir, real_dir)
        self.transform = transform
        self.img_size = img_size
        self.is_train = is_train

    def __getitem__(self, index):
        low_path, real_path = self.data_info[index]
        low_img = Image.open(low_path).convert('RGB')
        real_img = Image.open(real_path).convert('RGB')

        if self.is_train:
            img_w = low_img.size[0]
            img_h = low_img.size[1]
            idx_x = random.randint(0, img_w - self.img_size)
            idx_y = random.randint(0, img_h - self.img_size)
            patch = (idx_x, idx_y, idx_x + self.img_size, idx_y + self.img_size)
            low_img = low_img.crop(patch)
            real_img = real_img.crop(patch)

        if self.transform is not None:
            low_img = torch.FloatTensor(self.transform(low_img))
            real_img = torch.FloatTensor(self.transform(real_img))

        return low_img, real_img

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(low_dir, real_dir):
        data_info = list()
        data_list = os.listdir(low_dir)
        for idx, img_name in enumerate(data_list):
            low_path = os.path.join(low_dir, img_name)
            real_path = os.path.join(real_dir, img_name)
            data_info.append((low_path, real_path))
        return data_info





















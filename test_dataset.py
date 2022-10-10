import os
from torch.utils.data import Dataset
from PIL import Image


class LowLightDataset(Dataset):
    def __init__(self, low_dir, transform=None):
        self.data_info = self.get_img_info(low_dir)
        self.transform = transform
    def __getitem__(self, index):
        low_path = self.data_info[index]
        low_img = Image.open(low_path).convert('RGB')
        # low_img = low_img.resize((960, 540), Image.BILINEAR)
        if self.transform is not None:
            low_img = self.transform(low_img)
        return low_img, low_path.split('/')[-1]
    def __len__(self):
        return len(self.data_info)
    @staticmethod
    def get_img_info(low_dir):
        data_info = list()
        data_list = os.listdir(low_dir)
        for idx, img_name in enumerate(data_list):
            low_path = os.path.join(low_dir, img_name)
            data_info.append(low_path)
        return data_info










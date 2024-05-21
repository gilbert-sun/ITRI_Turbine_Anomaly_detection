import os
import random
import torch
from PIL import Image
from utils import getFolder, getImage
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    ''' 把 wav 檔轉成 spectogram 後的 dataset (在 spectrogram 上去做 autoencoder)
        SpectrogramDataset:
        - root: path to dataset
        - train: True if training, False if testing
    '''
    def __init__(self, root, train=True, transform=None, args=None):
        self.root = root
        self.train=train
        self.transform = transform
        self.normal_files = getFolder(os.path.join(self.root, "normal"))
        self.abnormal_files = getFolder(os.path.join(self.root, "abnormal" + "/" + args.abnormal_class))

        abnormal_size = len(self.abnormal_files)
        random.shuffle(self.normal_files)
        random.shuffle(self.abnormal_files)

        if args.dataset_name == "ITRI_Small":
            abnormal_size = 5

        self.train_files = self.normal_files[abnormal_size:]
        self.test_files = self.normal_files[:abnormal_size] + self.abnormal_files

        """ 
        Training: 每一張圖片為一筆 data
        Testing: 每一個資料夾(wav檔)為一筆 data
        """
        self.data = []
        if train:
            for file in self.train_files:
                self.data.extend(getImage(file))
            self.label = [0] * len(self.data)
            print("[Trainset] Normal data: {} (file)".format(len(self.data)))
        else:
            for file in self.test_files:
                self.data.append(getImage(file))
            self.label = [0] * len(self.normal_files[:abnormal_size]) + [1] * len(self.abnormal_files)
            print("[Testset]  Normal data: {}, Abnormal data: {} (folder)".format(len(self.normal_files[:abnormal_size]), len(self.abnormal_files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """ 
        Training data: 
            return image, label
        Testing data: 
            return file, label
        """
        if self.train:
            image = Image.open(self.data[index]).convert('RGB')
            if self.transform:
                image = self.transform(image)
        else:
            image = []
            for img in self.data[index]:
                img = Image.open(img).convert('RGB')
                if self.transform:
                    image.append(self.transform(img))
            image = torch.stack(image, dim=0)
        return image, self.label[index]
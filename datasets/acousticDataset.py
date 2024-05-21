import os
import random
import torch
from PIL import Image
from utils import getFolder, getImage
from torch.utils.data import Dataset

class AcousticDataset(Dataset):
    ''' 用 acoustic images 的 dataset 
        AcousticDataset:
        - root: path to dataset
        - train: True if training, False if testing
    '''
    def __init__(self, root, train=True, transform=None, args=None):
        self.root = root
        self.use_spatial = args.use_spatial
        self.transform = transform
        self.normal = getImage(os.path.join(self.root, "normal"))
        self.abnormal = getImage(os.path.join(self.root, "abnormal" + "/" + args.abnormal_class))

        if self.use_spatial:
            self.normal = self.stack_frames(self.normal, window_size=args.window_size, stride=1)
            self.abnormal = self.stack_frames(self.abnormal, window_size=args.window_size, stride=1)

        random.shuffle(self.normal)
        random.shuffle(self.abnormal)
        train_size = int(len(self.normal) * 0.8)
        test_size = len(self.normal) - train_size

        if train:
            self.data = self.normal[:train_size]
            self.label = [0] * len(self.data)
            print("[Trainset] Normal data: {}".format(len(self.data)))

        else:
            self.data = self.normal[train_size:] + self.abnormal[:test_size]
            self.label = [0] * len(self.normal[train_size:]) + [1] * len(self.abnormal[:test_size])
            print("[Testset] Normal data: {}, Abnormal data: {}".format(len(self.normal[train_size:]), len(self.abnormal[:test_size])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.use_spatial:
            frames = []
            for i in range(len(self.data[index])):
                image = Image.open(self.data[index][i]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
            frames = torch.stack(frames, dim=0)
        else:
            image = Image.open(self.data[index]).convert('RGB')
            if self.transform:
                frames = self.transform(image)
        return frames, self.label[index]
    

    def stack_frames(self, data, window_size=1, stride=1):
        frames = []
        for i in range(0, len(data) - window_size, stride):
            frames.append(data[i:i + window_size])
        return frames
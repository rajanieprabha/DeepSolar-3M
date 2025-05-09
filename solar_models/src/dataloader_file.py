import os
import cv2
import time
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import argparse
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils
from torchvision.transforms import Normalize

random.seed(42)
np.random.seed(42)

class ImageFolder(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_list = [f for f in os.listdir(root_dir) if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']]  # Change the extension as needed
        self.transform = transform
                        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = np.asarray(img)
            if self.transform:
                img = self.transform(img)
            return img, img_path

        except PIL.UnidentifiedImageError as e:
            print(f"Error in file {img_path}: {e}")



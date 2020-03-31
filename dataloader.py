import torch
from torch import nn
import torchvision.transforms as transforms
import torch.utils.data as data

import os
import pickle
import numpy as np
import pandas as pd
import nltk
from PIL import Image

label_dict = {'normal' : 0,'pneumonia' :1,'COVID-19':2}

class CovidDataset(data.Dataset):
    """Custom Covid Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, file_path, transform=None):
        """Set the path for images, labels wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.data = pd.read_csv(file_path, sep = " ", header = None )
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and label)."""
       
        img_path = self.data.iloc[index,1]
        label = self.data.iloc[index,2]
        

        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        
        return image, label_dict.get(label,3)

    def __len__(self):
        return self.data.shape[0]


def get_loader(root, file_path, transform, batch_size, num_workers, shuffle):
    """Returns torch.utils.data.DataLoader for custom covid dataset."""
    covid = CovidDataset(root=root,
                        file_path=file_path,
                        transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=covid,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader

import torch
from torch import nn
import torchvision.transforms as transforms
import torch.utils.data as data

import os
import pickle
import numpy as np
import nltk
from PIL import Image


class CovidDataset(data.Dataset):
    """Custom Covid Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, ids, transform=None):
        """Set the path for images, labels wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = ids
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and label)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def get_loader(root, json, ids, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom covid dataset."""
    coco = CovidDataset(root=root,
                        json=json,
                        ids=ids,
                        vocab=vocab,
                        transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader

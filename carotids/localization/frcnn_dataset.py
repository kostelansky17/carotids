import os

import torch
from torch.utils.data.dataset import Dataset

from carotids.preprocessing import load_img, load_position


class FastCarotidDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transformations):
        self.data_path = imgs_path
        self.labels_path = labels_path

        self.data_files = sorted(os.listdir(imgs_path))
        self.labels_files = sorted(os.listdir(labels_path))

        self.transformations = transformations

    def __getitem__(self, index):
        print(self.data_files[index])
        img = load_img(self.data_path, self.data_files[index])
        label = load_position(self.labels_path, self.labels_files[index])

        boxes = [label]
        boxes = torch.as_tensor([label], dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        label = {
            "boxes": boxes,
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
        }

        return self.transformations(img), label

    def __len__(self):
        return len(self.data_files)


class FastCarotidDatasetEval(Dataset):
    def __init__(self, imgs_path, transformations):
        self.data_path = imgs_path
        self.data_files = sorted(os.listdir(imgs_path))
        self.transformations = transformations
        
    def __getitem__(self, index):
        img = load_img(self.data_path, self.data_files[index])

        return self.transformations(img), self.data_files[index]
        
    def __len__(self):
        return len(self.data_files)

import torch
from torch.utils.data.dataset import Dataset

from carotids.preprocessing import load_position_labels, load_imgs_dir


class ResnetCartroidDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transformations):
        self.data = load_imgs_dir(imgs_path)
        self.labels = load_position_labels(labels_path)
        self.transformations = transformations

    def __getitem__(self, index):
        label_tensor = torch.from_numpy(self.labels[index])
        return self.transformations(self.data[index]).double(), label_tensor.double()[0]

    def __len__(self):
        return len(self.data)

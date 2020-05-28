import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from preprocessing import load_img

IMG_DIRS = {
   0: "/home/martin/Documents/cartroids/data/categorization/diff",
   1: "/home/martin/Documents/cartroids/data/categorization/long",
   2: "/home/martin/Documents/cartroids/data/categorization/trav",
}


class CategorizationDataset(Dataset):
    def __init__(self, img_dirs, transformations):
        self.data_files, self.labels = self._prepare_data(img_dirs)

        self.img_dirs = img_dirs
        self.transformations = transformations
    
    def _prepare_data(self, img_dirs):
        data_files = []
        labels = []
        
        for key in img_dirs:
            img_names = sorted(os.listdir(img_dirs[key]))
            data_files.extend(img_names)
            labels.extend([key] * len(img_names))

        return data_files, labels

    def __getitem__(self, index):
        label =  self.labels[index]

        img = load_img(self.img_dirs[label], self.data_files[index])      
        return self.transformations(img), torch.tensor(label)
        
    def __len__(self):
        return len(self.labels)


def main():
    dataset = CategorizationDataset(IMG_DIRS, TRANSFORMATIONS)


if __name__ == "__main__":
    
    main()

from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler


class ImageDataSet(torch.utils.data.Dataset):
    """
    A class used for representing arbitrary image-datasets. Used for inference.
    """

    def __init__(self, image_paths, output_size, in_channels=3, file_extension="png"):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        self.image_paths = image_paths
        self.images = []
        for path in self.image_paths:
            self.images = self.images + sorted(Path(path).glob('*.' + file_extension))

        self.img_transformer = torchvision.transforms.ToTensor()
        self.output_size = output_size
        self.in_channels = in_channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_file = self.images[item]

        try:
            if self.in_channels == 3:
                img = Image.open(img_file.resolve()).convert('RGB')
            else:
                img = Image.open(img_file.resolve()).convert('L')

            img = img.resize((self.output_size, self.output_size), Image.BILINEAR)
        except FileNotFoundError:
            return None

        return self.img_transformer(img)


class ProcessedDataSet(torch.utils.data.Dataset):

    def __init__(self, main_directory, normalize_coefficients=True, in_channels=3, img_dir="images", mask_dir="masks",
                 output_size=0, file_extension="png"):
        """
        Assumes the main directory contains

        main_directory / images
        main_directory / masks
        """
        self.main_directory = Path(main_directory)
        self.image_directory = self.main_directory / img_dir
        self.mask_directory = self.main_directory / mask_dir
        self.output_size = output_size

        self.images = sorted(self.image_directory.glob('*.' + file_extension))
        self.labels = sorted(self.mask_directory.glob('*.' + file_extension))

        self.img_transformer = torchvision.transforms.ToTensor()
        self.normalize_coefficients = normalize_coefficients
        self.in_channels = in_channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_file = self.images[item]
        lbl_file = self.labels[item]

        try:
            if self.in_channels == 3:
                img = Image.open(img_file.resolve()).convert('RGB')
            else:
                img = Image.open(img_file.resolve()).convert('L')

            if self.output_size:
                img = img.resize((self.output_size, self.output_size))
        except FileNotFoundError:
            return None

        try:
            lbl = Image.open(lbl_file.resolve())
            if self.output_size:
                lbl = lbl.resize((self.output_size, self.output_size))

            lbl = np.array(lbl)
            lbl = torch.tensor(lbl, dtype=torch.float)
        except FileNotFoundError:
            return None

        return self.img_transformer(img), lbl


def test_train_split(dataset, batch_size=1, shuffle=True, test_size=0.2, random_seed=None):
    """
    Auxiliary function for splitting a DataSet-object into Test and Train-subsets.

    :param dataset    : DataSet to split.
    :param batch_size : BatchSize of the returned DataLoaders.
    :param shuffle    : Whether to shuffle the datasets or not.
    :param test_size  : The proportion of test-data to train-data. 0.2 yields 20% test and 80% train.
    :param random_seed: Whether to set a fixed seed.
    :return: DataLoaders for test and train data.
    """
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))

    split = int(np.floor(test_size * dataset_size))

    if shuffle:
        if random_seed:
            np.random.seed(random_seed)
        np.random.shuffle(dataset_indices)
    train_indices, test_indices = dataset_indices[split:], dataset_indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return test_loader, train_loader

# https://github.com/YannDubs/disentangling-vae
import abc
import logging
import os
import subprocess

from typing import Optional, List

import matplotlib.pyplot as plt
import torch

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1

DATASETS_DICT = {"dsprites": "DSprites"}


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_dataloader(dataset_cls_name: str, root=None, shuffle=True, pin_memory=True,
                   batch_size=128, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader
    Parameters
    ----------
    dataset : {"dsprites"}
        Name of the dataset to load
    root : str
        Path to the dataset root. If `None` uses the default one.
    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    dataset_cls = get_dataset(dataset_cls_name)

    if root is None:
        dataset: SceneDataset = dataset_cls(logger=logger)
    else:
        dataset = dataset_cls(root=root, logger=logger)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)


class SceneDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.
    Parameters
    ----------
    root : string
        Root directory of dataset.
    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root: str,
                 transforms_list=Optional[List], logger=logging.getLogger(__name__)):
        if transforms_list is None:
            transforms_list = []

        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if root is None or not os.path.isdir(root):
            self.logger.info(f"Downloading {str(type(self))} ...")
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class DSprites(SceneDataset):
    """DSprites dataset
    Parameters
    ----------
    root : string
        Root directory of dataset.
    Color: white
    Shape: square, ellipse, heart
    Scale: 6 values linearly spaced in [0.5, 1]
    Orientation: 40 values in [0, 2 pi]
    Position X: 32 values in [0, 1]
    Position Y: 32 values in [0, 1]
    """

    urls = {
        "train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)

    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                           0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                           1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                           2.57771705, 2.73882436, 2.89993168, 3.061039,
                                           3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                           4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                           5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799, 6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}

    def __init__(self, root=os.path.join(DIR, '../../data/dsprites/'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        return sample, lat_value


if __name__ == '__main__':
    dataloader = get_dataloader('dsprites')
    x = next(iter(dataloader))
    for vector in x[0][0:15]:
        image = vector.numpy().transpose(1, 2, 0)
        plt.imshow(image, cmap='gray')
        plt.show()

    print("Done")

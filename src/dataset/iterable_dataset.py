import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset


class Dsprites:
    def __init__(self, path=r'C:\Users\Yessense\PycharmProjects\multi-dsprites-vae\src\dataset\data\dsprite_train.npz'):
        dataset_zip = np.load(path)
        self.img = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']
        self.size = self.img.shape[0]

    def get_img(self, n) -> np.ndarray:
        return self.img[n]

    def get_label(self, n) -> np.ndarray:
        return self.lat_values[n]


class MultiDsprites(IterableDataset):
    def __init__(self,
                 path: str = r'C:\Users\Yessense\PycharmProjects\multi-dsprites-vae\src\dataset\data\dsprite_train.npz',
                 size: int = 10 ** 5):
        self.dsprites = Dsprites(path)
        self.size = size

    def generate_sample(self):
        scene = np.zeros((1, 64, 64), dtype=int)
        masks = []
        labels = []

        n_images = random.randint(2, 5)
        print(f'{n_images} images ->')
        for _ in range(n_images):
            for i in range(50):
                n = random.randint(0, self.dsprites.size - 1)
                img = np.expand_dims(self.dsprites.get_img(n), 0)
                if np.any(scene & img):
                    continue
                else:
                    scene += img
                    label = self.dsprites.get_label(n)
                    masks.append(img)
                    labels.append(label)
                    break
        return scene, masks, labels


if __name__ == '__main__':
    md = MultiDsprites()
    for i in range(10):
        sample = md.generate_sample()
        img = sample[0]
        plt.imshow(img.squeeze(0), cmap='gray')
        plt.show()

    print("Done")

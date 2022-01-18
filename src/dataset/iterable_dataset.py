import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader


class Dsprites:
    def __init__(self, path='/home/yessense/PycharmProjects/multi-dsprites-vae/src/dataset/data/dsprite_train.npz'):
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
                 path: str = '/home/yessense/PycharmProjects/multi-dsprites-vae/src/dataset/data/dsprite_train.npz',
                 size: int = 10 ** 5):
        self.dsprites = Dsprites(path)
        self.size = size

    def __iter__(self):
        return self.sample_generator()

    def sample_generator(self):
        for i in range(self.size):
            yield self.generate_sample()

    def generate_sample(self):
        scene = np.zeros((1, 64, 64), dtype=int)
        objs = []
        labels = []
        masks = []

        n_images = random.randint(2, 5)
        for _ in range(n_images):
            for i in range(50):
                n = random.randint(0, self.dsprites.size - 1)
                img = np.expand_dims(self.dsprites.get_img(n), 0)
                if np.any(scene & img):
                    continue
                else:
                    scene += img
                    label = self.dsprites.get_label(n)
                    objs.append(img)
                    labels.append(label)
                    masks.append(1.)
                    break

        for i in range(5 - len(objs)):
            mask = np.zeros((1, 64, 64))
            objs.append(mask)
            label = np.zeros(6)
            labels.append(label)
            masks.append(0.)

        scene = torch.from_numpy(scene).float()
        objs = torch.from_numpy(np.array(objs)).float()
        labels = torch.from_numpy(np.array(labels)).float()
        masks = torch.from_numpy(np.array(masks)).float()

        return scene, objs, labels, masks


if __name__ == '__main__':
    md = MultiDsprites()

    loader = DataLoader(md, batch_size=4)

    x = next(iter(loader))

    for i in range(10):
        sample = md.generate_sample()
        img = sample[0]
        plt.imshow(img.squeeze(0), cmap='gray')
        plt.show()

    print("Done")

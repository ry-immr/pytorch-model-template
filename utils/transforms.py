import torch
from torchvision import transforms
from skimage import transform
import numpy as np


def _rotate_flip(org_img, p):
    img = org_img.copy()
    p = p % 8
    k = p % 4
    flip = p // 4 == 1

    if flip:
        img = np.flip(img, axis=1)
    img = np.rot90(img, k=k, axes=(0, 1))

    return img


class RotateFlip(object):
    def __call__(self, sample):
        p = torch.randint(0, 8, (1,))
        sample["img"] = _rotate_flip(sample["img"], p)
        return sample


def _to_tensor(img):
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img).float()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {k: _to_tensor(sample[k]) for k in sample}

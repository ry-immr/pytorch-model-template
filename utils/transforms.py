import torch
from torchvision import transforms
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
    def __call__(self, img):
        p = torch.randint(0, 8, (1,))
        return _rotate_flip(img, p)

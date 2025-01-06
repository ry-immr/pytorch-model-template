import torch
from torchvision import transforms
from skimage import transform
import numpy as np

def _rotate_flip(img, p):
    if p==0:
        img=img
    elif p==1:
        img=np.flip(img, axis=1)
    elif p==2:
        img=np.rot90(img, k=1, axes=(0,1))
    elif p==3:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=1, axes=(0,1))
    elif p==4:
        img=np.rot90(img, k=2, axes=(0,1))
    elif p==5:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=2, axes=(0,1))
    elif p==6:
        img=np.rot90(img, k=3, axes=(0,1))
    elif p==7:
        img=np.flip(img, axis=1)
        img=np.rot90(img, k=3, axes=(0,1))

    # The flip is realized by giving -1 to the slice of the variable,
    # and torch does not support slices with negative numbers
    img=img.copy()

    return img 


class RotateFlip(object):
    def __call__(self, sample):
        p = torch.randint(0,8,(1,))
        sample['img'] = _rotate_flip(sample['img'], p)
        return sample


def _to_tensor(img):
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img).float()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return {k: _to_tensor(sample[k]) for k in sample}

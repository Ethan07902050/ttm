
import numpy as np
from scipy.ndimage import rotate
from typing import List
import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class MyCollate():
    def pad_zero_tensor(self, batch: List[torch.Tensor]) -> torch.Tensor:
        dims = batch[0].dim()

        max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
        size = (len(batch),) + tuple(max_size)
        canvas = batch[0].new_zeros(size=size)
        for i, b in enumerate(batch):
            sub_tensor = canvas[i]
            for d in range(dims):
                sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
            sub_tensor.add_(b)
        return canvas
    
    def __call__(self, batch):
        imgs_with_mask = self.pad_zero_tensor([item[0] for item in batch])
        audios = self.pad_zero_tensor([item[1] for item in batch])
        
        targets = [item[2].unsqueeze(0) for item in batch]

        targets = torch.cat(targets, dim=0)
        return imgs_with_mask, audios, targets

# np image data augmentation code from link:
"""  https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd  """

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img

def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img
def change_channel_ratio_gauss(img, channel='r', mean=0, sigma=0.03):
    assert channel in 'rgb', "cahenel must be r|g|b"
    img = img.copy()
    ci = 'rgb'.index(channel)
    img[:, :, ci] = gaussian_noise(img[:, :, ci], mean=mean, sigma=sigma)
    return img

def change_channel_ratio(img, channel='r', ratio=0.5):
    assert channel in 'rgb', "Value for channel: r|g|b"
    img = img.copy()
    ci = 'rgb'.index(channel)
    img[:, :, ci] *= ratio
    return img

def distort(img, orientation='horizontal', func=np.sin, x_scale=0.05, y_scale=5):
    assert orientation[:3] in ['hor', 'ver'], "dist_orient should be 'horizontal'|'vertical'"
    assert func in [np.sin, np.cos], "supported functions are np.sin and np.cos"
    assert 0.00 <= x_scale <= 0.1, "x_scale should be in [0.0, 0.1]"
    assert 0 <= y_scale <= min(img.shape[0], img.shape[1]), "y_scale should be less then image size"
    img_dist = img.copy()
    
    def shift(x):
        return int(y_scale * func(np.pi * x * x_scale))
    
    for c in range(3):
        for i in range(img.shape[orientation.startswith('ver')]):
            if orientation.startswith('ver'):
                img_dist[:, i, c] = np.roll(img[:, i, c], shift(i))
            else:
                img_dist[i, :, c] = np.roll(img[i, :, c], shift(i))
            
    return img_dist

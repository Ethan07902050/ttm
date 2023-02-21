import random
import torch
import torchvision.transforms as transforms

def horizontal_flip(frame):
    return transforms.functional.hflip(frame)

def gaussian_blur(frame):
    return transforms.functional.gaussian_blur(frame, 9)

def grayscale(frame):
    return transforms.functional.rgb_to_grayscale(frame, 3)

def crop(frame, size):
    pos = (frame.shape[1] - size) // 2
    return transforms.functional.resized_crop(frame, pos, pos, size, size, 96)

def rotate(frame, angle):
    return transforms.functional.rotate(frame, angle, fill=0.5)


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    return clip.flip((-1))

class RandomHorizontalFlipVideo(object):
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)

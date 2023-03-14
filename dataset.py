import random
import torchvision
import os.path
import torch.utils.data as data
import pandas as pd
from PIL import Image
from util import *
import numpy as np
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class AlignedDataset:
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, phase='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(AlignedDataset, self).__init__()
        self.data_root = data_root
        self.phase = phase

        self.load_size = 286
        self.crop_size = 256
        self.input_nc = 1
        self.output_nc = 3

        self.dir_ABM = os.path.join(self.data_root, self.phase)  # get the image directory
        self.ABM_paths = sorted(make_dataset(self.dir_ABM))  # get image paths

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, M, A_paths, B_paths, M_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            M (tensor) - - tumor area mask
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            M_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        ABM_path = self.ABM_paths[index]
        ABM = Image.open(ABM_path).convert('RGB')
        # split ABM image into A, B and Mask
        w, h = ABM.size
        w2 = int(w / 3)
        A = ABM.crop((0, 0, w2, h))  # left, upper, right, lower
        B = ABM.crop((w2, 0, 2*w2, h))
        M = ABM.crop((2*w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = self.get_params(A.size)
        A_transform = self.get_transform(transform_params, grayscale=(self.input_nc == 1))
        B_transform = self.get_transform(transform_params, grayscale=(self.output_nc == 1))
        M_transform = self.get_transform(transform_params, grayscale=True)

        A = A_transform(A)
        B = B_transform(B)
        M = M_transform(M)

        # get bounding bboxes coordinates
        M_num = M.numpy()  # (-1, 1)
        M_num = np.squeeze(M_num)
        logical_M = np.where(M_num < 0, 0, 1)
        bbox = extract_bboxes(logical_M)  # (y0, x0, y1, x1)

        return {'A': A, 'B': B, 'M': M, 'bbox': bbox}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABM_paths)

    # misc
    def get_params(self, size):
        w, h = size
        new_h = new_w = self.load_size

        x = random.randint(0, np.maximum(0, new_w - self.crop_size))
        y = random.randint(0, np.maximum(0, new_h - self.crop_size))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def get_transform(self, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))

        osize = [self.load_size, self.load_size]
        transform_list.append(transforms.Resize(osize, method))

        if params is None:
            transform_list.append(transforms.RandomCrop(self.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: my_crop(img, params['crop_pos'], self.crop_size)))

        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: my_flip(img, params['flip'])))

        transform_list += [transforms.ToTensor()]
        if convert:
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)


def my_make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    my_print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def my_crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def my_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def my_print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(my_print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        my_print_size_warning.has_printed = True



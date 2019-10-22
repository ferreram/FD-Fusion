# from __future__ import division
import random
import numpy as np
import numbers

import cv2

from PIL import Image, ImageEnhance

from torchvision.transforms import functional as TF

# import torch
# import types

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''


def apply_function_list(x, fun):
    """Apply a function or list over a list of object, or single object."""
    if isinstance(x, list):
        y = []
        if isinstance(fun,list):
            for x_id, x_elem in enumerate(x):
                y.append(fun[x_id](x_elem))
        else:
            for x_id, x_elem in enumerate(x):
                y.append(fun(x_elem))
    else:
        y = fun(x)
    
    return y


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input_, target):
        for t in self.co_transforms:
            input_,target = t(input_,target)
        return input_,target


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __cropCenter__(self, input_):

        if len(input_.shape)==2:
            h1, w1 = input_.shape
        else:
            h1, w1, _ = input_.shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        return input_[y1: y1 + th, x1: x1 + tw]

    def __call__(self, inputs, target):

        # # test if is inputs is list / numpy
        # if isinstance(inputs,list):
        #     for input_id, input_ in enumerate(inputs):
        #         inputs[input_id] = self.__cropCenter__(input_)
        # else: # else it is numpy
        #     inputs = self.__cropCenter__(inputs)

        # # for now suport only one target
        # target = self.__cropCenter__(target)


        inputs = apply_function_list(inputs, self.__cropCenter__)
        target = apply_function_list(target, self.__cropCenter__)
            
        return inputs,target


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __randomCrop__(self, input_, x, y, tw, th):
        return input_[y: y + th,x: x + tw]

    def __call__(self, inputs, targets):

        # import matplotlib.pyplot as plt
        # plt.imshow(inputs[0])

        # test if is inputs is list / numpy
        if isinstance(inputs,list):

            h, w, _ = inputs[0].shape
            th, tw = self.size
            if w == tw and h == th:
                return inputs, targets

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            for input_id, input_ in enumerate(inputs):
                inputs[input_id] = self.__randomCrop__(input_, x1, y1, tw, th)
            
        else: # else it is numpy
            h, w, _ = inputs.shape
            th, tw = self.size
            if w == tw and h == th:
                return inputs,targets

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            inputs = self.__randomCrop__(inputs, x1, y1, tw, th)

    
        if isinstance(targets, list):
            for target_id, target_ in enumerate(targets):
                targets[target_id] = self.__randomCrop__(target_, x1, y1, tw, th)
        else:
            targets = self.__randomCrop__(targets, x1, y1, tw, th)
        # plt.figure()
        # plt.imshow(inputs[0])
        # plt.show()

        return inputs,targets



class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __HorizontalFlip__(self, input_):
        return np.copy(np.fliplr(input_))

    def __call__(self, inputs, targets):
        if random.random() < 0.5:
            inputs = apply_function_list(inputs, self.__HorizontalFlip__)
            targets = apply_function_list(targets, self.__HorizontalFlip__)
        return inputs,targets



class RandomDispHole(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __CreateHole__(self, input_):

        h, w, c = input_.shape

        if c == 1:
            hole_width = int(random.uniform(0,int(w/3)))

            hole_start = int(random.uniform(0,w-hole_width))
            
            min_val = np.min(input_)
            input_[:,hole_start:hole_start+hole_width-1] = min_val

        return input_

    def __call__(self, inputs, targets):

        if random.random() < 0.33:
            inputs = apply_function_list(inputs, self.__CreateHole__)
        return inputs,targets


class RandomFlowHole(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __CreateHole__(self, input_):

        h, w, c = input_.shape

        th = int(w/3.)

        if c == 2:
            hole_start = int(random.uniform(th,w - th))
            hole_end = int(hole_start + th)
            min_val = np.min(input_)
            input_[:,hole_start:hole_end] = min_val

        return input_

    def __call__(self, inputs, targets):

        inputs = apply_function_list(inputs, self.__CreateHole__)
        return inputs,targets


class RandomColorJitter(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None, gamma=None):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma

        self.brightness_factor = 0.
        self.contrast_factor = 0.
        self.saturation_factor = 0.
        self.hue_factor = 0.
        self.gamma_factor = 0.

    def __ColorJitter__(self, input_):

        if input_.shape[2] < 3:
            return input_

        img = Image.fromarray(np.uint8(input_ * 255.),'RGB')

        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        l = [1,2,3,4,5]

        random.shuffle(l)

        for e in l:

            # if e == 1 and self.brightness is not None and random.random() < 0.5:
            if e == 1 and self.brightness is not None:
                img = TF.adjust_brightness(img, self.brightness_factor)
            
            # if e == 2 and self.contrast is not None and random.random() < 0.5:
            if e == 2 and self.contrast is not None:
                img = TF.adjust_contrast(img, self.contrast_factor)

            # if e == 3 and self.saturation is not None and random.random() < 0.5:
            if e == 3 and self.saturation is not None:
                img = TF.adjust_saturation(img, self.saturation_factor)
            
            # if e == 4 and self.hue is not None and random.random() < 0.5:
            if e == 4 and self.hue is not None:
                img = TF.adjust_hue(img, self.hue_factor)
            
            # if e == 4 and self.hue is not None and random.random() < 0.5:
            if e == 5 and self.gamma is not None:
                img = TF.adjust_gamma(img, self.gamma_factor)                
        
        return np.array(img).astype(np.float32) / 255.


    def __call__(self, inputs, targets):

        self.brightness_factor = random.uniform(1. - self.brightness, 1. + self.brightness)
        self.contrast_factor = random.uniform(1. - self.contrast, 1. + self.contrast)
        self.saturation_factor = random.uniform(1. - self.saturation, 1. + self.saturation)
        self.hue_factor = random.uniform(-self.hue, self.hue)
        self.gamma_factor = random.uniform(1. - self.gamma, 1. + self.gamma)

        inputs = apply_function_list(inputs, self.__ColorJitter__)
        targets = apply_function_list(targets, self.__ColorJitter__)

        return inputs,targets





class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, scale_min, scale_max, min_val, cv_interpolation=cv2.INTER_NEAREST):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interp = cv_interpolation
        self.scale = 1.
        self.min_val = min_val

    def __ScalingInputs__(self, input_):

        th, tw, _ = input_.shape

        # We test if it is SGBM or RGB
        if input_.shape[2] == 3:
            input_ = cv2.resize(input_, None, fx = self.scale, fy = self.scale, interpolation=self.interp)
        else:
            input_ = self.scale * cv2.resize(input_, None, fx = self.scale, fy = self.scale, interpolation=self.interp)
            input_ = input_.reshape(input_.shape+(1,))

        # We test if it is SGBM
        # if len(input_.shape) == 2:

        #     if np.min(input_) < self.min_val * 10:
        #         input_[input_ < self.min_val * self.scale] = -1000.

        #     input_ = input_.reshape(input_.shape+(1,))
        
        h1, w1, _ = input_.shape

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        input_ = input_[y1: y1 + th, x1: x1 + tw]
        
        # We test if it is SGBM
        if input_.shape[2] == 1:
            min_disp = np.min(input_)
            input_[:,:60] = min_disp


        return input_


    def __ScalingTarget__(self, input_):

        th, tw, _ = input_.shape

        if input_.shape[2] == 3:
            input_ = cv2.resize(input_, None, fx = self.scale, fy = self.scale, interpolation=self.interp)
        else:
            input_ = self.scale * cv2.resize(input_, None, fx = self.scale, fy = self.scale, interpolation=self.interp)

        if len(input_.shape) == 2:

            if np.min(input_) < self.min_val * 10:
                input_[input_ < self.min_val * self.scale] = -1000.

            input_ = input_.reshape(input_.shape+(1,))
        
        h1, w1, _ = input_.shape

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        input_ = input_[y1: y1 + th, x1: x1 + tw]

        return input_


    def __call__(self, inputs, targets):

        self.scale = self.scale_min + (self.scale_max - self.scale_min) * random.random()

        inputs = apply_function_list(inputs, self.__ScalingInputs__)
        targets = apply_function_list(targets, self.__ScalingTarget__)

        return inputs,targets




class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __VerticalFlip__(self, input_):
        return np.copy(np.flipud(input_))

    def __call__(self, inputs, targets):
        if random.random() < 0.5:
            # if isinstance(inputs,list):
            #     for input_id, input_ in enumerate(inputs):
            #         inputs[input_id] = self.__VerticalFlip__(input_)
            # else:
            #     inputs = self.__VerticalFlip__(inputs)

            # if isinstance(targets, list):
            #     for target_id, target_ in enumerate(targets):
            #         targets[target_id] = self.__VerticalFlip__(target_)
            # else:
            #     targets = self.__VerticalFlip__(targets)

        
            inputs = apply_function_list(inputs, self.__VerticalFlip__)
            targets = apply_function_list(targets, self.__VerticalFlip__)
        return inputs,targets

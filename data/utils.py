import math
import numpy as np
import torch
import random
from skimage.transform import rotate

"""
code reference: https://github.com/facebookresearch/SlowFast
# written by Hoseong Lee
"""

def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order BGR.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `height` x `width` x`channel`
    Returns:
        img_gray (tensor): blended images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = images #torch.tensor(images)
    gray_channel = (
        0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
    )
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray

def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    Args:
        images1 (tensor): the first images to be blended, the dimension is
            `num frames` x `height` x `width` x`channel`
        images2 (tensor): the second images to be blended, the dimension is
            `num frames` x `height` x `width` x`channel`
        alpha (float): the blending weight.
    Returns:
        (tensor): blended images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    return images1 * alpha + images2 * (1 - alpha)

def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x`channel`
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    images = images.astype(np.float32)
    return images


def brightness_jitter(var, images):
    """
    Perfrom brightness jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x`channel`
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_bright = np.zeros(images.shape) #torch.zeros(images.shape)
    images = blend(images, img_bright, alpha)
    return images


def contrast_jitter(var, images):
    """
    Perfrom contrast jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x`channel`
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_gray = grayscale(images)
    #img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    img_gray[:] = np.mean(img_gray, axis=(1, 2, 3),dtype=np.float32, keepdims=True)
    images = blend(images, img_gray, alpha)
    return images


def saturation_jitter(var, images):
    """
    Perfrom saturation jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x`channel`
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x`channel`
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)

    return images


def lighting_jitter(images, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on the given images.
    Args:
        images (tensor): images to perform lighting jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (list): eigenvalues for PCA jitter.
        eigvec (list[list]): eigenvectors for PCA jitter.
    Returns:
        out_images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    if alphastd == 0:
        return images
    # generate alpha1, alpha2, alpha3.
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = torch.zeros_like(images)
    for idx in range(images.shape[1]):
        out_images[:, idx] = images[:, idx] + rgb[2 - idx]

    return out_images
# image_filter/gaussian_filter.py
import cv2
import numpy as np
from .base_filter import BaseFilter

class GaussianFilter(BaseFilter):
    def __init__(self, sigma=1.0, kernel_size=None):
        """
        Initialize the GaussianFilter
        :param sigma: Standard deviation of the Gaussian kernel
        :param kernel_size: Size of the kernel (automatically calculated if None)
        """
        self.sigma = sigma
        self.kernel_size = kernel_size or int(6 * sigma + 1) | 1  # Ensure odd

    def apply(self, image):
        """Apply Gaussian blur to the image"""
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
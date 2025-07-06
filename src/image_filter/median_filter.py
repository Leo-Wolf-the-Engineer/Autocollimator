# image_filter/median_filter.py
import cv2
from .base_filter import BaseFilter

class MedianFilter(BaseFilter):
    def __init__(self, kernel_size=5):
        """
        Initialize the MedianFilter
        :param kernel_size: Size of the kernel (must be odd)
        """
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def apply(self, image):
        """Apply median filter (good for salt-and-pepper noise)"""
        return cv2.medianBlur(image, self.kernel_size)
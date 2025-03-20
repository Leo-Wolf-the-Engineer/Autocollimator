# image_filter/bilateral_filter.py
import cv2
from .base_filter import BaseFilter

class BilateralFilter(BaseFilter):
    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, image):
        """Apply bilateral filter for edge-preserving smoothing"""
        return cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)
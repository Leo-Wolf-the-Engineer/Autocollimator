# image_filter/background_subtraction_filter.py
import cv2
import numpy as np
from .base_filter import BaseFilter

class BackgroundSubtractionFilter(BaseFilter):
    def __init__(self, method='mean'):
        self.method = method

    def apply(self, image):
        """Subtract background from image"""
        if self.method == 'mean':
            background = np.mean(image)
            return cv2.subtract(image, np.full_like(image, background))
        elif self.method == 'gaussian':
            blur = cv2.GaussianBlur(image, (21, 21), 0)
            return cv2.subtract(image, blur)
        else:
            raise ValueError(f"Unknown background subtraction method: {self.method}")
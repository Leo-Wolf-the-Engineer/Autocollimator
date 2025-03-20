# image_filter/background_subtraction_filter.py
import cv2
import numpy as np
from .base_filter import BaseFilter

class BackgroundSubtractionFilter(BaseFilter):
    def __init__(self, method):
        """
        Initialize the BackgroundSubtractionFilter
        """
        self.method = method

    def apply(self, image):
        """Subtract background from image"""
        background = np.mean(image)
        return cv2.subtract(image, np.full_like(image, background))
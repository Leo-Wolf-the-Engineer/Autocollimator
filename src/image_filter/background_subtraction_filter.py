# image_filter/background_subtraction_filter.py
import cv2
import numpy as np
from .base_filter import BaseFilter

class BackgroundSubtractionFilter(BaseFilter):
    def __init__(self, method, value=None):
        """
        Initialize the BackgroundSubtractionFilter
        """
        self.method = method
        self.value = value

    def apply(self, image):
        """Subtract background from image"""
        if self.method == 'mean':
            background = np.mean(image)
            return cv2.subtract(image, np.full_like(image, background))
        elif self.method == 'fixed':
            if self.value is None:
                raise ValueError("Value must be provided for fixed method")
            return cv2.subtract(image, np.full_like(image, self.value))
        elif self.method == 'median':
            background = np.median(image)
            return cv2.subtract(image, np.full_like(image, background))
        elif self.method == 'square_and_divide':
            background = np.mean(image)
            # Convert to float32 for operations
            float_image = image.astype(np.float32)
            # Square the image
            squared = cv2.multiply(float_image, float_image)
            # Divide by background
            result = cv2.divide(squared, float(background)+40)
            # Convert back to uint8
            return cv2.convertScaleAbs(result)
        else:
            raise ValueError("Background subtraction method not supported")
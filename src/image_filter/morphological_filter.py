# image_filter/morphological_filter.py
import cv2
import numpy as np
from .base_filter import BaseFilter

class MorphologicalFilter(BaseFilter):
    def __init__(self, operation='opening', kernel_size=5, iterations=1):
        self.operation = operation
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def apply(self, image):
        """Apply morphological operations to the image"""
        if self.operation == 'erosion':
            return cv2.erode(image, self.kernel, iterations=self.iterations)
        elif self.operation == 'dilation':
            return cv2.dilate(image, self.kernel, iterations=self.iterations)
        elif self.operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        elif self.operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)
        elif self.operation == 'gradient':
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, self.kernel)
        else:
            raise ValueError(f"Unknown morphological operation: {self.operation}")
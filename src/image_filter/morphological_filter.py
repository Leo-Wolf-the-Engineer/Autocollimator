# image_filter/morphological_filter.py
import cv2
import numpy as np
from .base_filter import BaseFilter

class MorphologicalFilter(BaseFilter):
    def __init__(self, operation='opening', kernel_size=5, iterations=1):
        """
        Initialize the MorphologicalFilter
        :param operation: must be one of 'erosion', 'dilation', 'opening', 'closing', or 'gradient'
        :param kernel_size: Size of the kernel
        :param iterations: Number of times to apply the operation
        """
        self.operation = operation
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def apply(self, image):
        """Apply morphological operations to the image"""
        if self.operation == 'erosion':
            return cv2.erode(image, self.kernel, iterations=self.iterations)
            #Shrinks bright regions, expands dark regions
            #Removes small bright details
            #Useful for removing small noise or breaking connections
        elif self.operation == 'dilation':
            return cv2.dilate(image, self.kernel, iterations=self.iterations)
            #Enlarges bright regions, shrinks dark regions
            #Fills small holes and connects nearby objects
            #Useful for joining broken parts of objects
        elif self.operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
            #Removes small bright objects
            #Preserves shape and size of larger objects
            #Smooths contours without significantly changing area
        elif self.operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)
            #Fills small dark holes within objects
            #Closes small gaps in contours
            #Smooths boundaries while preserving overall shape
        elif self.operation == 'gradient':
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, self.kernel)
            #Extracts object boundaries
            #Highlights edges in the image
            #Creates an outline effect
        else:
            raise ValueError(f"Unknown morphological operation: {self.operation}")
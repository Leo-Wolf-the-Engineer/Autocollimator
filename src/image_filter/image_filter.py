import numpy as np
import cv2
import logging

# Import will be needed when you implement the actual filter classes
# from image_filter.gaussian_filter import GaussianFilter
# from image_filter.median_filter import MedianFilter
# etc.

class ImageFilter:
    """
    Main filter class that selects and applies different image filtering techniques
    """
    def __init__(self, filter_type, **kwargs):
        """
        Initialize with specified filter type

        Args:
            filter_type: Type of filter to use
            **kwargs: Filter-specific parameters
        """
        self.filter_type = filter_type
        self.filter_instance = self._create_filter(filter_type, **kwargs)

    def _create_filter(self, filter_type, **kwargs):
        """Create the appropriate filter instance"""
        if filter_type == "Gaussian":
            return GaussianFilter(**kwargs)
        elif filter_type == "Median":
            return MedianFilter(**kwargs)
        elif filter_type == "Bilateral":
            return BilateralFilter(**kwargs)
        elif filter_type == "Background":
            return BackgroundSubtractionFilter(**kwargs)
        elif filter_type == "Morphological":
            return MorphologicalFilter(**kwargs)
        elif filter_type == "Fourier":
            return FourierFilter(**kwargs)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def apply(self, image):
        """Apply the filter to the input image"""
        return self.filter_instance.apply(image)

    def get_parameters(self):
        """Get current filter parameters"""
        return self.filter_instance.get_parameters()

    def set_parameters(self, **kwargs):
        """Update filter parameters"""
        self.filter_instance.set_parameters(**kwargs)
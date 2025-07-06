# Expose all filter classes directly from the package
from .gaussian_filter import GaussianFilter
from .median_filter import MedianFilter
from .bilateral_filter import BilateralFilter
from .background_subtraction_filter import BackgroundSubtractionFilter
from .morphological_filter import MorphologicalFilter
from .fourier_filter import FourierFilter

# Import and expose the main ImageFilter class
from .image_filter import ImageFilter

# Define which classes are available when using "from image_filter import *"
__all__ = [
    'GaussianFilter',
    'MedianFilter',
    'BilateralFilter',
    'BackgroundSubtractionFilter',
    'MorphologicalFilter',
    'FourierFilter',
    'ImageFilter'
]
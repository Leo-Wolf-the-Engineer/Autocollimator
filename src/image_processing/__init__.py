# Make processor classes available directly from the package
from .fast_gaussian_processor import FastGaussian
from .accurate_gaussian_processor import AccurateGaussian
from .peakfinder_processor import Peakfinder
from .linefit_processor import Linefit
from .dummy_processor import Dummy
from .weighted_peakfinder_processor import WeightedPeakfinder

# Import and expose the ImageProcessor class
from .image_processing import ImageProcessor
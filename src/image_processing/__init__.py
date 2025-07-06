# Make processor classes available directly from the package
from .fast_gaussian import FastGaussian
from .accurate_gaussian import AccurateGaussian
from .peakfinder import Peakfinder
from .linefit import Linefit
from .dummy import Dummy
from .weighted_peakfinder import WeightedPeakfinder

# Import and expose the ImageProcessor class
from .image_processing import ImageProcessor
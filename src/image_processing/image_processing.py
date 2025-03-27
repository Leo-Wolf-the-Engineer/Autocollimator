import numpy as np
import logging

# Import processor classes
from image_processing.fast_gaussian_processor import FastGaussian
from image_processing.accurate_gaussian_processor import AccurateGaussian
from image_processing.peakfinder_processor import Peakfinder
from image_processing.linefit_processor import Linefit
from image_processing.dummy_processor import Dummy
from image_processing.weighted_peakfinder_processor import WeightedPeakfinder

# Configure logging with location information
logging.basicConfig(
    level=logging.DEBUG,  # Set your desired log level
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    datefmt='%H:%M:%S'
)

class ImageProcessor:
    def __init__(self, processor_type, Width, Heigth, **kwargs):
        """
        Initialize the ImageProcessor
        :param processor_type: must be a certain type
        """
        self.latest_frame = None
        self.Width = Width
        self.Heigth = Heigth

        # Use dictionary-based processor selection for faster lookup
        processors = {
            "FastGaussian": FastGaussian,
            "AccurateGaussian": AccurateGaussian,
            "Peakfinder": Peakfinder,
            "Linefit": Linefit,
            "Dummy": Dummy,
            "WeightedPeakfinder": WeightedPeakfinder
        }
        
        processor_class = processors.get(processor_type)
        if processor_class is None:
            raise ValueError("processor_type does not exist")
        
        self.Processor = processor_class(**kwargs)

    def process_frame(self, frame):
        """
        Process the frame using the selected processor
        Values are then centered around the center of the frame
        :param frame: The input image frame
        :return: The calculated peak positions in Pixels
        """
        values_X, values_Y = self.Processor.process_frame(frame)

        # Convert to numpy arrays if not already
        values_X = np.asarray(values_X, dtype=float)
        values_Y = np.asarray(values_Y, dtype=float)
        
        # Ensure arrays are at least 1D
        if values_X.ndim == 0:
            values_X = np.array([values_X])
        if values_Y.ndim == 0:
            values_Y = np.array([values_Y])

        # Create mask for valid values - vectorized operation
        valid_mask = ~(np.isnan(values_X) | np.isnan(values_Y) | 
                      (values_X < 0) | (values_X >= self.Width) | 
                      (values_Y < 0) | (values_Y >= self.Heigth))
        
        # Extract only valid values using mask
        values_X = values_X[valid_mask]
        values_Y = values_Y[valid_mask]

        # Validate that both arrays have the same length
        if len(values_X) != len(values_Y):
            raise ValueError("X and Y values must have the same length")

        # Center values (in-place operation)
        values_X -= self.Width / 2
        values_Y -= self.Heigth / 2

        return values_X, values_Y

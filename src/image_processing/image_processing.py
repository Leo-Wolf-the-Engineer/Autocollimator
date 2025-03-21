import numpy as np
import logging

# Import processor classes
from image_processing.fast_gaussian_processor import FastGaussian
from image_processing.accurate_gaussian_processor import AccurateGaussian
from image_processing.peakfinder_processor import Peakfinder
from image_processing.linefit_processor import Linefit
from image_processing.dummy_processor import Dummy

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
        :param processor_type: must be either 'FastGaussian', 'AccurateGaussian', 'Peakfinder' or 'Linefit'
        """
        self.latest_frame = None
        self.Width = Width
        self.Heigth = Heigth

        if processor_type == "FastGaussian":
            self.Processor = FastGaussian(**kwargs)
        elif processor_type == "AccurateGaussian":
            self.Processor = AccurateGaussian(**kwargs)
        elif processor_type == "Peakfinder":
            self.Processor = Peakfinder(**kwargs)
        elif processor_type == "Linefit":
            self.Processor = Linefit(**kwargs)
        elif processor_type  == "Dummy":
            self.Processor = Dummy(**kwargs)
        else
            raise ValueError("processor_type does not exist")

    def process_frame(self, frame):
        """
        Process the frame using the selected processor
        Values are then centered around the center of the frame
        :param frame: The input image frame
        :return: The calculated peak positions in Pixels
        """
        values_X, values_Y = self.Processor.process_frame(frame)

        # Ensure values_X and values_Y are iterable
        if not isinstance(values_X, (list, np.ndarray)):
            values_X = [values_X]
        if not isinstance(values_Y, (list, np.ndarray)):
            values_Y = [values_Y]

        # Filter invalid values efficiently
        valid_indices = []
        for i, (x, y) in enumerate(zip(values_X, values_Y)):
            if (not np.isnan(x) and not np.isnan(y) and
                0 <= x < self.Width and 0 <= y < self.Heigth):
                valid_indices.append(i)

        # Extract only valid values
        if len(valid_indices) < len(values_X):
            values_X = [values_X[i] for i in valid_indices]
            values_Y = [values_Y[i] for i in valid_indices]

        # Validate that both lists have the same length
        if len(values_X) != len(values_Y):
            raise ValueError("X and Y values must have the same length")

        # Convert to arrays and center
        values_X = np.array(values_X) - self.Width / 2
        values_Y = np.array(values_Y) - self.Heigth / 2

        return values_X, values_Y

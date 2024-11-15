import numpy as np
from scipy.optimize import curve_fit

#TODO: Add Handling of the case when the peak is not found
#Todo: Limit Range to image size
#Todo: Handle Multiple Peaks
class ImageProcessor:
    def __init__(self, conversion_factor, processor_type):
        """
        Initialize the ImageProcessor
        :param conversion_factor:
        :param processor_type:
        must be either 'Gaussian' or 'CircleFit'
        """
        self.conversion_factor = conversion_factor
        self.latest_frame = None

        if processor_type not in ["Gaussian", "CircleFit", "Linefit"]:
            raise ValueError("processor_type must be either 'Gaussian' or 'CircleFit'")
        self.camera_type = processor_type
        self.Processor = None

        if processor_type == "Gaussian":
            self.Processor = Gaussian_Processor(conversion_factor)

        if processor_type == "CircleFit":
            raise Exception("processor_type CircleFit is not implemented yet")

        if processor_type == "Linefit":
            raise Exception("processor_type Linefit is not implemented yet")

    def process_frame(self, frame):
        """
        Process the frame using the selected processor
        """
        return self.Processor.process_frame(frame)

class Gaussian_Processor:
    def __init__(self, conversion_factor):
        """
        Initialize the Gaussian_Processor
        :param conversion_factor:
        """
        self.conversion_factor = conversion_factor
        self.latest_frame = None

    def gaussian(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    def process_frame(self, frame):
        self.latest_frame = frame

        # Sum the intensities of the grayscale image
        intensity_x = np.sum(frame, axis=0)
        intensity_y = np.sum(frame, axis=1)

        # Fit Gaussian in the X direction
        x = np.arange(frame.shape[1])
        try:
            popt_x, _ = curve_fit(self.gaussian, x, intensity_x, p0=[np.max(intensity_x), np.argmax(intensity_x), 10])
            peak_x_arcsec = popt_x[1] * self.conversion_factor
        except RuntimeError:
            peak_x_arcsec = np.nan

        # Fit Gaussian in the Y direction
        y = np.arange(frame.shape[0])
        try:
            popt_y, _ = curve_fit(self.gaussian, y, intensity_y, p0=[np.max(intensity_y), np.argmax(intensity_y), 10])
            peak_y_arcsec = popt_y[1] * self.conversion_factor
        except RuntimeError:
            peak_y_arcsec = np.nan

        return peak_x_arcsec, peak_y_arcsec
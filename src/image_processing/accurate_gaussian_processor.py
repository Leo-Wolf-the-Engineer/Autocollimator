import numpy as np
from scipy.optimize import curve_fit

class AccurateGaussian:
    def __init__(self, ):
        self.latest_frame = None

    def gaussian(self, x, a, x0, sigma):
        """
        Gaussian function
        :param x: The input variable
        :param a: The amplitude of the Gaussian
        :param x0: The center of the Gaussian
        :param sigma: The standard deviation of the Gaussian
        :return: The value of the Gaussian function at x
        """
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def process_frame(self, frame):
        """
        Process the frame using the Gaussian_Processor
        :param frame: grayscale image
        :return:
        """
        self.latest_frame = frame

        # Sum the intensities of the grayscale image
        intensity_x = np.sum(frame, axis=0)
        intensity_y = np.sum(frame, axis=1)

        # Fit Gaussian in the X direction
        x = np.arange(frame.shape[1])
        try:
            popt_x, _ = curve_fit(self.gaussian, x, intensity_x, p0=[np.max(intensity_x), np.argmax(intensity_x), 10])
            peak_x = popt_x[1]
        except RuntimeError:
            peak_x = np.nan

        # Fit Gaussian in the Y direction
        y = np.arange(frame.shape[0])
        try:
            popt_y, _ = curve_fit(self.gaussian, y, intensity_y, p0=[np.max(intensity_y), np.argmax(intensity_y), 10])
            peak_y = popt_y[1]
        except RuntimeError:
            peak_y = np.nan

        return peak_x, peak_y
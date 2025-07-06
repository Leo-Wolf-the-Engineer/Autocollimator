import numpy as np
from scipy.interpolate import griddata


class Corrector:
    def __init__(self, target_x, actual_x, target_y, actual_y):
        self.target_x = target_x
        self.actual_x = actual_x
        self.target_y = target_y
        self.actual_y = actual_y

        # Create a grid for interpolation
        self.grid_x, self.grid_y = np.meshgrid(np.linspace(min(target_x), max(target_x), 100),
                                               np.linspace(min(target_y), max(target_y), 100))

        # Interpolate the correction values
        self.correction_x = griddata((target_x, target_y), actual_x, (self.grid_x, self.grid_y), method='cubic')
        self.correction_y = griddata((target_x, target_y), actual_y, (self.grid_x, self.grid_y), method='cubic')

    def get_correction(self, x, y):
        """
        Get the correction values for the given position
        :param x:
        :param y:
        :return:
        """
        if target_x is None or target_y is None or actual_x is None or actual_y is None:
            return x, y
        else:
            # Find the nearest correction value for the given position
            correction_x = griddata((self.grid_x.flatten(), self.grid_y.flatten()), self.correction_x.flatten(), (x, y),
                                    method='cubic')
            correction_y = griddata((self.grid_x.flatten(), self.grid_y.flatten()), self.correction_y.flatten(), (x, y),
                                    method='cubic')
            return correction_x, correction_y
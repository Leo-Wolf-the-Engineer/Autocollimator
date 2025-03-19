from collections import deque
import numpy as np
import threading
import time
import logging

# Configure logging with location information
logging.basicConfig(
    level=logging.DEBUG,  # Set your desired log level
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    datefmt='%H:%M:%S'
)

class ContinousDataStorage:
    def __init__(self, conversion_factor, max_points=100000):
        self.conversion_factor = conversion_factor
        self.microrad_conv_factor = conversion_factor * np.pi * 1e3 / 648
        self.max_points = max_points

        # Use deques for efficient append/pop operations
        self.x_data = deque(maxlen=max_points)
        self.y_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)

        # Thread safety
        self.lock = threading.RLock()

        # Cache for faster unit conversions
        self._cached = {}
        self._cache_dirty = True

    def new_data(self, x_values, y_values):
        """
        Add new data point(s) to storage

        :param x_values: Single value, list, or array of x coordinates
        :param y_values: Single value, list, or array of y coordinates
        """
        # Convert inputs to numpy arrays and ensure they're 1D
        x_array = np.atleast_1d(x_values).flatten()
        y_array = np.atleast_1d(y_values).flatten()

        len_x = len(x_array)
        len_y = len(y_array)

        if len_x == len_y and len_x > 0 and len_y > 0 and len_x <= 10 and len_y <= 10:
            # Add data to storage
            with self.lock:
                timestamp = time.time_ns()
                self.timestamps.append(timestamp)
                self.x_data.append(x_array)
                self.y_data.append(y_array)
                self._cache_dirty = True
        else:
            logging.debug(f"Data not added, invalid input - got {len_x} x values and {len_y} y values")

    def get_XY_data(self, unit="Pixels"):
        """
        Retrieve all data as NumPy arrays.

        :param unit: "Pixels", "Arcseconds", or "Microradians"
        :return: Tuple of (x_array, y_array) as NumPy arrays, where each array
                 contains all points flattened into a 1D array

        """
        with self.lock:
            # Handle empty data case
            if len(self.x_data) == 0:
                return np.array([]), np.array([])

            # Use cached results if available
            if not self._cache_dirty and unit in self._cached:
                return self._cached[unit]

            # Concatenate all arrays into flat arrays
            x_array = np.concatenate([np.atleast_1d(x) for x in self.x_data])
            y_array = np.concatenate([np.atleast_1d(y) for y in self.y_data])

            # Apply unit conversion
            if unit == "Pixels":
                result = (x_array, y_array)
            elif unit == "Arcseconds":
                result = (x_array * self.conversion_factor,
                         y_array * self.conversion_factor)
            elif unit == "Microradians":
                result = (x_array * self.microrad_conv_factor,
                         y_array * self.microrad_conv_factor)
            else:
                raise ValueError(f"Unit '{unit}' not recognized.")

            # Cache result
            self._cached[unit] = result
            if unit == "Pixels":
                self._cache_dirty = False

            return result

    def get_timestamps(self):
        """Return timestamps as NumPy array"""
        with self.lock:
            return np.array(list(self.timestamps))

    def get_XY_time_data(self, unit="Pixels"):
        """
        Retrieve all data as NumPy arrays, with timestamps included.

        :param unit: "Pixels", "Arcseconds", or "Microradians"
        :return: Tuple of (x_array, y_array, timestamps) as NumPy arrays, where each array
                 contains all points flattened into a 1D array
        """
        with self.lock:
            x_array, y_array = self.get_XY_data(unit)
            timestamps = self.get_timestamps()
            return x_array, y_array, timestamps

    def clear_data(self):
        """
        Clear all data from the storage.
        """
        self.x_data = []
        self.y_data = []
        self.timestamps = []

# Todo handle multiple peaks
class POIDataStorage:
    def __init__(self, conversion_factor):
        """
        Initialize the data storage
        """
        self.dataX = []
        self.dataY = []
        self.timestamp = []
        self.sample_num = []
        self.conversion_factor = conversion_factor

    def new_data(self, dataX, dataY):
        """
        Add new data to the storage.
        :param values_X: The list of X values in Pixels to be added
        :param values_Y: The list of Y values in Pixels to be added
        """
        if len(dataX) > 10 or len(dataY) > 10:
            raise ValueError("Cannot store more than 10 values at a time")
        else:
            self.dataX.append([value for value in dataX])
            self.dataY.append([value for value in dataY])
            self.timestamp.append(time.time_ns())
            self.sample_num.append(1)

    def average_data(self, index, dataX, dataY):
        """
        Calculate the average of the stored data plus the new one.
        Takes amount of preceding samples into account.
        :param index: The index of the value to be overridden
        :param dataX: The X value to be added
        :param dataY: The Y value to be added
        :return: The average value of the stored data
        """
        if len(dataX) > 10 or len(dataY) > 10:
            raise ValueError("Cannot store more than 10 values at a time")
        if 0 <= index < len(self.dataX):
            self.dataX[index] = (self.dataX[index] * self.sample_num[index] + dataX[0]) / (self.sample_num[index] + 1)
            self.dataY[index] = (self.dataY[index] * self.sample_num[index] + dataY[1]) / (self.sample_num[index] + 1)
            self.timestamp[index] = time.time_ns()
            self.sample_num[index] += 1
        else:
            warnings.warn("Index out of range")

    def get_data(self, unit="Arcseconds"):
        """
        Retrieve all data from the storage.
        :param unit: The unit of the data to be returned, default is Arcseconds, can be pixels or microradians
        :return: A list of all stored data in the specified unit
        """
        if unit == "Arcseconds":
            return [[x * self.conversion_factor for x in sublist] for sublist in self.dataX], \
                [[y * self.conversion_factor for y in sublist] for sublist in self.dataY]
        elif unit == "Pixels":
            return self.dataX, self.dataY
        elif unit == "Microradians":
            return [[x / self.conversion_factor * np.pi * 1e3 / 648 for x in sublist] for sublist in self.dataX], \
                [[y / self.conversion_factor * np.pi * 1e3 / 648 for y in sublist] for sublist in self.dataY]
        else:
            raise Exception("Unit not recognized")

    def get_data_detrended(self, index=None, unit="Arcseconds"):
        """
        Retrieve all detrended data from the storage.
        :param index: The index of the data to be returned (optional)
        :param unit: The unit of the data to be returned, default is Arcseconds, can be pixels or microradians
        :return: The stored detrended data
        """

        def detrend(data, timestamps):
            trend = np.polyval(np.polyfit(timestamps, data, 1), timestamps)
            return [value - trend for value, trend in zip(data, trend)]

        if index is not None:
            if index < len(self.dataX):
                if unit == "Arcseconds":
                    return detrend([x * self.conversion_factor for x in self.dataX[index]], self.timestamp), \
                        detrend([y * self.conversion_factor for y in self.dataY[index]], self.timestamp)
                elif unit == "Pixels":
                    return detrend(self.dataX[index], self.timestamp), \
                        detrend(self.dataY[index], self.timestamp)
                elif unit == "Microradians":
                    return detrend([x / self.conversion_factor * np.pi * 1e3 / 648 for x in self.dataX[index]],
                                   self.timestamp), \
                        detrend([y / self.conversion_factor * np.pi * 1e3 / 648 for y in self.dataY[index]],
                                self.timestamp)
                else:
                    raise Exception("Unit not recognized")
            else:
                raise Exception("Index out of range")
        else:
            if unit == "Arcseconds":
                return [detrend([x * self.conversion_factor for x in sublist], self.timestamp) for sublist in
                        self.dataX], \
                    [detrend([y * self.conversion_factor for y in sublist], self.timestamp) for sublist in self.dataY]
            elif unit == "Pixels":
                return [detrend(sublist, self.timestamp) for sublist in self.dataX], \
                    [detrend(sublist, self.timestamp) for sublist in self.dataY]
            elif unit == "Microradians":
                return [detrend([x / self.conversion_factor * np.pi * 1e3 / 648 for x in sublist], self.timestamp) for
                        sublist in self.dataX], \
                    [detrend([y / self.conversion_factor * np.pi * 1e3 / 648 for y in sublist], self.timestamp) for
                     sublist in self.dataY]
            else:
                raise Exception("Unit not recognized")

    def get_timestamp(self, index):
        """
        Retrieve all data from the storage.
        :param index: The index of the data to be returned (optional)
        :return: The stored data
        """
        if index is not none and index < len(self.timestamp):
            return self.timestamp[index]
        elif index >= len(self.timestamp):
            raise exception("Index out of range")
        else:
            return self.timestamp

    def clear_data(self):
        """
        Clear all data from the storage.
        """
        self.dataX = []
        self.dataY = []
        self.timestamp = []
        self.sample_num = []

from collections import deque
import numpy as np
import threading
import time


class ContinousDataStorage:
    def __init__(self, conversion_factor, max_points=100000):
        self.conversion_factor = conversion_factor
        self.max_points = max_points
        # Use deques for faster append/pop operations
        self.x_data = deque(maxlen=max_points)
        self.y_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        self.lock = threading.RLock()  # Reentrant lock

        # Cache for converted values
        self._cached_arcsec_x = None
        self._cached_arcsec_y = None
        self._cache_dirty = True

    def new_data(self, x_values, y_values):
        # Convert to numpy arrays for vectorized operations
        x_array = np.atleast_1d(x_values)
        y_array = np.atleast_1d(y_values)

        with self.lock:
            timestamp = time.time_ns()
            # Batch append instead of individual appends
            self.timestamps.append(timestamp)
            self.x_data.append(x_array)
            self.y_data.append(y_array)
            self._cache_dirty = True  # Mark cache as dirty

    def get_data(self, unit="Pixels"):
        with self.lock:
            if unit == "Pixels":
                return list(self.x_data), list(self.y_data)
            elif unit == "Arcseconds":
                # Use cached values if available and not dirty
                if self._cache_dirty or self._cached_arcsec_x is None:
                    # Vectorized conversion
                    self._cached_arcsec_x = [x * self.conversion_factor for x in self.x_data]
                    self._cached_arcsec_y = [y * self.conversion_factor for y in self.y_data]
                    self._cache_dirty = False
                return self._cached_arcsec_x, self._cached_arcsec_y

    def get_timestamp(self):
        with self.lock:
            return list(self.timestamps)

    def get_windowed_data(self, time_window_seconds, unit="Pixels"):
        """Retrieve only data from the last N seconds"""
        current_time = time.time_ns()
        window_ns = time_window_seconds * 1e9

        with self.lock:
            # Find indices of elements within time window
            valid_indices = [i for i, ts in enumerate(self.timestamps)
                             if current_time - ts <= window_ns]

            if not valid_indices:
                return [], []

            # Get data only for valid timestamps
            x_windowed = [self.x_data[i] for i in valid_indices]
            y_windowed = [self.y_data[i] for i in valid_indices]

            if unit == "Arcseconds":
                x_windowed = [x * self.conversion_factor for x in x_windowed]
                y_windowed = [y * self.conversion_factor for y in y_windowed]

            return x_windowed, y_windowed

    def get_data(self, unit="Arcseconds"):
        """
        Retrieve all data from the storage.
        :param unit: The unit of the data to be returned, default is Arcseconds, can be Pixels or Microradians
        :return: A list of all stored data in the specified unit
        """
        if unit == "Arcseconds":
            return [[x * self.conversion_factor for x in sublist] for sublist in self.x_data], \
                [[y * self.conversion_factor for y in sublist] for sublist in self.y_data]
        elif unit == "Pixels":
            return self.x_data, self.y_data
        elif unit == "Microradians":
            return [[x / self.conversion_factor * np.pi * 1e3 / 648 for x in sublist] for sublist in self.x_data], \
                [[y / self.conversion_factor * np.pi * 1e3 / 648 for y in sublist] for sublist in self.y_data]
        else:
            raise Exception("Unit not recognized")

    def get_timestamp(self):
        """
        Retrieve all data from the storage.
        :return: A list of all stored data in the specified unit
        """
        return self.timestamps

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

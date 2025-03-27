from collections import deque
import numpy as np
import threading
import time
import logging
import warnings  # Added import for warnings

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
        
        # Precompute unit conversion dictionaries
        self._unit_converters = {
            "Pixels": lambda x: x,
            "Arcseconds": lambda x: x * self.conversion_factor,
            "Microradians": lambda x: x * self.microrad_conv_factor
        }

    def new_data(self, x_values, y_values):
        """
        Add new data point(s) to storage

        :param x_values: Single value, list, or array of x coordinates
        :param y_values: Single value, list, or array of y coordinates
        """
        # Quick validation to avoid unnecessary conversions
        if not isinstance(x_values, np.ndarray) and not isinstance(y_values, np.ndarray):
            if not x_values or not y_values:
                logging.debug("Data not added, empty input")
                return
        
        # Convert inputs to numpy arrays and ensure they're 1D
        try:
            # Optimization: use numpy's faster paths when possible
            x_array = np.asarray(x_values, dtype=np.float32).flatten()
            y_array = np.asarray(y_values, dtype=np.float32).flatten()
        except:
            logging.debug("Failed to convert input to arrays")
            return

        len_x = len(x_array)
        len_y = len(y_array)

        # Reuse timestamp for all entries in this batch
        timestamp = time.time_ns()
        
        if len_x == len_y and len_x > 0 and len_x <= 10 and len_y <= 10:
            # Add data to storage - minimize time spent in the lock
            with self.lock:
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
        :return: Tuple of (x_array, y_array) as NumPy arrays
        """
        # Validate unit early to avoid unnecessary processing
        if unit not in self._unit_converters:
            raise ValueError(f"Unit '{unit}' not recognized.")
            
        with self.lock:
            # Handle empty data case
            if len(self.x_data) == 0:
                return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

            # Use cached results if available
            if not self._cache_dirty and unit in self._cached:
                return self._cached[unit]

            # Concatenate all arrays into flat arrays - preallocate for speed
            # Optimization: calculate total length once
            total_len = sum(len(np.atleast_1d(x)) for x in self.x_data)
            
            x_array = np.empty(total_len, dtype=np.float32)
            y_array = np.empty(total_len, dtype=np.float32)
            
            # Fill the preallocated arrays
            pos = 0
            for x, y in zip(self.x_data, self.y_data):
                x_flat = np.atleast_1d(x)
                y_flat = np.atleast_1d(y)
                x_len = len(x_flat)
                
                x_array[pos:pos+x_len] = x_flat
                y_array[pos:pos+x_len] = y_flat
                pos += x_len

            # Apply unit conversion using precomputed converters
            converter = self._unit_converters[unit]
            result = (converter(x_array), converter(y_array))
            
            # Cache result
            self._cached[unit] = result
            if unit == "Pixels":
                self._cache_dirty = False

            return result

    def get_timestamps(self):
        """Return timestamps as NumPy array"""
        with self.lock:
            # Optimization: preallocate array with correct dtype
            return np.array(self.timestamps, dtype=np.int64)

    def get_XY_time_data(self, unit="Pixels"):
        """
        Retrieve all data as NumPy arrays, with timestamps included.

        :param unit: "Pixels", "Arcseconds", or "Microradians"
        :return: Tuple of (x_array, y_array, timestamps) as NumPy arrays
        """
        with self.lock:
            x_array, y_array = self.get_XY_data(unit)
            timestamps = self.get_timestamps()
            return x_array, y_array, timestamps

    def clear_data(self):
        """
        Clear all data from the storage.
        """
        with self.lock:
            self.x_data.clear()
            self.y_data.clear()
            self.timestamps.clear()
            self._cached.clear()
            self._cache_dirty = True

class POIDataStorage:
    def __init__(self, conversion_factor):
        """
        Initialize the data storage
        """
        self.conversion_factor = conversion_factor
        self.microrad_factor = self.conversion_factor * np.pi * 1e3 / 648
        
        # Use numpy arrays instead of lists for better performance
        self.dataX = np.array([], dtype=object)
        self.dataY = np.array([], dtype=object)
        self.timestamp = np.array([], dtype=np.int64)
        self.sample_num = np.array([], dtype=np.int32)
        
        # Precompute conversion factors
        self._unit_converters = {
            "Pixels": lambda x: x,
            "Arcseconds": lambda x: x * self.conversion_factor,
            "Microradians": lambda x: x * self.microrad_factor
        }

    def new_data(self, dataX, dataY):
        """
        Add new data to the storage.
        :param dataX: The list of X values in Pixels to be added
        :param dataY: The list of Y values in Pixels to be added
        """
        if len(dataX) > 10 or len(dataY) > 10:
            raise ValueError("Cannot store more than 10 values at a time")
        else:
            # Convert to numpy arrays for efficiency
            x_values = np.array(dataX, dtype=np.float32)
            y_values = np.array(dataY, dtype=np.float32)
            
            # Append to arrays
            self.dataX = np.append(self.dataX, [x_values])
            self.dataY = np.append(self.dataY, [y_values])
            self.timestamp = np.append(self.timestamp, time.time_ns())
            self.sample_num = np.append(self.sample_num, 1)

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
            # Fix: properly handle array operations for averaging
            x_data = np.asarray(self.dataX[index])
            y_data = np.asarray(self.dataY[index])
            
            # Weighted average calculation
            sample_count = self.sample_num[index]
            self.dataX[index] = (x_data * sample_count + np.asarray(dataX)[0]) / (sample_count + 1)
            self.dataY[index] = (y_data * sample_count + np.asarray(dataY)[0]) / (sample_count + 1)
            self.timestamp[index] = time.time_ns()
            self.sample_num[index] += 1
        else:
            warnings.warn("Index out of range")

    def get_data(self, unit="Arcseconds"):
        """
        Retrieve all data from the storage.
        :param unit: The unit of the data to be returned, default is Arcseconds
        :return: A list of all stored data in the specified unit
        """
        if unit not in self._unit_converters:
            raise ValueError(f"Unit '{unit}' not recognized")
        
        converter = self._unit_converters[unit]
        
        # Vectorized conversion for better performance
        result_x = []
        result_y = []
        
        for x_subarray, y_subarray in zip(self.dataX, self.dataY):
            result_x.append(converter(np.asarray(x_subarray)))
            result_y.append(converter(np.asarray(y_subarray)))
            
        return result_x, result_y

    def get_data_detrended(self, index=None, unit="Arcseconds"):
        """
        Retrieve all detrended data from the storage.
        :param index: The index of the data to be returned (optional)
        :param unit: The unit of the data to be returned
        :return: The stored detrended data
        """
        if unit not in self._unit_converters:
            raise ValueError(f"Unit '{unit}' not recognized")
            
        converter = self._unit_converters[unit]

        def detrend(data, timestamps):
            # Optimization: use numpy operations for detrending
            data_array = np.asarray(data, dtype=np.float32)
            timestamps_array = np.asarray(timestamps, dtype=np.float64)
            
            # Fast polyfit for detrending
            coeffs = np.polyfit(timestamps_array, data_array, 1)
            trend = np.polyval(coeffs, timestamps_array)
            
            return data_array - trend

        if index is not None:
            if index < len(self.dataX):
                x_values = converter(np.asarray(self.dataX[index]))
                y_values = converter(np.asarray(self.dataY[index]))
                
                return (detrend(x_values, self.timestamp), 
                        detrend(y_values, self.timestamp))
            else:
                raise IndexError("Index out of range")
        else:
            result_x = []
            result_y = []
            
            for i in range(len(self.dataX)):
                x_values = converter(np.asarray(self.dataX[i]))
                y_values = converter(np.asarray(self.dataY[i]))
                
                result_x.append(detrend(x_values, self.timestamp))
                result_y.append(detrend(y_values, self.timestamp))
                
            return result_x, result_y

    def get_timestamp(self, index=None):
        """
        Retrieve timestamp data from the storage.
        :param index: The index of the data to be returned (optional)
        :return: The stored timestamps
        """
        if index is not None and index < len(self.timestamp):
            return self.timestamp[index]
        elif index is not None and index >= len(self.timestamp):
            raise IndexError("Index out of range")
        else:
            return self.timestamp

    def clear_data(self):
        """
        Clear all data from the storage.
        """
        self.dataX = np.array([], dtype=object)
        self.dataY = np.array([], dtype=object)
        self.timestamp = np.array([], dtype=np.int64)
        self.sample_num = np.array([], dtype=np.int32)

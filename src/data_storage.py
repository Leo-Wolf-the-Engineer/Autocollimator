import time
import warnings
import numpy as np


class ContinousDataStorage:
    def __init__(self, conversion_factor):
        """
        Initialize the data storage
        """
        self.dataX = []
        self.dataY = []
        self.time = []
        self.conversion_factor = conversion_factor

    def new_data(self, values_X, values_Y):
        """
        Add new data to the storage.
                :param values_X: The list of X values in Pixels to be added
        :param values_Y: The list of Y in Pixels values to be added
        """
        if len(values_X) > 10 or len(values_Y) > 10:
            raise ValueError("Cannot store more than 10 values at a time")
        else:
            for value in values_X:
                if value == np.nan or value > self.Width or value < 0:
                    values_X.remove(value)
                    values_Y.remove(value)
                    warnings.warn("Removing Nan or out of range value Pair...")
            for value in values_Y:
                if value == np.nan or value > self.Heigth or value < 0:
                    values_X.remove(value)
                    values_Y.remove(value)
                    warnings.warn("Removing Nan or out of range value Pair...")

        self.dataX.append([value - self.offsetX for value in values_X])
        self.dataY.append([value - self.offsetY for value in values_Y])
        self.time.append(time.time_ns())

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

    def get_timestamp(self):
        """
        Retrieve all data from the storage.
        :return: A list of all stored data in the specified unit
        """
        return self.time

    def clear_data(self):
        """
        Clear all data from the storage.
        """
        self.dataX = []
        self.dataY = []
        self.time = []


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

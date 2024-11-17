import time
import warnings
import numpy as np


class ContinousDataStorage:
    def __init__(self, Heigth, Width, conversion_factor):
        """
        Initialize the data storage
        """
        self.Heigth = Heigth
        self.Width = Width
        # position is stored in pixels
        self.dataX = []
        self.dataY = []
        self.time = []
        self.offsetX = Width / 2
        self.offsetY = Heigth / 2
        self.conversion_factor = conversion_factor

    def new_data(self, value_X, value_Y):
        """
        Add new data to the storage.
        :param value_X: The X value to be added
        :param value_Y: The Y value to be added
        """
        if value_X > self.Width or value_X < 0 or value_Y > self.Heigth or value_Y < 0 or value_X is Nan or value_Y is None:
            raise warnings.warn("Value out of range")
        else:
            self.dataX.append(value_X - self.offsetX)
            self.dataY.append(value_Y - self.offsetY)
            self.time.append(time.time_ns())

    def get_data(self, unit="Arcseconds"):
        """
        Retrieve all data from the storage.
        :param unit: The unit of the data to be returned, default is Arcseconds, can be pixels or microradians
        :return: A list of all stored data in the specified unit
        """
        if unit == "Arcseconds":
            return self.dataX * self.conversion_factor, self.dataY * self.conversion_factor
        elif unit == "Pixels":
            return self.dataX, self.dataY
        elif unit == "Microradians":
            return self.dataX / self.conversion_factor * np.pi * 1e3 / 648, self.dataY / self.conversion_factor * np.pi * 1e3 / 648
        else:
            raise Exception("Unit not recognized")

    def get_data_mov_avg(self, unit="Arcseconds"):
        """
        Retrieve all data from the storage and filter with a moving average.
        :param index: The index of the data to be returned
        :param unit: The unit of the data to be returned, default is Arcseconds, can be pixels or microradians
        :return: The stored data
        """
        if unit == "Arcseconds":
            return np.convolve(self.dataX, np.ones(5) / 5, mode='valid') * self.conversion_factor, np.convolve(self.dataY, np.ones(5) / 5, mode='valid') * self.conversion_factor
        elif unit == "Pixels":
            return np.convolve(self.dataX, np.ones(5) / 5, mode='valid'), np.convolve(self.dataY, np.ones(5) / 5, mode='valid')
        elif unit == "Microradians":
            return np.convolve(self.dataX, np.ones(5) / 5, mode='valid') / self.conversion_factor * np.pi * 1e3 / 648, np.convolve(self.dataY, np.ones(5) / 5, mode='valid') / self.conversion_factor * np.pi * 1e3 / 648
        else:
            raise Exception("Unit not recognized")

    def clear_data(self):
        """
        Clear all data from the storage.
        """
        self.dataX = []
        self.dataY = []
        self.time = []


# Todo handle multiple peaks
class POIDataStorage:
    def __init__(self, Heigth, Width, conversion_factor):
        """
        Initialize the data storage
        """
        self.Heigth = Heigth
        self.Width = Width
        self.dataX = []
        self.dataY = []
        self.timestamp = []
        self.sample_num = []
        self.offsetX = Width / 2
        self.offsetY = Heigth / 2
        self.conversion_factor = conversion_factor

    def new_data(self, index, dataX, dataY):
        """
        Override specific value or add a new one in the data storage.
        :param index: The index of the value to be overridden
        :param dataX: The X value to be added
        :param dataY: The Y value to be added
        """
        value_X = np.average(checkdata(dataX))
        value_Y = np.average(checkdata(dataY))
        if 0 <= index < len(self.dataX):
            self.dataX[index] = value_X - self.offsetX
            self.dataY[index] = value_Y - self.offsetY
            self.timestamp[index] = time.time_ns()
            self.sample_num[index] = 1
        elif index == len(self.dataX):
            self.dataX.append(value_X - self.offsetX)
            self.dataY.append(value_Y - self.offsetY)
            self.timestamp.append(time.time_ns())
            self.sample_num.append(1)
        else:
            raise Exception("Index out of range")

    def checkdata(self, values):
        """
        Check if the value is within the range of the image
        :param index: The index of the value to be checked
        :param values: Vector of values to be checked
        """
        checked_values = []
        for value in values:
            if value is not nan and value <= self.Heigth and value >= 0:
                checked_values.append(value)
        if not checked_values:
            raise warnings.warn("No valid values")
        return checked_values

    def average_data(self, index, dataX, dataY):
        """
        Calculate the average of the stored data plus the new one.
        Takes amount of preceding samples into account.
        :param index: The index of the value to be overridden
        :param dataX: The X value to be added
        :param dataY: The Y value to be added
        :return: The average value of the stored data
        """
        value_X = np.average(checkdata(dataX))
        value_Y = np.average(checkdata(dataY))
        if 0 <= index < len(self.dataX):
            self.dataX[index] = (self.dataX[index] * self.sample_num[index] + value_X[0]) / (self.sample_num[index] + 1)
            self.dataY[index] = (self.dataY[index] * self.sample_num[index] + value_Y[1]) / (self.sample_num[index] + 1)
            self.timestamp[index] = time.time_ns()
            self.sample_num[index] += 1
        else:
            raise Exception("Index out of range")

    def get_data(self, index, unit="Arcseconds"):
        """
        Retrieve all data from the storage.
        :param index: The index of the data to be returned (optional)
        :param unit: The unit of the data to be returned, default is Arcseconds, can be pixels or microradians
        :return: The stored data
        """
        if index is not none and index < len(self.dataX):
            if unit == "Arcseconds":
                return self.dataX[index] * self.conversion_factor, self.dataY[index] * self.conversion_factor
            elif unit == "Pixels":
                return self.dataX[index], self.dataY[index]
            elif unit == "Microradians":
                return self.dataX[index] / self.conversion_factor * np.pi * 1e3 / 648, self.dataY[index] / self.conversion_factor * np.pi * 1e3 / 648
            else:
                raise Exception("Unit not recognized")
        elif index >= len(self.dataX):
            raise exception("Index out of range")
        else:
            if unit == "Arcseconds":
                return self.dataX * self.conversion_factor, self.dataY * self.conversion_factor
            elif unit == "Pixels":
                return self.dataX, self.dataY
            elif unit == "Microradians":
                return self.dataX / self.conversion_factor * np.pi * 1e3 / 648, self.dataY / self.conversion_factor * np.pi * 1e3 / 648
            else:
                raise Exception("Unit not recognized")


    def get_data_detrended(self, index, unit="Arcseconds"):
        """
        Retrieve all data from the storage.
        :param index: The index of the data to be returned (optional)
        :param unit: The unit of the data to be returned, default is Arcseconds, can be pixels or microradians
        :return: The stored detrended data
        """
        if index is not none and index < len(self.dataX):
            if unit == "Arcseconds":
                return self.dataX[index] * self.conversion_factor - np.polyval(np.polyfit(self.timestamp, self.dataX, 1), self.timestamp), self.dataY[index] * self.conversion_factor - np.polyval(np.polyfit(self.timestamp, self.dataY, 1), self.timestamp)
            elif unit == "Pixels":
                return self.dataX[index] - np.polyval(np.polyfit(self.timestamp, self.dataX, 1), self.timestamp), self.dataY[index] - np.polyval(np.polyfit(self.timestamp, self.dataY, 1), self.timestamp)
            elif unit == "Microradians":
                return self.dataX[index] / self.conversion_factor * np.pi * 1e3 / 648 - np.polyval(np.polyfit(self.timestamp, self.dataX, 1), self.timestamp), self.dataY[index] / self.conversion_factor * np.pi * 1e3 / 648 - np.polyval(np.polyfit(self.timestamp, self.dataY, 1), self.timestamp)
            else:
                raise Exception("Unit not recognized")
        elif index >= len(self.dataX):
            raise exception("Index out of range")
        else:
            if unit == "Arcseconds":
                return self.dataX * self.conversion_factor - np.polyval(np.polyfit(self.timestamp, self.dataX, 1), self.timestamp), self.dataY * self.conversion_factor - np.polyval(np.polyfit(self.timestamp, self.dataY, 1), self.timestamp)
            elif unit == "Pixels":
                return self.dataX - np.polyval(np.polyfit(self.timestamp, self.dataX, 1), self.timestamp), self.dataY - np.polyval(np.polyfit(self.timestamp, self.dataY, 1), self.timestamp)
            elif unit == "Microradians":
                return self.dataX / self.conversion_factor * np.pi * 1e3 / 648 - np.polyval(np.polyfit(self.timestamp, self.dataX, 1), self.timestamp), self.dataY / self.conversion_factor * np.pi * 1e3 / 648 - np.polyval(np.polyfit(self.timestamp, self.dataY, 1), self.timestamp)
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

import numpy as np
import cv2
from pypylon import pylon
from scipy.optimize import curve_fit
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QScreen
from pyqtgraph.exporters import ImageExporter
import threading
import time
from image_aquisition import CameraManager
import utils
from image_processing import ImageProcessor
from win_live import AutocollimatorLiveWindow
from win_straightness import StraightnessMeasurementWindow

# Constants for conversion from pixels to arcseconds
PIXEL_PITCH = 3.45e-6  # in meters
FOCAL_LENGTH = 0.385  # in meters
CONVERSION_FACTOR = PIXEL_PITCH / (2 * FOCAL_LENGTH) * 180 / np.pi * 3600

# Initialize the camera
camera = CameraManager("Basler")

# Initialize the ImageProcessor instance
processor = ImageProcessor(CONVERSION_FACTOR, "Gaussian")

# Initialize PyQtGraph application
app = QtWidgets.QApplication([])

# Create and show the Autocollimator live window
autocollimator_live_window = AutocollimatorLiveWindow(processor, CONVERSION_FACTOR, app)
autocollimator_live_window.win.show()

# Create and show the Straightness Measurement window
straightness_measurement_window = StraightnessMeasurementWindow(app, autocollimator_live_window)
straightness_measurement_window.win.show()

# Function to grab frames and process them
def grab_and_process():
    while True:
        frame = camera.retrieve_frame()

        # Update frame count for FPS calculation
        autocollimator_live_window.frame_count += 1

        # Process the frame using the ImageProcessor instance
        peak_x_arcsec, peak_y_arcsec = processor.process_frame(frame)

        # Zero the peak positions
        peak_x_arcsec -= autocollimator_live_window.zero_x
        peak_y_arcsec -= autocollimator_live_window.zero_y

        # Store peak positions
        autocollimator_live_window.peak_x_history.append(
            ((time.time_ns() - autocollimator_live_window.average_start_time) / 6e10, peak_x_arcsec))
        autocollimator_live_window.peak_y_history.append(
            ((time.time_ns() - autocollimator_live_window.average_start_time) / 6e10, peak_y_arcsec))

        # Update the latest frame and peaks
        autocollimator_live_window.latest_frame = frame
        autocollimator_live_window.latest_peak_x = peak_x_arcsec if not np.isnan(
            peak_x_arcsec) else autocollimator_live_window.latest_peak_x
        autocollimator_live_window.latest_peak_y = peak_y_arcsec if not np.isnan(
            peak_y_arcsec) else autocollimator_live_window.latest_peak_y

        # Print the determined values
        print(f"X Direction: Mean={peak_x_arcsec:.2f} arcseconds")
        print(f"Y Direction: Mean={peak_y_arcsec:.2f} arcseconds")

        # Handle averaging
        if autocollimator_live_window.averaging:
            current_time = time.time()
            if current_time - autocollimator_live_window.average_start_time <= 3:
                if not np.isnan(peak_x_arcsec):
                    autocollimator_live_window.average_x_values.append(peak_x_arcsec)
                if not np.isnan(peak_y_arcsec):
                    autocollimator_live_window.average_y_values.append(peak_y_arcsec)
            else:
                autocollimator_live_window.averaging = False
                avg_x = np.mean(
                    autocollimator_live_window.average_x_values) if autocollimator_live_window.average_x_values else 0
                avg_y = np.mean(
                    autocollimator_live_window.average_y_values) if autocollimator_live_window.average_y_values else 0
                autocollimator_live_window.average_display.setText(f"Averaged Values: X = {avg_x:.2f}, Y = {avg_y:.2f}")

# Start a thread for grabbing and processing frames
thread = threading.Thread(target=grab_and_process)
thread.daemon = True
thread.start()

# Start the PyQtGraph application
app.exec_()

# Close the camera when the application is closed
camera.close()
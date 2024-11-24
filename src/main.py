import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
import threading
import time
from image_aquisition import CameraManager
from image_processing import ImageProcessor
from win_live import AutocollimatorLiveWindow
from win_straightness import StraightnessMeasurementWindow
from data_storage import ContinousDataStorage
from data_storage import POIDataStorage
from calibration import Corrector
import warnings

# Constants for conversion from pixels to arcseconds
PIXEL_PITCH = 3.45e-6  # in meters
FOCAL_LENGTH = 0.385  # in meters
CONVERSION_FACTOR = PIXEL_PITCH / (2 * FOCAL_LENGTH) * 180 / np.pi * 3600

# Init Frame Storage
image_frame_storage = []

# Initialize the camera
camera = CameraManager("Basler")

# Initialize the ImageProcessor instance
imagewidth, imageheight = camera.get_image_size()
processor = ImageProcessor("Gaussian", imagewidth, imageheight)

# Initialize the corrector
#target_x = []
#actual_x = []
#target_y = []
#actual_y = []
#Corrector = Corrector(target_x, actual_x, target_y, actual_y)

# Initialize PyQtGraph application
app = QtWidgets.QApplication([])

# Initialize the data storage
ContinousDataStorage = ContinousDataStorage(CONVERSION_FACTOR)

# Initialize the data storage for straightness measurements
straightness_data = POIDataStorage(CONVERSION_FACTOR)

# Create and show the Autocollimator live window
autocollimator_live_window = AutocollimatorLiveWindow(app, image_frame_storage, ContinousDataStorage)
autocollimator_live_window.win.show()

# Create and show the Straightness Measurement window
straightness_measurement_window = StraightnessMeasurementWindow(app, straightness_data, ContinousDataStorage)
straightness_measurement_window.win.show()


# Function to grab frames and process them
def grab_and_process():
    while True:
        image_frame_storage = camera.retrieve_frame()

        # Calculate the brightest pixel value
        # brightest_pixel_value = np.max(frame)
        # print(f"Brightest Pixel Value: {brightest_pixel_value}")

        # Process the frame using the ImageProcessor
        peaks_x, peaks_y = processor.process_frame(image_frame_storage)

        # Correct for linearity
        peaks_x, peaks_y = Corrector.correct(peaks_x, peaks_y)

        # Shove the data into the data storage
        ContinousDataStorage.new_data(peaks_x, peaks_y)


# Start a thread for grabbing and processing frames
thread = threading.Thread(target=grab_and_process)
thread.daemon = True
thread.start()

# Start the PyQtGraph application
app.exec_()

# Close the camera when the application is closed
camera.close()

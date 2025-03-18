import numpy
import numpy as np
from PyQt5 import QtWidgets
import threading
from image_aquisition import CameraManager
from image_processing import ImageProcessor
from win_live import AutocollimatorLiveWindowThread
from win_straightness import StraightnessMeasurementWindow
from data_storage import ContinousDataStorage
from data_storage import POIDataStorage
from calibration import Corrector
import warnings
import logging
import time
import cv2

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Constants for conversion from pixels to arcseconds
#logging.debug("calculating Constants for conversion from pixels to arcseconds")
PIXEL_PITCH = 3.45e-6  # in meters
FOCAL_LENGTH = 0.385  # in meters
CONVERSION_FACTOR = PIXEL_PITCH / (2 * FOCAL_LENGTH) * 180 / np.pi * 3600

# Init Frame Storage
image_frame_storage = []

# Initialize the camera
camera = CameraManager("AVI")

# Initialize the ImageProcessor instance
imagewidth, imageheight = camera.get_image_size()

processor = ImageProcessor("Linefit", imagewidth, imageheight)

# Initialize the corrector
#logging.debug("")
#target_x = []
#actual_x = []
#target_y = []
#actual_y = []
#Corrector = Corrector(target_x, actual_x, target_y, actual_y)

# Initialize the data storage
#logging.debug("Initialize the data storage")
ContinousStorage = ContinousDataStorage(CONVERSION_FACTOR)

# Initialize PyQtGraph application
#logging.debug("Initialize PyQtGraph application")
app = QtWidgets.QApplication([])

# Create and show the Autocollimator live window
#logging.debug("Create the Autocollimator live window")

# Initialize the data storage for straightness measurements
straightness_data = POIDataStorage(CONVERSION_FACTOR)

# Create and show the Straightness Measurement window
#logging.debug("Create and show the Straightness Measurement window")
#straightness_measurement_window = StraightnessMeasurementWindow(app, straightness_data, ContinousStorage)
#straightness_measurement_window.win.show()


# Function to grab frames and process them
def grab_and_process(stop_event):
    logging.debug("Starting grab_and_process thread")
    while not stop_event.is_set():
        try:
            # Retrieve frames from camera
            frame = camera.retrieve_frame()  # Assuming this returns a single frame
            #print(frame)
            image_frame_storage.clear()  # Clear the storage to keep only the latest frame
            #todo change this from a list to a single object thingy
            #frame = cv2.GaussianBlur(frame, (19, 19), 0)
            #frame[frame < 130] = 0
            image_frame_storage.append(frame)

            # Process frame
            frame_array = np.array(image_frame_storage)

            print(np.mean(frame_array))
            #if frame_array.ndim == 3:
            #    frame_array = frame_array.squeeze(axis=0)  # Ensure the correct shape

            peaks_x, peaks_y = processor.process_frame(frame_array)

            #print(peaks_x, peaks_y)

            # Store the data
            ContinousStorage.new_data(peaks_x, peaks_y)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            break
        # Calculate the brightest pixel value
        # brightest_pixel_value = np.max(frame)
        # print(f"Brightest Pixel Value: {brightest_pixel_value}")

# Create a stop event for the thread
stop_event = threading.Event()

# Start a thread for grabbing and processing frames
thread = threading.Thread(target=grab_and_process, args=(stop_event,))
thread.daemon = True
thread.start()

# Create and start the Autocollimator live window thread
live_window_thread = AutocollimatorLiveWindowThread(image_frame_storage, ContinousStorage)
live_window_thread.run()

# Start the PyQtGraph application
app.exec_()

# Close the camera when the application is closed
camera.close()

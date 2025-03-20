import numpy as np
from PyQt5 import QtWidgets
import threading
import queue
from image_aquisition import CameraManager
from image_processing import ImageProcessor
from win_live import AutocollimatorLiveWindowThread
from data_storage import ContinousDataStorage
from data_storage import POIDataStorage
import logging

# Configure logging with location information
logging.basicConfig(
    level=logging.DEBUG,  # Set your desired log level
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    datefmt='%H:%M:%S'
)

# Constants for conversion from pixels to arcseconds
PIXEL_PITCH = 3.45e-6  # in meters
FOCAL_LENGTH = 0.385  # in meters
CONVERSION_FACTOR = PIXEL_PITCH / (2 * FOCAL_LENGTH) * 180 / np.pi * 3600

# Thread-safe frame management
class FrameManager:
    def __init__(self):
        self.current_frame = None
        self.lock = threading.Lock()

    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame

    def get_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

# Initialize components
frame_manager = FrameManager()
frame_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent memory issues
camera = CameraManager("AVI")
imagewidth, imageheight = camera.get_image_size()
processor = ImageProcessor("AccurateGaussian", imagewidth, imageheight)
ContinousStorage = ContinousDataStorage(CONVERSION_FACTOR)
straightness_data = POIDataStorage(CONVERSION_FACTOR)
app = QtWidgets.QApplication([])

# Producer: captures frames from camera
def frame_producer(stop_event):
    logging.debug("Starting frame producer thread")
    while not stop_event.is_set():
        try:
            frame = camera.retrieve_frame()
            frame_manager.update_frame(frame)  # Update display frame

            # Add to processing queue, non-blocking
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                # Skip frame if queue is full
                #logging.debug("Processing queue full, skipping frame")
                pass

        except Exception as e:
            logging.error(f"Producer error: {e}")
            break

# Consumer: processes frames
def frame_consumer(stop_event):
    logging.debug("Starting frame consumer thread")
    while not stop_event.is_set():
        logging.debug("Consumer thread running")
        try:
            # Get frame with timeout to check stop_event periodically
            frame = frame_queue.get(timeout=0.5)

            # Process frame
            if frame is not None:
                #print(frame[0][0])
                peaks_x, peaks_y = processor.process_frame(frame)
                ContinousStorage.new_data(peaks_x, peaks_y)


            frame_queue.task_done()
        except queue.Empty:
            # No new frames, continue checking stop_event
            continue
        except Exception as e:
            logging.error(f"Consumer error: {e}")
            continue

# Create stop events
stop_event = threading.Event()

# Start producer and consumer threads
producer_thread = threading.Thread(target=frame_producer, args=(stop_event,))
consumer_thread = threading.Thread(target=frame_consumer, args=(stop_event,))
producer_thread.daemon = True
consumer_thread.daemon = True
producer_thread.start()
consumer_thread.start()

# Create and start the Autocollimator live window thread
# Updated to use frame_manager instead of image_frame_storage list
live_window_thread = AutocollimatorLiveWindowThread(frame_manager, ContinousStorage)
live_window_thread.run()

try:
    # Start the PyQtGraph application
    app.exec_()
finally:
    # Clean up
    stop_event.set()
    producer_thread.join(timeout=1.0)
    consumer_thread.join(timeout=1.0)
    camera.close()
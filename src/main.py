import numpy as np
from PyQt5 import QtWidgets
import threading
import queue
from image_aquisition import CameraManager
from image_filter import ImageFilter
from image_processing import ImageProcessor
from win_live import AutocollimatorLiveWindowThread
from data_storage import ContinousDataStorage
from data_storage import POIDataStorage
import logging
import time

# Configure logging with location information
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce logging overhead
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    datefmt='%H:%M:%S'
)

# Constants for conversion from pixels to arcseconds
PIXEL_PITCH = 3.45e-6  # in meters
FOCAL_LENGTH = 0.385  # in meters
CONVERSION_FACTOR = PIXEL_PITCH / (2 * FOCAL_LENGTH) * 180 / np.pi * 3600
print(f"Conversion factor: {CONVERSION_FACTOR} arcseconds/pixel")

# Improved thread-safe frame management with minimal locking
class FrameManager:
    def __init__(self):
        self.current_frame = None
        self.lock = threading.RLock()  # Reentrant lock for better performance

    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame

    def get_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame  # Return direct reference for display to avoid copy

# Initialize components
frame_manager = FrameManager()
# Increased queue size for better throughput
frame_queue = queue.Queue(maxsize=10)
#camera = CameraManager("AVI", video_path="C:/Users/Leo/Documents/GitHub/Autocollimator/test/Videos/6.avi")
camera = CameraManager("Basler", ExposureMode='Standard', Exposuretime=10000, gamma=1.00) #UltraShort / Standard
imagewidth, imageheight = camera.get_image_size()
filter1 = ImageFilter("Background", method="fixed", value=150)
#processor = ImageProcessor("WeightedPeakfinder", imagewidth, imageheight, window_size=50, distance=25, prominence=5000)
processor = ImageProcessor("PeakfinderQuadratic", imagewidth, imageheight, width=7, degree=2) #13 ist gut
ContinousStorage = ContinousDataStorage(CONVERSION_FACTOR)
#straightness_data = POIDataStorage(CONVERSION_FACTOR)
app = QtWidgets.QApplication([])

# Frame buffer pool to avoid frequent memory allocations
class FrameBufferPool:
    def __init__(self, shape, dtype=np.uint8, pool_size=5):
        self.buffers = [np.zeros(shape, dtype=dtype) for _ in range(pool_size)]
        self.lock = threading.Lock()
        
    def get_buffer(self):
        with self.lock:
            if not self.buffers:
                # If no buffers are available, create a new one (should be rare)
                return np.zeros(camera.get_image_size()[::-1] + (3,), dtype=np.uint8)
            return self.buffers.pop()
    
    def return_buffer(self, buf):
        with self.lock:
            self.buffers.append(buf)

# Create a buffer pool based on camera frame size (assuming BGR format)
buffer_pool = FrameBufferPool((imageheight, imagewidth, 3))

# Producer: optimized to capture frames with minimal overhead
def frame_producer(stop_event):
    logging.debug("Starting frame producer thread")
    skip_counter = 0
    last_time = time.time()
    
    while not stop_event.is_set():
        try:
            current_time = time.time()
            frame_time = current_time - last_time
            last_time = current_time
            
            # Get a frame buffer from the pool
            frame = camera.retrieve_frame()
            frame = filter1.apply(frame)  # Apply background filter
            
            if frame is None:
                time.sleep(0.001)  # Short sleep to avoid CPU spinning
                continue
                
            frame_manager.update_frame(frame)  # Update display frame

            # Dynamic frame skipping based on processing backlog
            if frame_queue.qsize() < frame_queue.maxsize * 0.8:
                # Add to processing queue, non-blocking
                try:
                    frame_queue.put(frame, block=False)
                    skip_counter = 0
                except queue.Full:
                    skip_counter += 1
            else:
                skip_counter += 1
                # Skip frames when system is under load
                if skip_counter % 3 != 0:  # Process every 3rd frame under load
                    continue
                    
        except Exception as e:
            logging.error(f"Producer error: {e}")
            time.sleep(0.01)  # Avoid busy-waiting on error

# Consumer: optimized processing with minimal contention
def frame_consumer(stop_event):
    logging.debug("Starting frame consumer thread")
    processing_times = []
    
    while not stop_event.is_set():
        try:
            # Get frame with timeout to check stop_event periodically
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Process frame
            if frame is not None:
                peaks_x, peaks_y = processor.process_frame(frame)
                ContinousStorage.new_data(peaks_x, peaks_y)

            frame_queue.task_done()
            
        except Exception as e:
            logging.error(f"Consumer error: {e}")

# Create stop events
stop_event = threading.Event()

# Start producer and consumer threads
producer_thread = threading.Thread(target=frame_producer, args=(stop_event,), name="ProducerThread")
consumer_thread = threading.Thread(target=frame_consumer, args=(stop_event,), name="ConsumerThread")
producer_thread.daemon = True
consumer_thread.daemon = True
producer_thread.start()
consumer_thread.start()

# Create and start the Autocollimator live window thread
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

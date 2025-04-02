from pypylon import pylon
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import cv2
import time
import threading
from collections import deque
from queue import Queue, Empty

class CameraManager:
    """
    CameraManager class to manage the available camera types
    camera_type: str
        The type of camera to use. Must be either 'Basler', 'USB', or 'AVI'
    """
    def __init__(self, camera_type='Basler', buffer_size=10, threaded=True, **kwargs) -> None:
        self.camera = None
        self.buffer_size = buffer_size
        self.threaded = threaded
        self.frame_buffer = deque(maxlen=buffer_size)
        self.frame_counter = 0
        self.last_frame = None
        self.lock = threading.RLock()
        self._image_size = None
        self._running = False
        self._thread = None

        if camera_type == "Basler":
            self.camera = BaslerCamera(**kwargs)
        elif camera_type == "USB":
            raise Exception("camera_type USB is not implemented yet")
        elif camera_type == "AVI":
            self.camera = AVIReader(**kwargs)
        else:
            raise ValueError("camera_type does not exist")
            
        # Cache the image size
        self._image_size = self.camera.get_image_size()
        
        # Start the background thread if requested
        if self.threaded:
            self._start_thread()

    def _start_thread(self):
        """Start background thread for continuous frame grabbing"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._thread.start()

    def _grab_loop(self):
        """Background thread function to continuously grab frames"""
        while self._running:
            try:
                # Get frame from camera
                frame = self.camera.retrieve_frame()
                
                # Convert to grayscale if needed (using faster method)
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Add to buffer
                with self.lock:
                    self.frame_buffer.append(frame)
                    self.last_frame = frame
                    self.frame_counter += 1
                
                # Small sleep to avoid CPU overload
                time.sleep(0.001)
            except Exception:
                # Just continue on error, don't raise
                time.sleep(0.01)
                continue

    def get_image_size(self) -> tuple:
        """
        Get the image size of the camera (cached for performance)
        :return: tuple (width, height)
        """
        return self._image_size

    def retrieve_frame(self) -> np.ndarray:
        """
        Retrieve a frame from the camera - optimized with buffer
        :return: np.ndarray grayscale image
        """
        if self.threaded:
            with self.lock:
                if self.last_frame is not None:
                    return self.last_frame.copy()
                elif len(self.frame_buffer) > 0:
                    return self.frame_buffer[-1].copy()
        
        # Fallback to direct capture if not threaded or buffer empty
        try:
            frame = self.camera.retrieve_frame()
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        except Exception as e:
            # Return last valid frame on error if available
            if self.last_frame is not None:
                return self.last_frame.copy()
            raise Exception(f"Failed to retrieve frame: {str(e)}")

    def get_latest_frames(self, n=1):
        """
        Get the n latest frames from buffer
        :param n: number of frames to retrieve
        :return: list of frames
        """
        with self.lock:
            return list(self.frame_buffer)[-n:] if len(self.frame_buffer) > 0 else []

    def close(self) -> None:
        """
        Close the camera and stop background thread
        """
        # Stop background thread if running
        if self.threaded and self._running:
            self._running = False
            if self._thread:
                self._thread.join(timeout=1.0)
        
        # Close camera
        if self.camera is not None:
            self.camera.close()
        else:
            raise Exception("No camera initialized")

class BaslerCamera:
    """
    BaslerCamera class to manage the Basler Ace 2 camera
    """
    def __init__(self, Width=1936, Height=1216, PixelFormat='Mono8', ExposureMode='Standard', Exposuretime=None) -> None:
        """
        Initialize the Basler Ace 2 camera
        Set all setting to smart values
        """
        # Create an instant camera object with the camera device found first.
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Open the camera
        self.camera.Open()
        self.camera.PixelFormat.SetValue(PixelFormat)
        self.camera.Width.SetValue(Width)
        self.camera.Height.SetValue(Height)
        self.camera.BslExposureTimeMode.SetValue(ExposureMode) #UltraShort / Standard
        if Exposuretime is not None:
            self.camera.ExposureTime.SetValue(Exposuretime)

        # Start grabbing images
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def get_image_size(self):
        """
        Get the image size of the camera
        """
        return self.camera.Width.GetValue(), self.camera.Height.GetValue()

    def retrieve_frame(self):
        """
        Retrieve a frame from the camera
        """
        # Retrieve a frame from the camera
        if self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                # Access the image data
                image = grab_result.Array
                grab_result.Release()
                return image
            else:
                grab_result.Release()
                raise Exception("Failed to grab image")
        else:
            raise Exception("Camera is not grabbing")

    def close(self):
        """
        Close the camera
        """
        # Stop grabbing images
        self.camera.StopGrabbing()
        # Close the camera
        self.camera.Close()

class AVIReader:
    """
    AVIReader class to read frames from an AVI file, optimized for maximum speed
    """
    def __init__(self, video_path: str):
        """
        Initialize the AVIReader with a video file
        :param video_path: str
        """
        self.video_path = video_path

        # Try to use hardware acceleration via FFMPEG backend
        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            # Fallback to default backend if FFMPEG is not available
            self.cap = cv2.VideoCapture(self.video_path)
            print("Falling back to default backend for video capture.")
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

        # Cache image dimensions for faster access
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set optimal buffer size in OpenCV
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

        # Determine if video is color to optimize conversion
        ret, test_frame = self.cap.read()
        if ret:
            self.is_color = len(test_frame.shape) == 3
            # Reset position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            raise ValueError(f"Cannot read frames from video: {video_path}")

    def retrieve_frame(self):
        ret, frame = self.cap.read()
        #time.sleep(0.02)
        if not ret:
            # Reset to beginning if end of video is reached
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                # Return black frame if still can't read
                return np.zeros((self.height, self.width), dtype=np.uint8)

        # Process frame if it's color (faster than cv2.cvtColor)
        if self.is_color:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def get_image_size(self):
        # Return cached dimensions
        return self.width, self.height

    def close(self):
        self.cap.release()

def testing():
    # Initialize CameraManager with "Basler"
    camera_manager = CameraManager("AVI")

    # Initialize PyQtGraph application
    app = QtWidgets.QApplication([])

    # Create a window with a layout
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Camera Stream")
    central_widget = QtWidgets.QWidget()
    win.setCentralWidget(central_widget)
    layout = QtWidgets.QVBoxLayout()
    central_widget.setLayout(layout)

    # Create a plot widget for the camera stream
    plot_widget = pg.ImageView()
    layout.addWidget(plot_widget)

    # Show the window
    win.show()

    def update():
        # Retrieve a frame
        frame = camera_manager.retrieve_frame()

        # Update the plot with the new frame
        plot_widget.setImage(frame.T)

    # Set up a timer to update the plot periodically
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)  # Update every 50 ms

    # Start the PyQtGraph application
    app.exec_()

    # Close the camera when the application is closed
    camera_manager.close()

def captureIntoVideo():
    # Captures the video from the camera and saves it to a file
    # Initialize CameraManager with "Basler"
    camera_manager = CameraManager("Basler")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 40.0, (1936, 1216), isColor=False)
    start_time = time.time()

    try:
        # Capture the video
        while True:
            frame = camera_manager.retrieve_frame()
            # Convert frame to 8-bit grayscale if necessary
            if frame.dtype != np.uint8:
                frame = cv2.convertScaleAbs(frame)
            out.write(frame)
            # Stop recording after x seconds
            if time.time() - start_time > 20:
                break
    except KeyboardInterrupt:
        print("Video capture interrupted. Cleaning up...")
    finally:
        # Add a delay to ensure all frames are written
        time.sleep(1)
        # Close the camera and release the video writer
        camera_manager.close()
        out.release()

if __name__ == "__main__":
    #testing()
    captureIntoVideo()

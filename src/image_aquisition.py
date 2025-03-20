from pypylon import pylon
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import cv2
import time
import sys
#rom os.path import dirname, abspath
#from os.path import join

# Add the project root directory to the Python path
#project_root = dirname(dirname(abspath(__file__)))
#sys.path.append(project_root)

class CameraManager:
    """
    CameraManager class to manage the available camera types
    camera_type: str
        The type of camera to use. Must be either 'Basler' or 'USB'
    """
    def __init__(self, camera_type: str) -> None:
        if camera_type not in ["Basler", "USB", "AVI"]:
            raise ValueError("camera_type must be either 'Basler', 'USB' or 'AVI'")
        self.camera_type = camera_type
        self.camera = None

        if camera_type == "Basler":
            self.camera = BaslerCamera()

        if camera_type == "USB":
            raise Exception("camera_type USB is not implemented yet")

        if camera_type == "AVI":
            self.camera = AVIReader("src/20250106_10um_crosshair_trimmed.avi")

    def get_image_size(self) -> tuple:
        """
        Get the image size of the camera
        :return: tuple
        """
        if self.camera is not None:
            return self.camera.get_image_size()
        else:
            raise Exception("No camera initialized")

    def retrieve_frame(self) -> np.ndarray:
        """
        Retrieve a frame from the camera
        collapes RGB to grayscale if necessary
        :return: np.ndarray
        """
        if self.camera is not None:
            frame = self.camera.retrieve_frame()
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame

            #if len(frame.shape) == 3 and frame.shape[2] == 3:
            #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #elif len(frame.shape) == 2:
            #    gray = frame
            #else:
            #    raise ValueError("Unexpected number of channels in the input image")
        else:
            raise Exception("No camera initialized")

    def close(self) -> None:
        """
        Close the camera
        """
        if self.camera is not None:
            self.camera.close()
        else:
            raise Exception("No camera initialized")

class BaslerCamera:
    """
    BaslerCamera class to manage the Basler Ace 2 camera
    """
    def __init__(self):
        """
        Initialize the Basler Ace 2 camera
        Set all setting to smart values
        """
        # Create an instant camera object with the camera device found first.
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Open the camera
        self.camera.Open()

        # Set the pixel format
        self.camera.PixelFormat.SetValue('Mono8')

        # Set the width and height
        self.camera.Width.SetValue(1936)
        self.camera.Height.SetValue(1216)

        # Set the exposure time mode
        self.camera.BslExposureTimeMode.SetValue('Standard') #UltraShort / Standard

        # Set the exposure time
        #self.camera.ExposureTime.SetValue(20000.0)

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
    AVIReader class to read frames from an AVI file
    """
    def __init__(self, video_path):
        self.video_path = "20250106_10um_crosshair.avi"
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1.0 / self.fps

    def retrieve_frame(self):
        ret, frame = self.cap.read()
        #time.sleep(0.02)
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
            ret, frame = self.cap.read()
        return frame

    def get_image_size(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
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
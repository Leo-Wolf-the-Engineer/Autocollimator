from pypylon import pylon
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import time

class CameraManager:
    """
    CameraManager class to manage the available camera types
    camera_type: str
        The type of camera to use. Must be either 'Basler' or 'USB'
    """
    def __init__(self, camera_type: str) -> None:
        if camera_type not in ["Basler", "USB"]:
            raise ValueError("camera_type must be either 'Basler' or 'USB'")
        self.camera_type = camera_type
        self.camera = None

        if camera_type == "Basler":
            self.camera = BaslerCamera()

        if camera_type == "USB":
            raise Exception("camera_type USB is not implemented yet")

    def retrieve_frame(self) -> np.ndarray:
        """
        Retrieve a frame from the camera
        :return: np.ndarray
        """
        if self.camera is not None:
            return self.camera.retrieve_frame()
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

        # Set the pixel format to Mono12p
        self.camera.PixelFormat.SetValue('Mono12p')

        # Set the width and height
        self.camera.Width.SetValue(1936)
        self.camera.Height.SetValue(1216)

        # Set the exposure time mode to UltraShort
        self.camera.BslExposureTimeMode.SetValue('UltraShort')

        # Set the exposure time to 5.0 microseconds
        self.camera.ExposureTime.SetValue(5.0)

        # Start grabbing images
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

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

def testing():
    # Initialize CameraManager with "Basler"
    camera_manager = CameraManager("Basler")

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

if __name__ == "__main__":
    testing()
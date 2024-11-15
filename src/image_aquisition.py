from pypylon import pylon
from typing import Union

class CameraManager:
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
        if self.camera is not None:
            return self.camera.retrieve_frame()
        else:
            raise Exception("No camera initialized")

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
        else:
            raise Exception("No camera initialized")
class BaslerCamera:
    def __init__(self):
        # Create an instant camera object with the camera device found first.
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.configure_camera()

        # Open the camera
        self.camera.Open()

        # Set the pixel format to Mono12p
        self.camera.PixelFormat.SetValue('Mono12p')

        # Set the width and height
        self.camera.Width.SetValue(1936)
        self.camera.Height.SetValue(1236)

        # Set the exposure time mode to UltraShort
        self.camera.BslExposureTimeMode.SetValue('UltraShort')

        # Set the exposure time to 5.0 microseconds
        self.camera.ExposureTime.SetValue(5.0)

        # Start grabbing images
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def retrieve_frame(self):
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
        # Stop grabbing images
        self.camera.StopGrabbing()
        # Close the camera
        self.camera.Close()
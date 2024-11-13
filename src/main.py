import cv2
import numpy as np
from pypylon import pylon
from scipy.optimize import curve_fit
from utils import plot_current_frame, initialize_plot


# Function to define the Gaussian curve
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


# Connect to the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Configure the camera (removed)
# camera.PixelFormat = "BGR8"
# camera.ExposureTimeAbs.value = 20.0  # Set exposure time to 20 microseconds
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Determine frame size
grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
frame = grabResult.Array
height, width, _ = frame.shape
grabResult.Release()

# Initialize plot
fig, axs = initialize_plot()

# Lists to store peak positions
peak_x_history = []
peak_y_history = []

# Main loop to process frames
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        frame = grabResult.Array

        # Sum the intensities of all RGB channels
        intensity_x = np.sum(frame, axis=(0, 2))
        intensity_y = np.sum(frame, axis=(1, 2))

        # Fit in the X direction
        x = np.arange(width)
        popt_x, _ = curve_fit(gaussian, x, intensity_x, p0=[np.max(intensity_x), np.argmax(intensity_x), 10])

        # Fit in the Y direction
        y = np.arange(height)
        popt_y, _ = curve_fit(gaussian, y, intensity_y, p0=[np.max(intensity_y), np.argmax(intensity_y), 10])

        # Output the determined values
        print(f"X Direction: Amplitude={popt_x[0]:.2f}, Mean={popt_x[1]:.2f}, Stddev={popt_x[2]:.2f}")
        print(f"Y Direction: Amplitude={popt_y[0]:.2f}, Mean={popt_y[1]:.2f}, Stddev={popt_y[2]:.2f}")

        # Store peak positions
        peak_x_history.append(popt_x[1])
        peak_y_history.append(popt_y[1])

        # Plot the current frame and intensity distributions
        plot_current_frame(axs, frame, intensity_x, intensity_y, popt_x[1], popt_y[1], peak_x_history, peak_y_history)

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
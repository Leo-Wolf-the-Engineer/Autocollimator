import numpy as np
import cv2
from pypylon import pylon
from scipy.optimize import curve_fit
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# Constants for conversion
PIXEL_PITCH = 1.45e-6  # in meters
FOCAL_LENGTH = 0.385  # in meters
CONVERSION_FACTOR = PIXEL_PITCH / (2 * FOCAL_LENGTH) * 180 / np.pi * 3600  # conversion factor from pixels to arcseconds


# Function to define the Gaussian curve
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


# Initialize the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Determine frame size
grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
frame = grabResult.Array
height, width, _ = frame.shape
grabResult.Release()

# Initialize PyQtGraph application
app = QtWidgets.QApplication([])

# Create a window with plots
win = pg.GraphicsLayoutWidget(show=True, title="Autocollimator Real-time Plotting")
win.resize(1800, 600)

# Create plots
plot_frame = win.addPlot(title="Current Frame")
img_item = pg.ImageItem()
plot_frame.addItem(img_item)

win.nextRow()

plot_intensity_x = win.addPlot(title="Intensity Distribution in X Direction")
curve_intensity_x = plot_intensity_x.plot(pen='y')
peak_line_x = plot_intensity_x.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))

plot_intensity_y = win.addPlot(title="Intensity Distribution in Y Direction")
curve_intensity_y = plot_intensity_y.plot(pen='y')
peak_line_y = plot_intensity_y.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))

win.nextRow()

plot_peak_x = win.addPlot(title="Peak Position in X Direction")
curve_peak_x = plot_peak_x.plot(pen='y')

plot_peak_y = win.addPlot(title="Peak Position in Y Direction")
curve_peak_y = plot_peak_y.plot(pen='y')

# Initialize data storage
peak_x_history = []
peak_y_history = []


# Update function
def update():
    global frame, peak_x_history, peak_y_history

    if camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            frame = grabResult.Array

            # Extract blue channel
            blue_channel = frame[:, :, 0]

            # Sum the intensities of the blue channel
            intensity_x = np.sum(blue_channel, axis=0)
            intensity_y = np.sum(blue_channel, axis=1)

            # Fit Gaussian in the X direction
            x = np.arange(width)
            popt_x, _ = curve_fit(gaussian, x, intensity_x, p0=[np.max(intensity_x), np.argmax(intensity_x), 10])

            # Fit Gaussian in the Y direction
            y = np.arange(height)
            popt_y, _ = curve_fit(gaussian, y, intensity_y, p0=[np.max(intensity_y), np.argmax(intensity_y), 10])

            # Convert peak positions from pixels to arcseconds
            peak_x_arcsec = popt_x[1] * CONVERSION_FACTOR
            peak_y_arcsec = popt_y[1] * CONVERSION_FACTOR

            # Print the determined values
            print(f"X Direction: Amplitude={popt_x[0]:.2f}, Mean={popt_x[1]:.2f} pixels, Stddev={popt_x[2]:.2f}")
            print(f"X Direction: Mean={peak_x_arcsec:.2f} arcseconds")
            print(f"Y Direction: Amplitude={popt_y[0]:.2f}, Mean={popt_y[1]:.2f} pixels, Stddev={popt_y[2]:.2f}")
            print(f"Y Direction: Mean={peak_y_arcsec:.2f} arcseconds")

            # Store peak positions
            peak_x_history.append(peak_x_arcsec)
            peak_y_history.append(peak_y_arcsec)

            # Update the plots
            img_item.setImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(1, 0, 2))
            curve_intensity_x.setData(intensity_x)
            peak_line_x.setValue(popt_x[1])
            curve_intensity_y.setData(intensity_y)
            peak_line_y.setValue(popt_y[1])
            curve_peak_x.setData(peak_x_history)
            curve_peak_y.setData(peak_y_history)

        grabResult.Release()


# Set up a timer to call the update function every 5 milliseconds
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)  # 5 milliseconds

# Start the Qt event loop
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
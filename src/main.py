import numpy as np
import cv2
from pypylon import pylon
from scipy.optimize import curve_fit
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import threading
import time

# Constants for conversion
PIXEL_PITCH = 3.45e-6  # in meters
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

# Create a window with a layout
win = QtWidgets.QMainWindow()
win.setWindowTitle("Autocollimator live")
central_widget = QtWidgets.QWidget()
win.setCentralWidget(central_widget)
main_layout = QtWidgets.QHBoxLayout()
central_widget.setLayout(main_layout)

# Create a vertical layout for the left side (frame and intensity distributions)
left_layout = QtWidgets.QVBoxLayout()
main_layout.addLayout(left_layout)

# Create a plot widget for the current frame
plot_frame = pg.PlotWidget(title="Current Frame")
plot_frame.setBackground('k')
img_item = pg.ImageItem()
plot_frame.addItem(img_item)
left_layout.addWidget(plot_frame)

# Create a plot widget for intensity distribution in X direction
plot_intensity_x = pg.PlotWidget(title="Intensity Distribution in X Direction")
plot_intensity_x.setBackground('k')
curve_intensity_x = plot_intensity_x.plot(pen='y')
peak_line_x = plot_intensity_x.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
left_layout.addWidget(plot_intensity_x)

# Create a plot widget for intensity distribution in Y direction
plot_intensity_y = pg.PlotWidget(title="Intensity Distribution in Y Direction")
plot_intensity_y.setBackground('k')
curve_intensity_y = plot_intensity_y.plot(pen='y')
peak_line_y = plot_intensity_y.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
left_layout.addWidget(plot_intensity_y)

# Create a vertical layout for the right side (peak positions and buttons)
right_layout = QtWidgets.QVBoxLayout()
main_layout.addLayout(right_layout)

# Create a plot widget for peak position in X direction
plot_peak_x = pg.PlotWidget(title="Peak Position in X Direction")
plot_peak_x.setBackground('k')
curve_peak_x = plot_peak_x.plot(pen='y')
right_layout.addWidget(plot_peak_x)

# Create a plot widget for peak position in Y direction
plot_peak_y = pg.PlotWidget(title="Peak Position in Y Direction")
plot_peak_y.setBackground('k')
curve_peak_y = plot_peak_y.plot(pen='y')
right_layout.addWidget(plot_peak_y)

# Create a horizontal layout for the buttons
button_layout = QtWidgets.QHBoxLayout()
right_layout.addLayout(button_layout)

# Create a button to zero peak position values and reset the X plot
button_reset_x = QtWidgets.QPushButton("Reset X Peak Position")
button_layout.addWidget(button_reset_x)

# Create a button to zero peak position values and reset the Y plot
button_reset_y = QtWidgets.QPushButton("Reset Y Peak Position")
button_layout.addWidget(button_reset_y)

# Create a button to start averaging measurements
button_average = QtWidgets.QPushButton("Start Averaging")
button_layout.addWidget(button_average)

# Create a text box to display averaged values
average_display = QtWidgets.QLabel("Averaged Values: X = 0.0, Y = 0.0")
right_layout.addWidget(average_display)

# Initialize data storage
peak_x_history = []
peak_y_history = []
zero_x = 0
zero_y = 0
averaging = False
average_start_time = 0
average_x_values = []
average_y_values = []

# Variables to store the latest frame and peaks
latest_frame = None
latest_peak_x = 0
latest_peak_y = 0


# Function to grab frames and process them
def grab_and_process():
    global latest_frame, latest_peak_x, latest_peak_y, peak_x_history, peak_y_history, zero_x, zero_y, averaging, average_start_time, average_x_values, average_y_values

    while camera.IsGrabbing():
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
            try:
                popt_x, _ = curve_fit(gaussian, x, intensity_x, p0=[np.max(intensity_x), np.argmax(intensity_x), 10])
                peak_x_arcsec = (popt_x[1] - zero_x) * CONVERSION_FACTOR
            except RuntimeError:
                peak_x_arcsec = np.nan

            # Fit Gaussian in the Y direction
            y = np.arange(height)
            try:
                popt_y, _ = curve_fit(gaussian, y, intensity_y, p0=[np.max(intensity_y), np.argmax(intensity_y), 10])
                peak_y_arcsec = (popt_y[1] - zero_y) * CONVERSION_FACTOR
            except RuntimeError:
                peak_y_arcsec = np.nan

            # Print the determined values
            print(f"X Direction: Mean={peak_x_arcsec:.2f} arcseconds")
            print(f"Y Direction: Mean={peak_y_arcsec:.2f} arcseconds")

            # Store peak positions
            peak_x_history.append(peak_x_arcsec)
            peak_y_history.append(peak_y_arcsec)

            # Update the latest frame and peaks
            latest_frame = frame
            latest_peak_x = popt_x[1] if not np.isnan(peak_x_arcsec) else latest_peak_x
            latest_peak_y = popt_y[1] if not np.isnan(peak_y_arcsec) else latest_peak_y

            # Handle averaging
            if averaging:
                current_time = time.time()
                if current_time - average_start_time <= 3:
                    if not np.isnan(peak_x_arcsec):
                        average_x_values.append(peak_x_arcsec)
                    if not np.isnan(peak_y_arcsec):
                        average_y_values.append(peak_y_arcsec)
                else:
                    averaging = False
                    avg_x = np.mean(average_x_values) if average_x_values else 0
                    avg_y = np.mean(average_y_values) if average_y_values else 0
                    average_display.setText(f"Averaged Values: X = {avg_x:.2f}, Y = {avg_y:.2f}")

        grabResult.Release()
        time.sleep(0.005)  # Sleep for 5 milliseconds


# Update function to update the plots
def update_plots():
    global latest_frame, latest_peak_x, latest_peak_y, peak_x_history, peak_y_history

    if latest_frame is not None:
        img_item.setImage(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB).transpose(1, 0, 2))
        curve_intensity_x.setData(np.sum(latest_frame[:, :, 0], axis=0))
        peak_line_x.setValue(latest_peak_x)
        curve_intensity_y.setData(np.sum(latest_frame[:, :, 0], axis=1))
        peak_line_y.setValue(latest_peak_y)
        curve_peak_x.setData(peak_x_history)
        curve_peak_y.setData(peak_y_history)


# Reset function for X peak position plot
def reset_x_peak_position():
    global peak_x_history, zero_x, latest_peak_x
    zero_x = latest_peak_x
    peak_x_history = []
    curve_peak_x.setData(peak_x_history)
    curve_intensity_x.setData(np.zeros(width))


# Reset function for Y peak position plot
def reset_y_peak_position():
    global peak_y_history, zero_y, latest_peak_y
    zero_y = latest_peak_y
    peak_y_history = []
    curve_peak_y.setData(peak_y_history)
    curve_intensity_y.setData(np.zeros(height))


# Start averaging measurements
def start_averaging():
    global averaging, average_start_time, average_x_values, average_y_values
    averaging = True
    average_start_time = time.time()
    average_x_values = []
    average_y_values = []


# Connect buttons to their respective functions
button_reset_x.clicked.connect(reset_x_peak_position)
button_reset_y.clicked.connect(reset_y_peak_position)
button_average.clicked.connect(start_averaging)

# Start a thread for grabbing and processing frames
thread = threading.Thread(target=grab_and_process)
thread.daemon = True
thread.start()

# Set up a timer to call the update function every 50 milliseconds
timer = QtCore.QTimer()
timer.timeout.connect(update_plots)
timer.start(50)  # 50 milliseconds

# Show the window
win.show()

# Start the Qt event loop
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
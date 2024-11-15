import numpy as np
import cv2
from pypylon import pylon
from scipy.optimize import curve_fit
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QScreen
from pyqtgraph.exporters import ImageExporter
import threading
import time
import utils
from datetime import datetime

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
height, width = frame.shape
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
plot_intensity_x.setLabel('left', 'Intensity')
plot_intensity_x.setLabel('bottom', 'Pixel Position')
left_layout.addWidget(plot_intensity_x)

# Create a plot widget for intensity distribution in Y direction
plot_intensity_y = pg.PlotWidget(title="Intensity Distribution in Y Direction")
plot_intensity_y.setBackground('k')
curve_intensity_y = plot_intensity_y.plot(pen='y')
peak_line_y = plot_intensity_y.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
plot_intensity_y.setLabel('left', 'Intensity')
plot_intensity_y.setLabel('bottom', 'Pixel Position')
left_layout.addWidget(plot_intensity_y)

# Create a vertical layout for the right side (peak positions and buttons)
right_layout = QtWidgets.QVBoxLayout()
main_layout.addLayout(right_layout)

# Create a plot widget for peak position in X direction
plot_peak_x = pg.PlotWidget(title="Peak Position in X Direction")
plot_peak_x.setBackground('k')
curve_peak_x = plot_peak_x.plot(pen='y')
plot_peak_x.setLabel('left', 'Peak Position (arcseconds)')
plot_peak_x.setLabel('bottom', 'Time (minutes)')
right_layout.addWidget(plot_peak_x)

# Create a plot widget for peak position in Y direction
plot_peak_y = pg.PlotWidget(title="Peak Position in Y Direction")
plot_peak_y.setBackground('k')
curve_peak_y = plot_peak_y.plot(pen='y')
plot_peak_y.setLabel('left', 'Peak Position (arcseconds)')
plot_peak_y.setLabel('bottom', 'Time (minutes)')
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
button_average = QtWidgets.QPushButton("Take Average")
button_layout.addWidget(button_average)

# Create a text box to display averaged values
average_display = QtWidgets.QLabel("Averaged Values: X = 0.0, Y = 0.0")
right_layout.addWidget(average_display)

# Create a box to show how many frames per second are received
fps_display = QtWidgets.QLabel("FPS: 0")
right_layout.addWidget(fps_display)

# Create a button to save the whole window as an image
button_save_image_1 = QtWidgets.QPushButton("Save Window as Image")
button_layout.addWidget(button_save_image_1)
button_save_image_1.clicked.connect(lambda: utils.save_window_as_image(win,app))

# Initialize data storage
peak_x_history = []
peak_y_history = []
zero_x = 0
zero_y = 0
averaging = False
average_start_time = time.time_ns()
average_x_values = []
average_y_values = []

# Variables to store the latest frame and peaks
latest_frame = None
latest_peak_x = 0
latest_peak_y = 0

# Variables for FPS calculation
frame_count = 0
start_time = time.time()

# Function to grab frames and process them
def grab_and_process():
    global latest_frame, latest_peak_x, latest_peak_y, peak_x_history, peak_y_history, zero_x, zero_y, averaging, average_start_time, average_x_values, average_y_values, frame_count

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            frame = grabResult.Array

            # Sum the intensities of the grayscale image
            intensity_x = np.sum(frame, axis=0)
            intensity_y = np.sum(frame, axis=1)

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
            peak_x_history.append(((time.time_ns()- average_start_time)/6e10, peak_x_arcsec))
            peak_y_history.append(((time.time_ns()- average_start_time)/6e10, peak_y_arcsec))

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

            # Update frame count for FPS calculation
            frame_count += 1

        grabResult.Release()
        time.sleep(0.005)  # Sleep for 5 milliseconds to handle high frame rate

# Update function to update the plots
def update_plots():
    global latest_frame, latest_peak_x, latest_peak_y, peak_x_history, peak_y_history, frame_count, start_time

    if latest_frame is not None:
        img_item.setImage(latest_frame.T)
        curve_intensity_x.setData(np.sum(latest_frame, axis=0))
        peak_line_x.setValue(latest_peak_x)
        curve_intensity_y.setData(np.sum(latest_frame, axis=1))
        peak_line_y.setValue(latest_peak_y)
        curve_peak_x.setData(*zip(*peak_x_history))
        curve_peak_y.setData(*zip(*peak_y_history))

    # Update FPS display
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 3:
        fps = frame_count / elapsed_time
        fps_display.setText(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = current_time
        print(frame_count)

# Reset function for X peak position plot
def reset_x_peak_position():
    global peak_x_history, zero_x, latest_peak_x, start_time
    zero_x = latest_peak_x
    peak_x_history = []
    curve_peak_x.setData(peak_x_history)
    curve_intensity_x.setData(np.zeros(width))
    start_time = time.time_ns()

# Reset function for Y peak position plot
def reset_y_peak_position():
    global peak_y_history, zero_y, latest_peak_y, start_time
    zero_y = latest_peak_y
    peak_y_history = []
    curve_peak_y.setData(peak_y_history)
    curve_intensity_y.setData(np.zeros(height))
    start_time = time.time_ns()

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

# Set up a timer to call the update function every 40 milliseconds
timer = QtCore.QTimer()
timer.timeout.connect(update_plots)
timer.start(40)  # 40 milliseconds

# Create a second window for straightness measurement
win2 = QtWidgets.QMainWindow()
win2.setWindowTitle("Straightness Measurement")
central_widget2 = QtWidgets.QWidget()
win2.setCentralWidget(central_widget2)
main_layout2 = QtWidgets.QVBoxLayout()
central_widget2.setLayout(main_layout2)

# Create a plot widget for straightness measurements in X direction
plot_straightness_x = pg.PlotWidget(title="Straightness Measurements in X Direction")
plot_straightness_x.setBackground('k')
curve_straightness_x = plot_straightness_x.plot(pen='y')
plot_straightness_x.setLabel('left', 'Height Difference (microns)')
plot_straightness_x.setLabel('bottom', 'Position (meters)')
main_layout2.addWidget(plot_straightness_x)

# Create a plot widget for straightness measurements in Y direction
plot_straightness_y = pg.PlotWidget(title="Straightness Measurements in Y Direction")
plot_straightness_y.setBackground('k')
curve_straightness_y = plot_straightness_y.plot(pen='y')
plot_straightness_y.setLabel('left', 'Height Difference (microns)')
plot_straightness_y.setLabel('bottom', 'Position (meters)')
main_layout2.addWidget(plot_straightness_y)

# Create a horizontal layout for the straightness measurement controls
straightness_controls_layout = QtWidgets.QHBoxLayout()
main_layout2.addLayout(straightness_controls_layout)

# Create a box to type in the increment value
increment_label = QtWidgets.QLabel("Increment (m):")
straightness_controls_layout.addWidget(increment_label)
increment_box = QtWidgets.QLineEdit("0.1")
straightness_controls_layout.addWidget(increment_box)

# Create a box to type in the averaging timeframe
timeframe_label = QtWidgets.QLabel("Averaging Timeframe (s):")
straightness_controls_layout.addWidget(timeframe_label)
timeframe_box = QtWidgets.QLineEdit("3")
straightness_controls_layout.addWidget(timeframe_box)

# Create a box for the counting measurement positions
position_label = QtWidgets.QLabel("Measurement Position:")
straightness_controls_layout.addWidget(position_label)
position_box = QtWidgets.QLineEdit("1")
straightness_controls_layout.addWidget(position_box)

# Create a button to take a measurement
button_take_measurement = QtWidgets.QPushButton("Take Measurement")
straightness_controls_layout.addWidget(button_take_measurement)

# Create a button to save the plot
button_save_image_2 = QtWidgets.QPushButton("Save Plot")
straightness_controls_layout.addWidget(button_save_image_2)
button_save_image_2.clicked.connect(lambda: utils.save_window_as_image(win2,app))

# Create a button to clear all measured values
button_clear_values = QtWidgets.QPushButton("Clear All Values")
straightness_controls_layout.addWidget(button_clear_values)

# Create a dropdown menu for unit selection
unit_dropdown = QtWidgets.QComboBox()
unit_dropdown.addItems(["arcseconds", "microns"])
straightness_controls_layout.addWidget(unit_dropdown)

# Create a box to display the min to max difference
min_max_display_x = QtWidgets.QLabel("Min-Max Difference X: 0.0")
straightness_controls_layout.addWidget(min_max_display_x)
min_max_display_y = QtWidgets.QLabel("Min-Max Difference Y: 0.0")
straightness_controls_layout.addWidget(min_max_display_y)

# Initialize data storage for straightness measurements
straightness_measurements_x = []
straightness_measurements_y = []
current_position = 1

# Function to take a straightness measurement
def take_measurement():
    global straightness_measurements_x, straightness_measurements_y, current_position

    try:
        increment = float(increment_box.text())
        timeframe = float(timeframe_box.text())
        position = int(position_box.text())
    except ValueError:
        return

    # Start averaging measurements
    averaging = True
    average_start_time = time.time()
    average_x_values = []
    average_y_values = []

    while time.time() - average_start_time <= timeframe:
        if not np.isnan(latest_peak_x):
            average_x_values.append(latest_peak_x)
        if not np.isnan(latest_peak_y):
            average_y_values.append(latest_peak_y)
        time.sleep(0.005)

    avg_x = np.mean(average_x_values) if average_x_values else 0
    avg_y = np.mean(average_y_values) if average_y_values else 0

    # Convert arcseconds to microns if needed
    if unit_dropdown.currentText() == "microns":
        height_diff_x = avg_x * np.pi * increment * 1e6 / (3600 * 180)
        height_diff_y = avg_y * np.pi * increment * 1e6 / (3600 * 180)
    else:
        height_diff_x = avg_x
        height_diff_y = avg_y

    # Store the measurement
    if position > len(straightness_measurements_x):
        straightness_measurements_x.append((position, height_diff_x))
        straightness_measurements_y.append((position, height_diff_y))
    else:
        straightness_measurements_x[position - 1] = (position, height_diff_x)
        straightness_measurements_y[position - 1] = (position, height_diff_y)

    # Detrend the data
    positions_x, height_diffs_x = zip(*straightness_measurements_x)
    height_diffs_detrended_x = np.array(height_diffs_x) - np.polyval(np.polyfit(positions_x, height_diffs_x, 1),
                                                                     positions_x)

    positions_y, height_diffs_y = zip(*straightness_measurements_y)
    height_diffs_detrended_y = np.array(height_diffs_y) - np.polyval(np.polyfit(positions_y, height_diffs_y, 1),
                                                                     positions_y)

    # Update the plot
    curve_straightness_x.setData(np.array(positions_x) * increment, height_diffs_detrended_x)
    curve_straightness_y.setData(np.array(positions_y) * increment, height_diffs_detrended_y)

    # Update the min-max difference display
    min_max_diff_x = np.max(height_diffs_detrended_x) - np.min(height_diffs_detrended_x)
    min_max_display_x.setText(f"Min-Max Difference X: {min_max_diff_x:.2f}")
    min_max_diff_y = np.max(height_diffs_detrended_y) - np.min(height_diffs_detrended_y)
    min_max_display_y.setText(f"Min-Max Difference Y: {min_max_diff_y:.2f}")

    # Increment the position counter
    current_position = position + 1
    position_box.setText(str(current_position))


# Function to save the plot
#def save_plot():
#    exporter_x = ImageExporter(plot_straightness_x.plotItem)
#    filename_x = datetime.now().strftime("straightness_x_live_%Y%m%d_%H%M%S.png")
#    exporter_x.export(filename_x)
#    exporter_y = ImageExporter(plot_straightness_y.plotItem)
#    filename_y = datetime.now().strftime("straightness_y_live_%Y%m%d_%H%M%S.png")
#    exporter_y.export(filename_y)

# Function to clear all measured values
def clear_all_values():
    global straightness_measurements_x, straightness_measurements_y, current_position

    straightness_measurements_x = []
    straightness_measurements_y = []
    current_position = 1

    curve_straightness_x.setData([], [])
    curve_straightness_y.setData([], [])

    min_max_display_x.setText("Min-Max Difference X: 0.0")
    min_max_display_y.setText("Min-Max Difference Y: 0.0")

    position_box.setText(str(current_position))

# Connect buttons to their respective functions
button_take_measurement.clicked.connect(take_measurement)
button_clear_values.clicked.connect(clear_all_values)

# Show both windows
win.show()
win2.show()

# Start the PyQtGraph application
app.exec_()

# Stop the camera grabbing when the application is closed
camera.StopGrabbing()
camera.Close()
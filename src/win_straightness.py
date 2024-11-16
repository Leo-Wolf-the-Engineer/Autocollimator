# win_straightness.py

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import time
import utils

class StraightnessMeasurementWindow:
    def __init__(self, app, live_window):
        """
        Initialize the StraightnessMeasurementWindow
        :param app:
        :param live_window:
        """
        self.app = app
        self.live_window = live_window

        # Initialize data storage for straightness measurements
        self.straightness_measurements_x = []
        self.straightness_measurements_y = []
        self.current_position = 1

        # Create a window for straightness measurement
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Straightness Measurement")
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Create a plot widget for straightness measurements in X direction
        self.plot_straightness_x = pg.PlotWidget(title="Straightness Measurements in X Direction")
        self.plot_straightness_x.setBackground('k')
        self.curve_straightness_x = self.plot_straightness_x.plot(pen='y')
        self.plot_straightness_x.setLabel('left', 'Height Difference (microns)')
        self.plot_straightness_x.setLabel('bottom', 'Position (meters)')
        main_layout.addWidget(self.plot_straightness_x)

        # Create a plot widget for straightness measurements in Y direction
        self.plot_straightness_y = pg.PlotWidget(title="Straightness Measurements in Y Direction")
        self.plot_straightness_y.setBackground('k')
        self.curve_straightness_y = self.plot_straightness_y.plot(pen='y')
        self.plot_straightness_y.setLabel('left', 'Height Difference (microns)')
        self.plot_straightness_y.setLabel('bottom', 'Position (meters)')
        main_layout.addWidget(self.plot_straightness_y)

        # Create a horizontal layout for the straightness measurement controls
        straightness_controls_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(straightness_controls_layout)

        # Create a box to type in the increment value
        increment_label = QtWidgets.QLabel("Increment (m):")
        straightness_controls_layout.addWidget(increment_label)
        self.increment_box = QtWidgets.QLineEdit("0.1")
        straightness_controls_layout.addWidget(self.increment_box)

        # Create a box to type in the averaging timeframe
        timeframe_label = QtWidgets.QLabel("Averaging Timeframe (s):")
        straightness_controls_layout.addWidget(timeframe_label)
        self.timeframe_box = QtWidgets.QLineEdit("3")
        straightness_controls_layout.addWidget(self.timeframe_box)

        # Create a box for the counting measurement positions
        position_label = QtWidgets.QLabel("Measurement Position:")
        straightness_controls_layout.addWidget(position_label)
        self.position_box = QtWidgets.QLineEdit("1")
        straightness_controls_layout.addWidget(self.position_box)

        # Create a button to take a measurement
        self.button_take_measurement = QtWidgets.QPushButton("Take Measurement")
        straightness_controls_layout.addWidget(self.button_take_measurement)

        # Create a button to save the plot
        self.button_save_image_2 = QtWidgets.QPushButton("Save Plot")
        straightness_controls_layout.addWidget(self.button_save_image_2)
        self.button_save_image_2.clicked.connect(lambda: utils.save_window_as_image(self.win, self.app))

        # Create a button to clear all measured values
        self.button_clear_values = QtWidgets.QPushButton("Clear All Values")
        straightness_controls_layout.addWidget(self.button_clear_values)

        # Create a dropdown menu for unit selection
        self.unit_dropdown = QtWidgets.QComboBox()
        self.unit_dropdown.addItems(["microns", "arcseconds"])
        straightness_controls_layout.addWidget(self.unit_dropdown)

        # Create a box to display the min to max difference
        self.min_max_display_x = QtWidgets.QLabel("Min-Max Difference X: 0.0")
        straightness_controls_layout.addWidget(self.min_max_display_x)
        self.min_max_display_y = QtWidgets.QLabel("Min-Max Difference Y: 0.0")
        straightness_controls_layout.addWidget(self.min_max_display_y)

        # Connect buttons to their respective functions
        self.button_take_measurement.clicked.connect(self.take_measurement)
        self.button_clear_values.clicked.connect(self.clear_all_values)

    def take_measurement(self):
        """
        Take a measurement and update the plot
        :return:
        """
        try:
            increment = float(self.increment_box.text())
            timeframe = float(self.timeframe_box.text())
            position = int(self.position_box.text())
        except ValueError:
            return

        # Start averaging measurements
        average_start_time = time.time()
        average_x_values = []
        average_y_values = []

        while time.time() - average_start_time <= timeframe:
            if not np.isnan(self.live_window.latest_peak_x):
                average_x_values.append(self.live_window.latest_peak_x)
            if not np.isnan(self.live_window.latest_peak_y):
                average_y_values.append(self.live_window.latest_peak_y)

        avg_x = np.mean(average_x_values) if average_x_values else 0
        avg_y = np.mean(average_y_values) if average_y_values else 0

        # Convert arcseconds to microns if needed
        if self.unit_dropdown.currentText() == "microns":
            height_diff_x = avg_x * np.pi * increment * 1e6 / (3600 * 180)
            height_diff_y = avg_y * np.pi * increment * 1e6 / (3600 * 180)
        else:
            height_diff_x = avg_x
            height_diff_y = avg_y

        # Store the measurement
        if position > len(self.straightness_measurements_x):
            self.straightness_measurements_x.append((position, height_diff_x))
            self.straightness_measurements_y.append((position, height_diff_y))
        else:
            self.straightness_measurements_x[position - 1] = (position, height_diff_x)
            self.straightness_measurements_y[position - 1] = (position, height_diff_y)

        # Detrend the data
        positions_x, height_diffs_x = zip(*self.straightness_measurements_x)
        height_diffs_detrended_x = np.array(height_diffs_x) - np.polyval(np.polyfit(positions_x, height_diffs_x, 1), positions_x)

        positions_y, height_diffs_y = zip(*self.straightness_measurements_y)
        height_diffs_detrended_y = np.array(height_diffs_y) - np.polyval(np.polyfit(positions_y, height_diffs_y, 1), positions_y)

        # Update the plot
        self.curve_straightness_x.setData(np.array(positions_x) * increment, height_diffs_detrended_x)
        self.curve_straightness_y.setData(np.array(positions_y) * increment, height_diffs_detrended_y)

        # Update the min-max difference display
        min_max_diff_x = np.max(height_diffs_detrended_x) - np.min(height_diffs_detrended_x)
        self.min_max_display_x.setText(f"Min-Max Difference X: {min_max_diff_x:.2f}")
        min_max_diff_y = np.max(height_diffs_detrended_y) - np.min(height_diffs_detrended_y)
        self.min_max_display_y.setText(f"Min-Max Difference Y: {min_max_diff_y:.2f}")

        # Increment the position counter
        self.current_position = position + 1
        self.position_box.setText(str(self.current_position))

    def clear_all_values(self):
        """
        Clear all measured values
        :return:
        """
        self.straightness_measurements_x = []
        self.straightness_measurements_y = []
        self.current_position = 1

        self.curve_straightness_x.setData([], [])
        self.curve_straightness_y.setData([], [])

        self.min_max_display_x.setText("Min-Max Difference X: 0.0")
        self.min_max_display_y.setText("Min-Max Difference Y: 0.0")

        self.position_box.setText(str(self.current_position))
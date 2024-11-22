import numpy as np
import time
import pyqtgraph as pg
import utils
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer


class StraightnessMeasurementWindow:
    def __init__(self, app, data_storage, poi_data_storage):
        self.app = app
        self.data_storage = data_storage
        self.poi_data_storage = poi_data_storage
        self.current_position = 0

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

        # Create a button to take a new measurement
        self.button_take_new_measurement = QtWidgets.QPushButton("Take Measurement")
        straightness_controls_layout.addWidget(self.button_take_new_measurement)

        # Create a button to average a measurement with the previous ones
        self.button_take_avg = QtWidgets.QPushButton("Take Measurement")
        straightness_controls_layout.addWidget(self.button_take_avg)

        # Create a button to save the plot
        self.button_save_image = QtWidgets.QPushButton("Save Plot")
        straightness_controls_layout.addWidget(self.button_save_image)
        self.button_save_image.clicked.connect(lambda: utils.save_window_as_image(self.win, self.app))

        # Create a button to clear all measured values
        self.button_clear_values = QtWidgets.QPushButton("Clear All Values")
        straightness_controls_layout.addWidget(self.button_clear_values)

        # Create a dropdown menu for unit selection
        self.unit_dropdown = QtWidgets.QComboBox()
        self.unit_dropdown.addItems(["microns", "microradians", "arcseconds", "pixels"])
        straightness_controls_layout.addWidget(self.unit_dropdown)

        # Create a box to display the min to max difference
        self.min_max_display_x = QtWidgets.QLabel("Min-Max Difference X: 0.0")
        straightness_controls_layout.addWidget(self.min_max_display_x)
        self.min_max_display_y = QtWidgets.QLabel("Min-Max Difference Y: 0.0")
        straightness_controls_layout.addWidget(self.min_max_display_y)

        # Connect buttons to their respective functions
        self.button_take_new_measurement.clicked.connect(self.take_measurement("new"))
        self.button_take_avg.clicked.connect(self.take_measurement("avg"))
        self.button_clear_values.clicked.connect(self.clear_all_values)

    def take_measurement(self, measurement_type):
        """
        Take a measurement and update the plot
        :param measurement_type: The type of measurement to be taken, can be "new" or "avg"
        :return:
        """

        # Use QTimer to call the function after the specified timeframe
        QTimer.singleShot(int(self.timeframe * 1000), lambda: self.collect_data(measurement_type))

    def collect_data(self, measurement_type):
        """
        Collect data and update the plot
        :param measurement_type: The type of measurement to be taken, can be "new" or "avg"
        :return:
        """
        try:
            self.increment = float(self.increment_box.text())
            self.timeframe = float(self.timeframe_box.text())
            self.position = int(self.position_box.text())
        except ValueError:
            return
        unit = self.unit_dropdown.currentText()

        # Collect the values added to data_storage during the timeframe
        timestamps = self.data_storage.get_timestamp()
        current_time = time.time_ns()
        start_time = current_time - int(self.timeframe * 1e9)
        indices = [i for i, t in enumerate(timestamps) if start_time <= t <= current_time]

        # Collect the values from the data_storage
        collected_x_values, collected_y_values = self.data_storage.get_data("Pixels")
        collected_x_values = [collected_x_values[i] for i in indices]
        collected_y_values = [collected_y_values[i] for i in indices]

        if measurement_type == "new":
            # Store the values into POIDataStorage
            self.poi_data_storage.new_data(collected_x_values, collected_y_values)
        elif measurement_type == "avg":
            # Store the values into POIDataStorage
            self.poi_data_storage.average_data(collected_x_values, collected_y_values)

        if unit == "Microradians" or unit == "Arcseconds" or unit == "Pixels":
            # Retrieve data in the specified unit
            detrended_x, detrended_y = self.poi_data_storage.get_data(unit)
        else:
            # Retrieve detrended data
            detrended_x, detrended_y = self.poi_data_storage.get_data_detrended("Microradians") * self.increment

        # Update the plot
        self.curve_straightness_x.setData(np.array(positions_x) * self.increment, detrended_x)
        self.curve_straightness_y.setData(np.array(positions_y) * self.increment, detrended_y)

        # Update the min-max difference display
        min_max_diff_x = np.max(detrended_x) - np.min(detrended_x)
        self.min_max_display_x.setText(f"Min-Max Difference X: {min_max_diff_x:.2f}")
        min_max_diff_y = np.max(detrended_y) - np.min(detrended_y)
        self.min_max_display_y.setText(f"Min-Max Difference Y: {min_max_diff_y:.2f}")

        # Increment the position counter
        self.current_position = self.position + 1
        self.position_box.setText(str(self.current_position))
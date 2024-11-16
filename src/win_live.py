# win_live.py

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QScreen
import numpy as np
import time
import utils

class AutocollimatorLiveWindow:
    def __init__(self, processor, conversion_factor, app):
        self.processor = processor
        self.conversion_factor = conversion_factor

        # Initialize data storage
        self.peak_x_history = []
        self.peak_y_history = []
        self.zero_x = 0
        self.zero_y = 0
        self.averaging = False
        self.average_start_time = time.time_ns()
        self.average_x_values = []
        self.average_y_values = []

        # Variables to store the latest frame and peaks
        self.latest_frame = None
        self.latest_peak_x = 0
        self.latest_peak_y = 0

        # Variables for FPS calculation
        self.frame_count = 0
        self.start_time = time.time()

        # Initialize PyQtGraph application
        self.app = app

        # Create a window with a layout
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Autocollimator live")
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Create a vertical layout for the left side (frame and intensity distributions)
        left_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout)

        # Create a plot widget for the current frame
        self.plot_frame = pg.PlotWidget(title="Current Frame")
        self.plot_frame.setBackground('k')
        self.img_item = pg.ImageItem()
        self.plot_frame.addItem(self.img_item)
        left_layout.addWidget(self.plot_frame)

        # Create a plot widget for intensity distribution in X direction
        self.plot_intensity_x = pg.PlotWidget(title="Intensity Distribution in X Direction")
        self.plot_intensity_x.setBackground('k')
        self.curve_intensity_x = self.plot_intensity_x.plot(pen='y')
        self.peak_line_x = self.plot_intensity_x.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        self.plot_intensity_x.setLabel('left', 'Intensity')
        self.plot_intensity_x.setLabel('bottom', 'Pixel Position')
        left_layout.addWidget(self.plot_intensity_x)

        # Create a plot widget for intensity distribution in Y direction
        self.plot_intensity_y = pg.PlotWidget(title="Intensity Distribution in Y Direction")
        self.plot_intensity_y.setBackground('k')
        self.curve_intensity_y = self.plot_intensity_y.plot(pen='y')
        self.peak_line_y = self.plot_intensity_y.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        self.plot_intensity_y.setLabel('left', 'Intensity')
        self.plot_intensity_y.setLabel('bottom', 'Pixel Position')
        left_layout.addWidget(self.plot_intensity_y)

        # Create a vertical layout for the right side (peak positions and buttons)
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_layout)

        # Create a plot widget for peak position in X direction
        self.plot_peak_x = pg.PlotWidget(title="Peak Position in X Direction")
        self.plot_peak_x.setBackground('k')
        self.curve_peak_x = self.plot_peak_x.plot(pen='y')
        self.plot_peak_x.setLabel('left', 'Peak Position (arcseconds)')
        self.plot_peak_x.setLabel('bottom', 'Time (minutes)')
        right_layout.addWidget(self.plot_peak_x)

        # Create a plot widget for peak position in Y direction
        self.plot_peak_y = pg.PlotWidget(title="Peak Position in Y Direction")
        self.plot_peak_y.setBackground('k')
        self.curve_peak_y = self.plot_peak_y.plot(pen='y')
        self.plot_peak_y.setLabel('left', 'Peak Position (arcseconds)')
        self.plot_peak_y.setLabel('bottom', 'Time (minutes)')
        right_layout.addWidget(self.plot_peak_y)

        # Create a horizontal layout for the buttons
        button_layout = QtWidgets.QHBoxLayout()
        right_layout.addLayout(button_layout)

        # Create a button to zero peak position values and reset the X plot
        self.button_reset_x = QtWidgets.QPushButton("Reset X Peak Position")
        button_layout.addWidget(self.button_reset_x)

        # Create a button to zero peak position values and reset the Y plot
        self.button_reset_y = QtWidgets.QPushButton("Reset Y Peak Position")
        button_layout.addWidget(self.button_reset_y)

        # Create a button to start averaging measurements
        self.button_average = QtWidgets.QPushButton("Take Average")
        button_layout.addWidget(self.button_average)

        # Create a text box to display averaged values
        self.average_display = QtWidgets.QLabel("Averaged Values: X = 0.0, Y = 0.0")
        right_layout.addWidget(self.average_display)

        # Create a box to show how many frames per second are received
        self.fps_display = QtWidgets.QLabel("FPS: 0")
        right_layout.addWidget(self.fps_display)

        # Create a button to save the whole window as an image
        self.button_save_image_1 = QtWidgets.QPushButton("Save Window as Image")
        button_layout.addWidget(self.button_save_image_1)
        self.button_save_image_1.clicked.connect(lambda: utils.save_window_as_image(self.win, self.app))

        # Connect buttons to their respective functions
        self.button_reset_x.clicked.connect(self.reset_x_peak_position)
        self.button_reset_y.clicked.connect(self.reset_y_peak_position)
        self.button_average.clicked.connect(self.start_averaging)

        # Set up a timer to call the update function every 40 milliseconds
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(40)  # 40 milliseconds

    def update_plots(self):
        if self.latest_frame is not None:
            self.img_item.setImage(self.latest_frame.T)
            self.curve_intensity_x.setData(np.sum(self.latest_frame, axis=0))
            self.peak_line_x.setValue(self.latest_peak_x)
            self.curve_intensity_y.setData(np.sum(self.latest_frame, axis=1))
            self.peak_line_y.setValue(self.latest_peak_y)
            self.curve_peak_x.setData(*zip(*self.peak_x_history))
            self.curve_peak_y.setData(*zip(*self.peak_y_history))

        # Update FPS display
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time > 3:
            fps = self.frame_count / elapsed_time
            self.fps_display.setText(f"FPS: {fps:.2f}")
            self.frame_count = 0
            self.start_time = current_time

    def reset_x_peak_position(self):
        """
        Reset the peak position in the X direction
        """
        self.zero_x = self.latest_peak_x
        self.peak_x_history = []
        self.curve_peak_x.setData(self.peak_x_history)
        self.curve_intensity_x.setData(np.zeros(self.plot_frame.width()))
        self.start_time = time.time_ns()

    def reset_y_peak_position(self):
        """
        Reset the peak position in the Y direction
        """
        self.zero_y = self.latest_peak_y
        self.peak_y_history = []
        self.curve_peak_y.setData(self.peak_y_history)
        self.curve_intensity_y.setData(np.zeros(self.plot_frame.height()))
        self.start_time = time.time_ns()

#Todo Fix this shit
    def start_averaging(self):
        """
        Start averaging peak positions
        """
        self.averaging = True
        self.average_start_time = time.time_ns()
        self.average_x_values = []
        self.average_y_values = []
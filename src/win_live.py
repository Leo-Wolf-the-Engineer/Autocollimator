import numpy as np
import time
import utils
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from data_storage import ContinousDataStorage
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class AutocollimatorLiveWindow:
    def __init__(self, app, image_frame_storage, data_storage):
        """
        Initialize the Autocollimator live window
        :param app:
        :param image_frame_storage:
        :param data_storage:
        """
        self.app = app
        self.image_frame_storage = image_frame_storage
        self.data_storage = data_storage

        # Initialize data storage
        self.peak_x_history = []
        self.peak_y_history = []
        self.zero_x = 0
        self.zero_y = 0
        self.zero_time = 0
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
        logging.debug("Initialize PyQtGraph Window")
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Autocollimator live")

        logging.debug("Creating central widget")
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Create a vertical layout for the left side (frame and intensity distributions)
        logging.debug("Creating left layout")
        left_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout)

        # Create a plot widget for the current frame
        logging.debug("Creating plot widget for current frame")
        self.plot_frame = pg.PlotWidget(title="Current Frame")
        self.plot_frame.setBackground('k')
        self.img_item = pg.ImageItem()
        self.plot_frame.addItem(self.img_item)
        left_layout.addWidget(self.plot_frame)

        # Create a plot widget for intensity distribution in X direction
        logging.debug("Creating plot widget for intensity distribution in X direction")
        self.plot_intensity_x = pg.PlotWidget(title="Intensity Distribution in X Direction")
        self.plot_intensity_x.setBackground('k')
        self.curve_intensity_x = self.plot_intensity_x.plot(pen='y')
        self.peak_line_x = self.plot_intensity_x.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        self.plot_intensity_x.setLabel('left', 'Intensity')
        self.plot_intensity_x.setLabel('bottom', 'Pixel Position')
        left_layout.addWidget(self.plot_intensity_x)

        # Create a plot widget for intensity distribution in Y direction
        logging.debug("Creating plot widget for intensity distribution in Y direction")
        self.plot_intensity_y = pg.PlotWidget(title="Intensity Distribution in Y Direction")
        self.plot_intensity_y.setBackground('k')
        self.curve_intensity_y = self.plot_intensity_y.plot(pen='y')
        self.peak_line_y = self.plot_intensity_y.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        self.plot_intensity_y.setLabel('left', 'Intensity')
        self.plot_intensity_y.setLabel('bottom', 'Pixel Position')
        left_layout.addWidget(self.plot_intensity_y)

        # Create a vertical layout for the right side (peak positions and buttons)
        logging.debug("Creating right layout")
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

        logging.debug("Creating Buttons")
        # Create a horizontal layout for the buttons
        button_layout = QtWidgets.QHBoxLayout()
        right_layout.addLayout(button_layout)

        # Create a button to reset peak position values
        self.button_reset_peaks = QtWidgets.QPushButton("Reset Peak Positions")
        button_layout.addWidget(self.button_reset_peaks)

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
        self.button_reset_peaks.clicked.connect(self.reset_peak_positions)
        self.button_average.clicked.connect(self.start_averaging)

        # Set up a timer to call the update function every 40 milliseconds
        logging.debug("Set up timer")
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(40)  # 40 milliseconds

    def update_plots(self):
        if self.image_frame_storage:
            self.latest_frame = self.image_frame_storage[-1]
            if self.latest_frame is not None:
                logging.debug("Displaying the latest frame.")
                self.img_item.setImage(self.latest_frame.T)
                self.curve_intensity_x.setData(np.sum(self.latest_frame, axis=0))
                self.curve_intensity_y.setData(np.sum(self.latest_frame, axis=1))
            else:
                logging.warning("Latest frame is None.")
        else:
            logging.warning("No frame in storage.")

        if self.data_storage:
            data_x, data_y = self.data_storage.get_data(unit="Arcseconds")
            timestamps = self.data_storage.get_timestamp()
            self.peak_x_history = [(t, x) for t, sublist in zip(timestamps, data_x) for x in sublist if
                                   t >= self.zero_time]
            self.peak_y_history = [(t, y) for t, sublist in zip(timestamps, data_y) for y in sublist if
                                   t >= self.zero_time]
            self.curve_peak_x.setData([t for t, x in self.peak_x_history],
                                      [x - self.zero_x for t, x in self.peak_x_history])
            self.curve_peak_y.setData([t for t, y in self.peak_y_history],
                                      [y - self.zero_y for t, y in self.peak_y_history])

        # Update FPS display using timestamps
        current_time = time.time_ns()
        three_seconds_ago = current_time - 3 * 1e9
        recent_frames = [t for t in self.data_storage.get_timestamp() if t >= three_seconds_ago]
        fps = len(recent_frames) / 3
        self.fps_display.setText(f"FPS: {fps:.2f}")

    def reset_peak_positions(self):
        """
        Reset the peak positions in both X and Y directions
        """
        self.zero_x = self.latest_peak_x
        self.zero_y = self.latest_peak_y
        self.zero_time = time.time_ns()

    def start_averaging(self):
        """
        Start averaging peak positions
        """
        self.averaging = True
        self.average_start_time = time.time_ns()
        self.average_x_values = []
        self.average_y_values = []

def testing():
    # launch the live window
    app = QtWidgets.QApplication([])
    image_frame_storage = []
    data_storage = ContinousDataStorage(1)
    autocollimator_live_window = AutocollimatorLiveWindow(app, image_frame_storage, data_storage)
    autocollimator_live_window.win.show()
    app.exec_()

if __name__ == "__main__":
    testing()
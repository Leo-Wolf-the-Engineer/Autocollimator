import threading
import queue
import logging
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import time
import utils
from data_storage import ContinousDataStorage

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class FFmpegThread(threading.Thread):
    def __init__(self, task_queue):
        super().__init__()
        self.task_queue = task_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                # Perform FFmpeg operations here
                # Example: process_frame(task)
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.task_queue.put(None)

class AutocollimatorLiveWindow:
    def __init__(self, app, frame_manager, data_storage, task_queue):
        self.app = app
        self.frame_manager = frame_manager
        self.data_storage = data_storage
        self.task_queue = task_queue
        self.latest_frame = None

        # Initialize zero_time and other related attributes
        self.zero_time = time.time_ns()
        self.zero_x = 0
        self.zero_y = 0
        self.latest_peak_x = 0
        self.latest_peak_y = 0
        self.peak_x_history = []
        self.peak_y_history = []
        self.averaging = False

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

        # Create history view plots
        logging.debug("Creating history view plots")
        self.x_history_view = pg.PlotWidget(title="X Position History")
        self.x_history_view.setBackground('k')
        self.x_history_plot = self.x_history_view.plot(pen='r')
        self.x_history_view.setLabel('left', 'X Position (pixels)')
        self.x_history_view.setLabel('bottom', 'Frame')
        right_layout.addWidget(self.x_history_view)

        self.y_history_view = pg.PlotWidget(title="Y Position History")
        self.y_history_view.setBackground('k')
        self.y_history_plot = self.y_history_view.plot(pen='g')
        self.y_history_view.setLabel('left', 'Y Position (pixels)')
        self.y_history_view.setLabel('bottom', 'Frame')
        right_layout.addWidget(self.y_history_view)

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
        self.timer.start(100)

    def update_plots(self):
        try:
            # Get frame from frame manager
            self.latest_frame = self.frame_manager.get_frame()

            if self.latest_frame is not None:
                self.img_item.setImage(self.latest_frame.T)
                self.curve_intensity_x.setData(np.sum(self.latest_frame, axis=0))
                self.curve_intensity_y.setData(np.sum(self.latest_frame, axis=1))
            else:
                logging.warning("Latest frame is None.")
        except Exception as e:
            logging.error(f"Error updating image plots: {e}")

        try:
            # Update peak history plots
            if self.data_storage:
                # Get X and Y data from storage
                data_x, data_y = self.data_storage.get_data(unit="Pixels")
                timestamps = self.data_storage.get_timestamp()
                                # Ensure we have data to process
                if True: #data_x and data_y and timestamps:
                    # Store data directly without flattening
                    self.peak_x_history = data_x
                    self.peak_y_history = data_y

                    # Update history plots
                    print("setting data")
                    self.x_history_plot.setData(range(len(data_x)), data_x)
                    self.y_history_plot.setData(range(len(data_y)), data_y)

                    # Update time series plots
                    time_data = [(t - self.zero_time)/60e9 for t in timestamps if t >= self.zero_time]

                    # Check data lengths to avoid index errors
                    min_length = min(len(time_data), len(data_x), len(data_y))
                    if min_length > 0:
                        self.curve_peak_x.setData(
                            time_data[:min_length],
                            [x - self.zero_x for x in data_x[:min_length]]
                        )
                        self.curve_peak_y.setData(
                            time_data[:min_length],
                            [y - self.zero_y for y in data_y[:min_length]]
                        )

                    # Update peak lines only if we have data
                    if self.latest_frame is not None and data_x:
                        # Update peak_line_x to the newest peak position
                        latest_peak_x = data_x[-1]
                        self.latest_peak_x = latest_peak_x
                        self.peak_line_x.setValue(latest_peak_x + (self.latest_frame.shape[1] / 2))

                    if self.latest_frame is not None and data_y:
                        # Update peak_line_y to the newest peak position
                        latest_peak_y = data_y[-1]
                        self.latest_peak_y = latest_peak_y
                        self.peak_line_y.setValue(latest_peak_y + (self.latest_frame.shape[0] / 2))
        except Exception as e:
            logging.error(f"Error updating peak plots: {e}")

        # Update FPS display using timestamps
        try:
            current_time = time.time_ns()
            three_seconds_ago = current_time - 3 * 1e9
            if self.data_storage:
                timestamps = self.data_storage.get_timestamp()
                recent_frames = [t for t in timestamps if t >= three_seconds_ago]
                fps = len(recent_frames) / 3 if recent_frames else 0
                self.fps_display.setText(f"FPS: {fps:.2f}")
        except Exception as e:
            logging.error(f"Error updating FPS: {e}")

    def reset_peak_positions(self):
        self.zero_x = self.latest_peak_x
        self.zero_y = self.latest_peak_y
        self.zero_time = time.time_ns()

    def start_averaging(self):
        self.averaging = True
        self.average_start_time = time.time_ns()
        self.average_x_values = []
        self.average_y_values = []

class AutocollimatorLiveWindowThread(threading.Thread):
    def __init__(self, frame_manager, data_storage):
        super().__init__()
        self.frame_manager = frame_manager
        self.data_storage = data_storage
        self.task_queue = queue.Queue()
        self.ffmpeg_thread = FFmpegThread(self.task_queue)

    def run(self):
        self.app = QtWidgets.QApplication([])
        self.window = AutocollimatorLiveWindow(self.app, self.frame_manager, self.data_storage, self.task_queue)
        self.window.win.show()

        logging.debug("Thread started")
        self.ffmpeg_thread.start()
        self.app.exec_()
        self.ffmpeg_thread.stop()
        self.ffmpeg_thread.join()

def testing():
    # Create a FrameManager for testing
    class FrameManager:
        def __init__(self):
            self.current_frame = None

        def get_frame(self):
            return self.current_frame

        def update_frame(self, frame):
            self.current_frame = frame

    frame_manager = FrameManager()
    data_storage = ContinousDataStorage(1)
    live_window_thread = AutocollimatorLiveWindowThread(frame_manager, data_storage)
    live_window_thread.start()

if __name__ == "__main__":
    testing()
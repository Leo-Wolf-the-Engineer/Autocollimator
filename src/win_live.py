import threading
import queue
import logging
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import time
import utils
from data_storage import ContinousDataStorage
from collections import deque
import gc

# Configure logging with less overhead
logging.basicConfig(
    level=logging.INFO,  # Raise to INFO to reduce logging overhead
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%H:%M:%S'
)

class FFmpegThread(threading.Thread):
    def __init__(self, task_queue):
        super().__init__()
        self.task_queue = task_queue
        self.running = True
        self.daemon = True  # Make the thread daemon so it exits when main thread exits

    def run(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                # Perform FFmpeg operations here
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
        self.last_update_time = time.time()
        self.update_count = 0
        self.last_fps_update = time.time()
        self.frame_count = 0

        # Initialize zero_time and other related attributes
        self.zero_time = time.time_ns()
        self.zero_x = 0
        self.zero_y = 0
        self.latest_peak_x = 0
        self.latest_peak_y = 0

        # Use deques with max length for history to avoid memory growth
        self.max_history_len = 1000  # Adjust based on your needs
        self.peak_x_history = deque(maxlen=self.max_history_len)
        self.peak_y_history = deque(maxlen=self.max_history_len)
        self.timestamps_history = deque(maxlen=self.max_history_len)
        self.averaging = False

        # Pre-allocate arrays for plot data to avoid constant reallocations
        self.plot_x_data = np.zeros(1000)
        self.plot_y_data = np.zeros(1000)
        self.plot_time_data = np.zeros(1000)

        # Cache for intensity projections
        self.intensity_x_cache = None
        self.intensity_y_cache = None
        self.frame_hash = None

        # Initialize PyQtGraph application with optimized settings
        self.setup_gui()

        # Adaptive update rate based on system load
        self.update_interval = 50  # Start at 50ms (20fps)
        self.min_update_interval = 33  # ~30fps max
        self.max_update_interval = 100  # 10fps min

        # Set up a timer with adaptive rate
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(self.update_interval)

    def setup_gui(self):
        # Initialize PyQtGraph window
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Autocollimator live")

        # Use a widget as the central widget
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Create layouts
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Configure plot widgets with optimized settings
        pg.setConfigOptions(antialias=False, background='k')  # Disable antialiasing for speed

        # Frame plot
        self.plot_frame = pg.PlotWidget(title="Current Frame")
        self.img_item = pg.ImageItem()
        self.plot_frame.addItem(self.img_item)
        self.plot_frame.setLabel('left', 'Height [pixels]')
        self.plot_frame.setLabel('bottom', 'Width [pixels]')
        left_layout.addWidget(self.plot_frame)

        # X intensity plot
        self.plot_intensity_x = pg.PlotWidget(title="X Intensity")
        self.curve_intensity_x = self.plot_intensity_x.plot(pen='y')
        self.peak_line_x = self.plot_intensity_x.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        left_layout.addWidget(self.plot_intensity_x)

        # Y intensity plot
        self.plot_intensity_y = pg.PlotWidget(title="Y Intensity")
        self.curve_intensity_y = self.plot_intensity_y.plot(pen='y')
        self.peak_line_y = self.plot_intensity_y.addLine(x=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        left_layout.addWidget(self.plot_intensity_y)

        # History plots
        self.x_history_view = pg.PlotWidget(title="X Position History")
        self.x_history_plot = self.x_history_view.plot(pen='r')
        right_layout.addWidget(self.x_history_view)

        self.y_history_view = pg.PlotWidget(title="Y Position History")
        self.y_history_plot = self.y_history_view.plot(pen='g')
        right_layout.addWidget(self.y_history_view)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        right_layout.addLayout(button_layout)

        self.button_reset_peaks = QtWidgets.QPushButton("Reset Peak Positions")
        button_layout.addWidget(self.button_reset_peaks)
        self.button_reset_peaks.clicked.connect(self.reset_peak_positions)

        self.button_average = QtWidgets.QPushButton("Take Average")
        button_layout.addWidget(self.button_average)
        self.button_average.clicked.connect(self.start_averaging)

        self.button_save_image_1 = QtWidgets.QPushButton("Save Window as Image")
        button_layout.addWidget(self.button_save_image_1)
        self.button_save_image_1.clicked.connect(lambda: utils.save_window_as_image(self.win, self.app))

        # Status displays
        self.average_display = QtWidgets.QLabel("Averaged Values: X = 0.0, Y = 0.0")
        right_layout.addWidget(self.average_display)

        self.fps_display = QtWidgets.QLabel("FPS: 0")
        right_layout.addWidget(self.fps_display)

    def update_plots(self):
        start_time = time.time()
        updated = False

        # Get frame and check if it's new
        new_frame = self.frame_manager.get_frame()
        if new_frame is not None and id(new_frame) != id(self.latest_frame):
            updated = True
            self.latest_frame = new_frame

            # Only compute projections when we have a new frame
            frame_hash = hash(new_frame.tobytes())
            if frame_hash != self.frame_hash:
                self.frame_hash = frame_hash

                # Compute projections once and reuse
                self.intensity_x_cache = np.sum(new_frame, axis=0, dtype=np.float32)
                self.intensity_y_cache = np.sum(new_frame, axis=1, dtype=np.float32)

                # Update image and projections
                self.img_item.setImage(new_frame.T)
                self.curve_intensity_x.setData(self.intensity_x_cache)
                self.curve_intensity_y.setData(self.intensity_y_cache)

                self.frame_count += 1

        # Update peak history plots (only if we have data and not too frequently)
        current_time = time.time()
        if current_time - self.last_update_time >= 0.04:  # Max 10 history updates per second
            self.last_update_time = current_time

            if self.data_storage:
                try:
                    # Get data efficiently
                    data_x, data_y, timestamps = self.data_storage.get_XY_time_data(unit="Pixels")

                    if len(timestamps) > 0 and len(data_x) > 0:
                        # Update peak lines if we have data
                        if self.latest_frame is not None and len(data_x) > 0:
                            self.latest_peak_x = data_x[-1]
                            self.latest_peak_y = data_y[-1]

                            # Center calculation
                            center_x = self.latest_frame.shape[1] / 2
                            center_y = self.latest_frame.shape[0] / 2

                            # Update peak lines
                            self.peak_line_x.setValue(self.latest_peak_x + center_x)
                            self.peak_line_y.setValue(self.latest_peak_y + center_y)

                        # Update history plots with efficient array handling
                        data_len = min(len(timestamps), self.max_history_len)
                        if data_len > 0:
                            # Resize arrays if needed
                            if data_len > len(self.plot_x_data):
                                self.plot_x_data = np.zeros(data_len * 2)
                                self.plot_y_data = np.zeros(data_len * 2)
                                self.plot_time_data = np.zeros(data_len * 2)

                            # Copy data efficiently
                            self.plot_x_data[:data_len] = data_x[-data_len:]
                            self.plot_y_data[:data_len] = data_y[-data_len:]
                            self.plot_time_data[:data_len] = (timestamps[-data_len:] - self.zero_time)/60e9

                            # Update plots with views (no copies)
                            self.x_history_plot.setData(self.plot_time_data[:data_len], self.plot_x_data[:data_len])
                            self.y_history_plot.setData(self.plot_time_data[:data_len], self.plot_y_data[:data_len])
                            updated = True

                            # Update FPS using timestamp data if enough data is available
                            if current_time - self.last_fps_update >= 2.0 and data_len >= 2:
                                # Calculate time differences between consecutive frames in seconds
                                # Convert from minutes (60e9 / 60 = 1e9) to seconds by multiplying with 60
                                time_diffs = np.diff(self.plot_time_data[:data_len]) * 60
                                # Remove any zero values to avoid division by zero
                                non_zero_diffs = time_diffs[time_diffs > 0]
                                if len(non_zero_diffs) > 0:
                                    # Calculate instantaneous FPS values
                                    inst_fps = 1.0 / non_zero_diffs
                                    # Use recent FPS values (last 10 or fewer)
                                    recent_fps = inst_fps[-min(10, len(inst_fps)):].mean()
                                    self.fps_display.setText(f"FPS: {recent_fps:.1f}")
                                    self.last_fps_update = current_time

                except Exception as e:
                    logging.error(f"Error updating peak plots: {e}", exc_info=False)

        # Update FPS display periodically - fallback if timestamp-based calculation fails
        if current_time - self.last_fps_update >= 2.0:  # Update once per second
            self.fps_display.setText(f"FPS: {self.frame_count / (current_time - self.last_fps_update):.1f}")
            self.frame_count = 0
            self.last_fps_update = current_time

            # Trigger garbage collection occasionally
            if self.update_count % 100 == 0:
                gc.collect()

        # Adjust timer interval based on performance
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        if elapsed > self.update_interval * 0.8:
            # Slow down if we're using too much time
            self.update_interval = min(self.update_interval + 5, self.max_update_interval)
            self.timer.setInterval(self.update_interval)
        elif elapsed < self.update_interval * 0.2 and updated:
            # Speed up if we have headroom
            self.update_interval = max(self.update_interval - 1, self.min_update_interval)
            self.timer.setInterval(self.update_interval)

        self.update_count += 1

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
        self.daemon = True  # Make thread exit when main program exits

    def run(self):
        self.app = QtWidgets.QApplication([])
        self.window = AutocollimatorLiveWindow(self.app, self.frame_manager, self.data_storage, self.task_queue)
        self.window.win.show()

        self.ffmpeg_thread.start()
        self.app.exec_()
        self.ffmpeg_thread.stop()
        self.ffmpeg_thread.join(timeout=1.0)  # Add timeout

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
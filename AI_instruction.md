# Autocollimator Project Structure (`src/`) Documentation

This document outlines the structure and purpose of the files and directories within the `src/` directory of the Autocollimator project based on the provided information.

## Core Components:

*   **`main.py`**:
    *   The main entry point for the application.
    *   Initializes and coordinates core components: `CameraManager`, `ImageFilter`, `ImageProcessor`, `DataStorage`, and the GUI (`AutocollimatorLiveWindowThread`).
    *   Sets up producer (frame acquisition) and consumer (frame processing) threads using `threading` and `queue` for efficient handling of the image stream.
    *   Defines constants like `CONVERSION_FACTOR` for pixel-to-angle conversion based on `PIXEL_PITCH` and `FOCAL_LENGTH`.
*   **`image_aquisition.py`**:
    *   Manages camera input sources.
    *   `CameraManager`: Provides a unified interface for different camera types ('Basler', 'USB', 'AVI'). Includes optional threading (`_grab_loop`) for continuous background frame grabbing and buffering using `deque`.
    *   `BaslerCamera`: Specific implementation for Basler cameras using the `pypylon` library.
    *   `AVIReader`: Implementation for reading frames from AVI video files using OpenCV (`cv2`), potentially acting as a simulator or for post-processing recorded data. Includes helper functions like `testing` and `captureIntoVideo`.
*   **`image_processing/` (Directory)**:
    *   (Contents not provided) Expected to contain modules responsible for analyzing image frames to extract relevant information (e.g., finding the center of the reflected beam/crosshair). `main.py` imports `ImageProcessor` from this location.
        *   `accurate_gaussian_processor.py`: Likely implements a Gaussian-based image processor for accurate peak detection.
        *   `dummy_processor.py`:  A placeholder or simple processor for testing or debugging.
        *   `fast_gaussian_processor.py`:  Likely a faster, potentially less accurate, Gaussian processor variant.
        *   `image_processing.py`:  Base class or abstract definition for image processors.
        *   `linefit_processor.py`:  Processor that uses line fitting for feature extraction.
        *   `peakfinder_processor.py`:  Basic peak finding processor.
        *   `weighted_peakfinder_processor.py`: Implements the weighted peak finder algorithm for subpixel accuracy, currently used in `main.py`.
*   **`image_filter/` (Directory)**:
    *   (Contents not provided) Expected to contain modules for applying filters to raw image frames before processing (e.g., noise reduction, background subtraction). `main.py` imports `ImageFilter` from this location.
        *   `__init__.py`: Initializes the `image_filter` package.
        *   `background_subtraction_filter.py`: Implements background subtraction techniques.
        *   `base_filter.py`: Base class or abstract definition for image filters.
        *   `bilateral_filter.py`: Implements bilateral filtering for noise reduction while preserving edges.
        *   `fourier_filter.py`:  Filters using Fourier transform techniques (frequency domain filtering).
        *   `gaussian_filter.py`: Implements Gaussian blurring for noise reduction.
        *   `image_filter.py`: Likely the main `ImageFilter` class, managing filter application.
        *   `median_filter.py`: Implements median filtering for noise reduction, good for salt-and-pepper noise.
        *   `morphological_filter.py`: Implements morphological operations (erosion, dilation, etc.) for image processing.
*   **`data_storage.py`**:
    *   Handles the storage and retrieval of processed data (e.g., detected peak coordinates).
    *   `ContinousDataStorage`: Designed for storing a continuous stream of data points (X, Y coordinates, timestamps) with a maximum buffer size (`max_points`). Uses `deque` and `numpy` for efficiency and provides data in different units (Pixels, Arcseconds, Microradians) via `get_XY_data`. Includes thread safety (`threading.RLock`).
    *   `POIDataStorage`: Designed for storing specific "Points of Interest" data, potentially related to discrete measurements (like straightness). Allows averaging new data with existing points (`average_data`) and retrieving detrended data (`get_data_detrended`). Uses `numpy` arrays.
*   **`config_ini.py`**:
    *   `ConfigReader`: Reads configuration parameters from an external INI file (hardcoded path in `__init__`).
    *   Uses `configparser` and supports extended interpolation.
    *   Includes functionality to convert values with units (e.g., "10 mm", "5 deg/s") to base units using the `pint` library (`convert_to_base_units`). Handles timestamp placeholders.
*   **`calibration.py`**:
    *   `Corrector`: Implements a calibration correction mechanism.
    *   Uses `scipy.interpolate.griddata` to interpolate correction values based on target vs. actual measurements, likely to compensate for optical distortions or sensor non-linearities.
*   **`utils.py`**:
    *   Contains general utility functions.
    *   Currently includes `save_window_as_image` to capture screenshots of PyQt5 windows using `QScreen.grabWindow`.

## Graphical User Interface (GUI) Windows:

*   **`win_live.py`**:
    *   `AutocollimatorLiveWindow`: Defines the main real-time display window using PyQt5 and `pyqtgraph`.
    *   Shows the live camera feed (`pg.ImageItem`), X/Y intensity profiles, and time-history plots of detected peak positions.
    *   Includes controls (buttons) for resetting relative positions, averaging measurements, and saving the window view. Displays FPS.
    *   Uses `QtCore.QTimer` for periodic updates (`update_plots`). Implements adaptive update rates and caching for performance.
    *   `AutocollimatorLiveWindowThread`: Runs the live window GUI in a separate thread to keep it responsive. Manages an `FFmpegThread` (purpose unclear from code, might be for recording).
*   **`win_straightness.py`**:
    *   `StraightnessMeasurementWindow`: Defines a dedicated window for performing straightness measurements using PyQt5 and `pyqtgraph`.
    *   Plots calculated straightness deviation against position.
    *   Includes controls for setting measurement increments, averaging times, taking/averaging points, clearing data, and selecting units (`QComboBox`).
    *   Interacts with `POIDataStorage` and `ContinousDataStorage` to collect data over a specified timeframe. Calculates and displays min-max difference.
*   **`win_excentricity.py`**:
    *   Placeholder for eccentricity measurement functionality.
    *   Interacts with `ContinousDataStorage` to collect data
    *   Takes the historic values of a certain timespan and fits a circle into them to find the center of spinning target
    *   timespan of values can be set
    *   shows radius of circle
*   **`win_multitarget.py`**:
    *   Currently empty. Placeholder for multi-target tracking or measurement functionality.

## Other Files/Directories:

*   **`testing.py`**: A script likely used for development testing, specifically launching and testing the `win_live.py` window with dummy data storage.
*   **`__init__.py`**: Standard Python file to mark the `src` directory as a package, enabling relative imports within the package.
*   **`.idea/`**: (Directory) Contains project-specific settings for JetBrains IDEs (like PyCharm). Not part of the core application logic.
*   **`__pycache__/`**: (Directory) Automatically created by Python to store compiled bytecode (`.pyc` files). Not part of the core application logic.
*   **`*.png`**: Image files found in the directory (e.g., `autocollimator_live_20250223_214905_gaussian.png`). Likely screenshots or sample data/results generated by the application.

## Optimization Plan for Real-time Performance

**Assumptions:**

*   Target PC: Intel Core i5, 8GB RAM, Integrated Graphics
*   Desired Frame Rate: 200 FPS
*   Accuracy Requirements: Subpixel accuracy is critical.
*   Image Characteristics: Low noise, Bright cross, simple background with some noise.
*   `WeightedPeakfinder` Parameters: Not currently tuned.

**Plan:**

1.  **Baseline Performance Measurement**: Measure the current frame rate and processing time on the target PC to establish a baseline. Profile the `process_frame` method in `WeightedPeakfinder`.
2.  **Profiling**: Use a profiler (e.g., `cProfile`, `line_profiler`) to identify the most time-consuming operations within `process_frame`.
3.  **Optimization - Window Size**: Experiment with reducing the `window_size` parameter in `WeightedPeakfinder` and measure the impact on both speed and accuracy.
4.  **Optimization - Algorithm**: If `WeightedPeakfinder` remains a bottleneck, explore alternative, faster peak-finding or centroiding algorithms.
5.  **Code Optimization**: Use `Numba` or `Cython` to optimize critical sections of the code, especially intensity projection and weighted averaging.
6.  **Parallel Processing**: Explore parallelizing the processing of X and Y intensity profiles.
7.  **Minimize Frame Copying**:  Review code to minimize unnecessary frame copying.
8.  **Performance Monitoring**: Implement performance monitoring to track frame rate, processing time, and CPU usage in real-time.
9.  **Testing on Target PC**:  Continuously test performance on the target PC throughout the optimization process.
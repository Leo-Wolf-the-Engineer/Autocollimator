import time

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import cv2
from numba import jit
import logging

# Configure logging with location information
logging.basicConfig(
    level=logging.DEBUG,  # Set your desired log level
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    datefmt='%H:%M:%S'
)

class ImageProcessor:
    def __init__(self, processor_type, Width, Heigth):
        """
        Initialize the ImageProcessor
        :param processor_type: must be either 'Gaussian', 'Peakfinder' or 'Linefit'
        """
        self.latest_frame = None
        self.Width = Width
        self.Heigth = Heigth

        if processor_type not in ["Gaussian", "Peakfinder", "Linefit", "Dummy"]:
            raise ValueError("processor_type does not exist")
        self.camera_type = processor_type

        if processor_type == "Gaussian":
            self.Processor = Gaussian()
        elif processor_type == "Peakfinder":
            self.Processor = Peakfinder()
        elif processor_type == "Linefit":
            self.Processor = Linefit()
        elif processor_type  == "Dummy":
            self.Processor = Dummy()

    def process_frame(self, frame):
        """
        Process the frame using the selected processor
        Values are then centered around the center of the frame
        :param frame: The input image frame
        :return: The calculated peak positions in Pixels
        """
        values_X, values_Y = self.Processor.process_frame(frame)

        # Ensure values_X and values_Y are iterable
        if not isinstance(values_X, (list, np.ndarray)):
            values_X = [values_X]
        if not isinstance(values_Y, (list, np.ndarray)):
            values_Y = [values_Y]

        # Filter invalid values efficiently
        valid_indices = []
        for i, (x, y) in enumerate(zip(values_X, values_Y)):
            if (not np.isnan(x) and not np.isnan(y) and
                0 <= x < self.Width and 0 <= y < self.Heigth):
                valid_indices.append(i)

        # Extract only valid values
        if len(valid_indices) < len(values_X):
            values_X = [values_X[i] for i in valid_indices]
            values_Y = [values_Y[i] for i in valid_indices]

        # Validate that both lists have the same length
        if len(values_X) != len(values_Y):
            raise ValueError("X and Y values must have the same length")

        # Convert to arrays and center
        values_X = np.array(values_X) - self.Width / 2
        values_Y = np.array(values_Y) - self.Heigth / 2

        return values_X, values_Y


class Gaussian:
    def __init__(self):
        self.latest_frame = None
        # Pre-allocate arrays
        self.intensity_x = None
        self.intensity_y = None
        self.x = None
        self.y = None
        # Cache for intermediate results
        self.last_frame_hash = None

    @staticmethod
    @jit(nopython=True)
    def gaussian(x, a, x0, sigma):
        """Optimized Gaussian function"""
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    @staticmethod
    @jit(nopython=True)
    def fast_gaussian_peak(intensity):
        """Fast peak estimation without curve_fit"""
        max_idx = np.argmax(intensity)
        max_val = intensity[max_idx]

        # If peak is at edge, it's likely invalid
        if max_idx == 0 or max_idx == len(intensity) - 1:
            return np.nan

        # Use weighted centroid around peak for subpixel accuracy
        window = 5  # Adjust window size based on your peak width
        start = max(0, max_idx - window)
        end = min(len(intensity), max_idx + window + 1)

        weights = intensity[start:end]
        positions = np.arange(start, end)

        # Avoid division by zero
        sum_weights = np.sum(weights)
        if sum_weights <= 0:
            return np.nan

        centroid = np.sum(weights * positions) / sum_weights
        return centroid

    def process_frame(self, frame):
        """Process the frame using the Gaussian method"""
        self.latest_frame = frame

        # Define ROI if needed (process center portion for large frames)
        h, w = frame.shape
        roi_size = 256  # Adjust based on your needs

        if max(h, w) > roi_size:
            start_x = max(0, w//2 - roi_size//2)
            end_x = min(w, start_x + roi_size)
            start_y = max(0, h//2 - roi_size//2)
            end_y = min(h, start_y + roi_size)
            roi = frame[start_y:end_y, start_x:end_x]
        else:
            roi = frame
            start_x = start_y = 0

        # Pre-compute sums with np.float32 for better performance
        intensity_x = np.sum(roi, axis=0, dtype=np.float32)
        intensity_y = np.sum(roi, axis=1, dtype=np.float32)

        # Fast peak detection without curve_fit
        peak_x_roi = self.fast_gaussian_peak(intensity_x)
        peak_y_roi = self.fast_gaussian_peak(intensity_y)

        # Convert back to original frame coordinates
        peak_x = np.nan if np.isnan(peak_x_roi) else peak_x_roi + start_x
        peak_y = np.nan if np.isnan(peak_y_roi) else peak_y_roi + start_y

        return peak_x, peak_y


class Peakfinder:
    def __init__(self):
        self.latest_frame = None

    def process_frame(self, frame):
        self.latest_frame = frame

        # Pre-compute sums once
        intensity_x = np.sum(frame, axis=0)
        intensity_y = np.sum(frame, axis=1)

        # Use more efficient peak finding parameters
        peaks_x, properties_x = find_peaks(intensity_x, height=None, distance=5, prominence=100)
        peaks_y, properties_y = find_peaks(intensity_y, height=None, distance=5, prominence=100)

        if len(peaks_x) == 0 or len(peaks_y) == 0:
            return np.nan, np.nan

        # Get most prominent peak directly
        max_idx_x = np.argmax(intensity_x[peaks_x])
        peak_x = peaks_x[max_idx_x]

        max_idx_y = np.argmax(intensity_y[peaks_y])
        peak_y = peaks_y[max_idx_y]

        return peak_x, peak_y


class Linefit:
    def __init__(self):
        self.latest_frame = None
        # Pre-allocate buffers
        self.edges = None
        self._DEBUG = False  # Control debug visualization

    @staticmethod
    @jit(nopython=True)
    def calculate_intersection(avg_rho1, avg_theta1, avg_rho2, avg_theta2):
        """Calculate the intersection point of two lines"""
        if avg_rho1 is None or avg_theta1 is None or avg_rho2 is None or avg_theta2 is None:
            return np.nan, np.nan

        # Convert polar coordinates to Cartesian coordinates
        a1 = np.cos(avg_theta1)
        b1 = np.sin(avg_theta1)
        x1 = a1 * avg_rho1
        y1 = b1 * avg_rho1

        a2 = np.cos(avg_theta2)
        b2 = np.sin(avg_theta2)
        x2 = a2 * avg_rho2
        y2 = b2 * avg_rho2

        # Calculate the intersection point
        # Using determinant method instead of solve for stability
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            return np.nan, np.nan

        peak_x = (b2 * x1 - b1 * x2) / det
        peak_y = (a1 * x2 - a2 * x1) / det

        return peak_x, peak_y

    def process_frame(self, frame, debug=False):
        """Process the frame using the Linefit method"""
        self.latest_frame = frame

        # Convert 16-bit to 8-bit if necessary
        if frame.dtype == np.uint16:
            frame = (frame / 256).astype(np.uint8)

        # Define ROI to focus processing on center area
        height, width = frame.shape[:2]
        roi_x = int(width * 0.25)
        roi_y = int(height * 0.25)
        roi_width = int(width * 0.5)
        roi_height = int(height * 0.5)

        # Process only the ROI
        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

        # Initialize or resize buffer if needed
        if self.edges is None or self.edges.shape != roi.shape:
            self.edges = np.zeros(roi.shape, dtype=np.uint8)

        # Optimized edge detection
        cv2.Canny(roi, 20, 80, edges=self.edges, apertureSize=3, L2gradient=False)

        # Use probabilistic Hough transform (much faster)
        lines = cv2.HoughLinesP(self.edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)

        if lines is not None:
            # Convert probabilistic lines to polar form for consistent processing
            set1 = []  # near vertical lines
            set2 = []  # near horizontal lines

            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Avoid division by zero
                if x2 == x1:
                    angle = np.pi/2
                else:
                    angle = np.arctan(abs(y2-y1)/abs(x2-x1))

                # Convert to standard form: ax + by + c = 0
                a = y2 - y1
                b = x1 - x2
                c = x2*y1 - x1*y2
                # Convert to normal form: rho = x*cos(theta) + y*sin(theta)
                norm = np.sqrt(a*a + b*b)
                if norm == 0:
                    continue

                rho = abs(c) / norm
                # Ensure rho is positive
                if c > 0:
                    a, b = -a, -b

                theta = np.arctan2(b, a)

                # Classify lines based on angle
                if angle < np.pi/4:  # near horizontal
                    set2.append((rho, theta))
                else:  # near vertical
                    set1.append((rho, theta))

            # Process line sets
            avg_rho1, avg_theta1 = self._average_lines(set1)
            avg_rho2, avg_theta2 = self._average_lines(set2)

            # Adjust for ROI offset
            if avg_rho1 is not None and avg_theta1 is not None:
                adj_rho1 = avg_rho1 + roi_x * np.cos(avg_theta1) + roi_y * np.sin(avg_theta1)
            else:
                adj_rho1 = avg_rho1

            if avg_rho2 is not None and avg_theta2 is not None:
                adj_rho2 = avg_rho2 + roi_x * np.cos(avg_theta2) + roi_y * np.sin(avg_theta2)
            else:
                adj_rho2 = avg_rho2

            # Calculate intersection
            peak_x, peak_y = self.calculate_intersection(adj_rho1, avg_theta1, adj_rho2, avg_theta2)

            # DEBUG visualization - only when requested
            if debug and self._DEBUG:
                self._visualize_linefit(frame, self.edges, set1, set2,
                                     avg_rho1, avg_theta1, avg_rho2, avg_theta2,
                                     peak_x, peak_y, roi_x, roi_y)
        else:
            peak_x = peak_y = np.nan

        return peak_x, peak_y

    def _average_lines(self, line_set):
        """Average the lines in the set"""
        if not line_set:
            return None, None

        # Handle duplicate/near-duplicate lines by clustering
        rhos = np.array([rho for rho, _ in line_set])
        thetas = np.array([theta for _, theta in line_set])

        if len(rhos) == 0:
            return None, None

        # Simple average (could be improved with RANSAC or clustering)
        avg_rho = np.mean(rhos)
        avg_theta = np.mean(thetas)

        return avg_rho, avg_theta

    def _visualize_linefit(self, frame, edges, set1, set2, avg_rho1, avg_theta1,
                          avg_rho2, avg_theta2, peak_x, peak_y, roi_x=0, roi_y=0):
        """Debug visualization function - only called when debug=True"""
        import cv2

        # Create a copy of the frame to draw on
        output_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)

        # Draw ROI
        cv2.rectangle(output_frame,
                     (roi_x, roi_y),
                     (roi_x + edges.shape[1], roi_y + edges.shape[0]),
                     (0, 255, 255), 2)

        # Draw lines from set1 (vertical)
        for rho, theta in set1:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho + roi_x
            y0 = b * rho + roi_y
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw lines from set2 (horizontal)
        for rho, theta in set2:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho + roi_x
            y0 = b * rho + roi_y
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Draw average lines
        if avg_rho1 is not None and avg_theta1 is not None:
            a = np.cos(avg_theta1)
            b = np.sin(avg_theta1)
            x0 = a * (avg_rho1 + roi_x * np.cos(avg_theta1) + roi_y * np.sin(avg_theta1))
            y0 = b * (avg_rho1 + roi_x * np.cos(avg_theta1) + roi_y * np.sin(avg_theta1))
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if avg_rho2 is not None and avg_theta2 is not None:
            a = np.cos(avg_theta2)
            b = np.sin(avg_theta2)
            x0 = a * (avg_rho2 + roi_x * np.cos(avg_theta2) + roi_y * np.sin(avg_theta2))
            y0 = b * (avg_rho2 + roi_x * np.cos(avg_theta2) + roi_y * np.sin(avg_theta2))
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw intersection point
        if not np.isnan(peak_x) and not np.isnan(peak_y):
            cv2.circle(output_frame, (int(peak_x), int(peak_y)), 5, (255, 255, 0), -1)

        # Show the result
        cv2.imshow('Linefit Debug', output_frame)
        cv2.waitKey(1)  # Non-blocking display

class Dummy:
    def __init__(self):
        self.latest_frame = None

    def process_frame(self, frame, debug=False):
        return 0, 0
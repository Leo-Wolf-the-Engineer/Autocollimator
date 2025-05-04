import numpy as np
from scipy.signal import find_peaks
import logging


class WeightedPeakfinder:
    def __init__(self, window_size=25, distance=25, prominence=5000):
        self.window_size = window_size
        self.distance = distance
        self.prominence = prominence
        self.latest_frame = None
        self.last_intensity_x = None
        self.last_intensity_y = None
        self._DEBUG = False

    def _find_weighted_peak(self, intensity, peak_pos):
        """Calculate weighted average around peak for subpixel accuracy"""
        half_window = self.window_size // 2

        # Handle boundary conditions
        start = max(0, peak_pos - half_window)
        end = min(len(intensity), peak_pos + half_window)

        # Extract window around peak
        window = intensity[start:end]
        positions = np.arange(start, end)
        positions = positions - peak_pos  # Center around peak position

        # Calculate weighted average
        total_intensity = np.sum(window)

        weighted_pos = np.sum(positions * window) / total_intensity + peak_pos
        return float(weighted_pos)

    def process_frame(self, frame, debug=False):
        """Process the frame to find peaks with subpixel accuracy"""
        # Cache intensity projections for repeated calls with same frame
        if self.latest_frame is None or not np.array_equal(self.latest_frame, frame):
            self.latest_frame = frame.copy()
            self.last_intensity_x = np.sum(frame, axis=0)
            self.last_intensity_y = np.sum(frame, axis=1)

        # Find initial peaks
        peaks_x, _ = find_peaks(self.last_intensity_x, distance=self.distance, prominence=self.prominence)
        peaks_y, _ = find_peaks(self.last_intensity_y, distance=self.distance, prominence=self.prominence)

        if len(peaks_x) == 0 or len(peaks_y) == 0:
            return np.nan, np.nan

        # Get highest intensity peaks
        peak_x = peaks_x[np.argmax(self.last_intensity_x[peaks_x])]
        peak_y = peaks_y[np.argmax(self.last_intensity_y[peaks_y])]

        # Refine with weighted average for subpixel accuracy
        refined_x = self._find_weighted_peak(self.last_intensity_x, peak_x)
        refined_y = self._find_weighted_peak(self.last_intensity_y, peak_y)


        # Debug visualization
        if debug and self._DEBUG:
            self._visualize_peaks(frame, peak_x, peak_y, refined_x, refined_y)

        return refined_x, refined_y

    def _visualize_peaks(self, frame, peak_x, peak_y, refined_x, refined_y):
        """Debug visualization function"""
        import cv2

        # Create a copy of the frame to draw on
        marked_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)

        # Draw initial peak (green)
        cv2.circle(marked_frame, (int(peak_x), int(peak_y)), 7, (0, 255, 0), 1)

        # Draw refined peak (red)
        cv2.circle(marked_frame, (int(refined_x), int(refined_y)), 5, (0, 0, 255), -1)

        # Show the result
        cv2.imshow('AccuratePeakfinder Debug', marked_frame)
        cv2.waitKey(1)  # Non-blocking display
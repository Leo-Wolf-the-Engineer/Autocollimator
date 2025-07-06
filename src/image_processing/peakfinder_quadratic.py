import numpy as np
from scipy.signal import find_peaks


class PeakfinderQuadratic:
    def __init__(self, width=3, degree=2):
        self.latest_frame = None
        self.frame_cache = {}
        self.width = width
        self.degree = degree

        # check if width is odd
        if self.width % 2 == 0:
            raise ValueError("Width must be an odd number.")

    def quadratic_refinement(self, peak_x, peak_y, intensity_x, intensity_y):
        """fit a parabola to the peak region of the given width for subpixel accuracy"""
        half_width = self.width // 2
        # Define the range for fitting
        x_range = np.arange(max(0, peak_x - half_width), min(len(intensity_x), peak_x + half_width + 1))
        y_range = np.arange(max(0, peak_y - half_width), min(len(intensity_y), peak_y + half_width + 1))
        # Fit a parabola to the peak region
        coeffs_x = np.polyfit(x_range, intensity_x[x_range], self.degree)
        coeffs_y = np.polyfit(y_range, intensity_y[y_range], self.degree)
        #print(f"Fitting coefficients for x: {coeffs_x}, y: {coeffs_y}")
        # Calculate the refined peak position
        x_refined = -coeffs_x[1] / (2 * coeffs_x[0])
        y_refined = -coeffs_y[1] / (2 * coeffs_y[0])
        #print(f"Refined peak positions: x={x_refined}, y={y_refined}")
        return x_refined, y_refined

    def process_frame(self, frame):
        # Early return for empty or invalid frames
        if frame is None or frame.size == 0:
            return np.nan, np.nan

        self.latest_frame = frame

        # Pre-compute sums once with axis specification for slight performance gain
        intensity_x = np.sum(frame, axis=0, dtype=np.float32)  # Use float32 for faster computation
        intensity_y = np.sum(frame, axis=1, dtype=np.float32)

        # Skip peak finding if intensities are too low
        if np.max(intensity_x) < 100 or np.max(intensity_y) < 100:
            return np.nan, np.nan

        # Optimized peak finding parameters
        peaks_x, _ = find_peaks(intensity_x, distance=5, prominence=5000, wlen=50)
        peaks_y, _ = find_peaks(intensity_y, distance=5, prominence=5000, wlen=50)

        if len(peaks_x) == 0 or len(peaks_y) == 0:
            return np.nan, np.nan

        # Faster direct peak finding if there are multiple peaks
        peak_x = peaks_x[np.argmax(intensity_x[peaks_x])]
        peak_y = peaks_y[np.argmax(intensity_y[peaks_y])]

        # Refine peak positions using quadratic interpolation
        peak_x, peak_y = self.quadratic_refinement(peak_x, peak_y, intensity_x, intensity_y)

        return peak_x, peak_y
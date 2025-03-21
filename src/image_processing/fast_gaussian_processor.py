import numpy as np
from numba import jit



class FastGaussian:
    def __init__(self, roi_size=256):
        self.latest_frame = None
        # Pre-allocate arrays
        self.intensity_x = None
        self.intensity_y = None
        self.x = None
        self.y = None
        # Cache for intermediate results
        self.last_frame_hash = None
        self.roi_size = roi_size

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

        if max(h, w) > self.roi_size:
            start_x = max(0, w//2 - self.roi_size//2)
            end_x = min(w, start_x + self.roi_size)
            start_y = max(0, h//2 - self.roi_size//2)
            end_y = min(h, start_y + self.roi_size)
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

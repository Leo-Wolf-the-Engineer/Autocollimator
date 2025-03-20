import numpy as np
from scipy.signal import find_peaks


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

import numpy as np
from scipy.signal import find_peaks


class Peakfinder:
    def __init__(self, cache_size=10):
        self.latest_frame = None
        self.frame_cache = {}
        self.cache_size = cache_size
        
    def process_frame(self, frame):
        # Early return for empty or invalid frames
        if frame is None or frame.size == 0:
            return np.nan, np.nan
            
        # Check cache for identical frame (using hash of frame data)
        frame_hash = hash(frame.tobytes())
        if frame_hash in self.frame_cache:
            return self.frame_cache[frame_hash]
            
        self.latest_frame = frame

        # Pre-compute sums once with axis specification for slight performance gain
        intensity_x = np.sum(frame, axis=0, dtype=np.float32)  # Use float32 for faster computation
        intensity_y = np.sum(frame, axis=1, dtype=np.float32)
        
        # Skip peak finding if intensities are too low
        if np.max(intensity_x) < 100 or np.max(intensity_y) < 100:
            return np.nan, np.nan

        # Optimized peak finding parameters
        peaks_x, _ = find_peaks(intensity_x, distance=5, prominence=5000, wlen=20)
        peaks_y, _ = find_peaks(intensity_y, distance=5, prominence=5000, wlen=20)

        if len(peaks_x) == 0 or len(peaks_y) == 0:
            return np.nan, np.nan

        # Faster direct peak finding if there are multiple peaks
        peak_x = peaks_x[np.argmax(intensity_x[peaks_x])]
        peak_y = peaks_y[np.argmax(intensity_y[peaks_y])]
        
        # Cache the result if caching is enabled
        if self.cache_size > 0:
            if len(self.frame_cache) >= self.cache_size:
                # Remove an arbitrary entry if cache is full
                self.frame_cache.pop(next(iter(self.frame_cache)))
            self.frame_cache[frame_hash] = (peak_x, peak_y)
            
        return peak_x, peak_y

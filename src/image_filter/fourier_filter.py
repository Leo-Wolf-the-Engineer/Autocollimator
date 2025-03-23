# image_filter/fourier_filter.py
import cv2
import numpy as np
from .base_filter import BaseFilter

class FourierFilter(BaseFilter):
    def __init__(self, type='lowpass', cutoff=0.5):
        """
        Initialize the FourierFilter
        :param filter_type: must be either 'lowpass' or 'highpass'
        :param cutoff: cutoff frequency as a fraction of the image size
        """
        self.filter_type = type
        self.cutoff = cutoff

    def apply(self, image):
        """Apply Fourier domain filtering"""
        # Convert to float and grayscale if needed
        img = image.astype(np.float32)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Fourier transform
        dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Create filter mask
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols, 2), np.float32)

        # Apply appropriate frequency domain filter
        if self.filter_type == 'lowpass':
            r = int(self.cutoff * min(rows, cols) / 2)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0])**2 + (y - center[1])**2 > r*r
            mask[mask_area] = 0
        elif self.filter_type == 'highpass':
            r = int(self.cutoff * min(rows, cols) / 2)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
            mask[mask_area] = 0

        # Apply mask and inverse DFT
        filtered_dft = dft_shift * mask
        filtered_dft_unshift = np.fft.ifftshift(filtered_dft)
        filtered_img = cv2.idft(filtered_dft_unshift)
        filtered_img = cv2.magnitude(filtered_img[:,:,0], filtered_img[:,:,1])

        # Normalize
        filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)
        return filtered_img.astype(np.uint8)
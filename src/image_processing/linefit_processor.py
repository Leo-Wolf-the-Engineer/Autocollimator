import numpy as np
import cv2
import logging
from numba import jit

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

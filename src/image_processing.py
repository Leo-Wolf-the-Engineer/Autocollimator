import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import cv2

# Todo: Handle Multiple Peaks
class ImageProcessor:
    def __init__(self, processor_type, Width, Heigth):
        """
        Initialize the ImageProcessor
        :param processor_type:
        must be either 'Gaussian', 'CircleFit' or 'Linefit'
        """
        self.latest_frame = None
        self.Width = Width
        self.Heigth = Heigth

        if processor_type not in ["Gaussian", "CircleFit", "Linefit"]:
            raise ValueError("processor_type must be either 'Gaussian' or 'CircleFit'")
        self.camera_type = processor_type
        self.Processor = None

        if processor_type == "Gaussian":
            self.Processor = Gaussian_Processor()

        if processor_type == "CircleFit":
            raise Exception("processor_type CircleFit is not implemented yet")

        if processor_type == "Linefit":
            self.Processor = Linefit()

    def process_frame(self, frame):
        """
        Process the frame using the selected processor
        Values are then centered around the center of the frame
        :param frame: The input image frame
        :return: The calculated peak positions in Pixels
        """
        frame = self.convert_to_grayscale(frame)
        values_X, values_Y = self.Processor.process_frame(frame)

        # Ensure values_X and values_Y are iterable
        if not isinstance(values_X, (list, np.ndarray)):
            values_X = [values_X]
        if not isinstance(values_Y, (list, np.ndarray)):
            values_Y = [values_Y]

        if len(values_X) > 10 or len(values_Y) > 10:
            raise ValueError("Cannot store more than 10 values at a time")
        else:
            for value in values_X:
                if np.isnan(value) or value > self.Width or value < 0:
                    values_X.remove(value)
                    values_Y.remove(value)
                    warnings.warn("Removing NaN or out of range value Pair...")
            for value in values_Y:
                if np.isnan(value) or value > self.Heigth or value < 0:
                    values_X.remove(value)
                    values_Y.remove(value)
                    warnings.warn("Removing NaN or out of range value Pair...")

        # Convert lists to NumPy arrays for element-wise operations
        values_X = np.array(values_X)
        values_Y = np.array(values_Y)

        # Center the values around the center of the frame and return
        return values_X - self.Width / 2, values_Y - self.Heigth / 2

    def convert_to_grayscale(self, frame):
        """
        Convert the input frame to grayscale if it is an RGB image
        :param frame: The input image frame
        :return: The grayscale image
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame


class Gaussian_Processor:
    def __init__(self, ):
        self.latest_frame = None

    def gaussian(self, x, a, x0, sigma):
        """
        Gaussian function
        :param x: The input variable
        :param a: The amplitude of the Gaussian
        :param x0: The center of the Gaussian
        :param sigma: The standard deviation of the Gaussian
        :return: The value of the Gaussian function at x
        """
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def process_frame(self, frame):
        """
        Process the frame using the Gaussian_Processor
        :param frame: grayscale image
        :return:
        """
        self.latest_frame = frame

        # Sum the intensities of the grayscale image
        intensity_x = np.sum(frame, axis=0)
        intensity_y = np.sum(frame, axis=1)

        # Fit Gaussian in the X direction
        x = np.arange(frame.shape[1])
        try:
            popt_x, _ = curve_fit(self.gaussian, x, intensity_x, p0=[np.max(intensity_x), np.argmax(intensity_x), 10])
            peak_x = popt_x[1]
        except RuntimeError:
            peak_x = np.nan

        # Fit Gaussian in the Y direction
        y = np.arange(frame.shape[0])
        try:
            popt_y, _ = curve_fit(self.gaussian, y, intensity_y, p0=[np.max(intensity_y), np.argmax(intensity_y), 10])
            peak_y = popt_y[1]
        except RuntimeError:
            peak_y = np.nan

        return peak_x, peak_y


class Peakfinder_Processor:
    def __init__(self):
        self.latest_frame = None

    def process_frame(self, frame):
        self.latest_frame = frame

        # Sum the intensities of the grayscale image
        intensity_x = np.sum(frame, axis=0)
        intensity_y = np.sum(frame, axis=1)

        # Find peaks in the X direction
        peaks_x, properties_x = find_peaks(intensity_x)
        peak_intensities_x = properties_x['peak_heights']
        sorted_indices_x = np.argsort(peak_intensities_x)[::-1][:10]
        peak_positions_x = peaks_x[sorted_indices_x]

        # Find peaks in the Y direction
        peaks_y, properties_y = find_peaks(intensity_y)
        peak_intensities_y = properties_y['peak_heights']
        sorted_indices_y = np.argsort(peak_intensities_y)[::-1][:10]
        peak_positions_y = peaks_y[sorted_indices_y]

        return peak_positions_x, peak_positions_y


class Linefit:
    def __init__(self):
        self.latest_frame = None

    def process_frame(self, frame):
        """
        Process the frame using the Linefit
        :param frame:
        :return:
        """
        self.latest_frame = frame

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2:
            gray = frame
        else:
            raise ValueError("Unexpected number of channels in the input image")

        if gray.dtype == np.uint16:
            gray = (gray / 16).astype(np.uint8)

        edges = cv2.Canny(gray, 20, 80, apertureSize=3, L2gradient=True)

        lines = cv2.HoughLines(edges, 2, np.pi / 180, 500)

        if lines is not None:
            set1 = []
            set2 = []

            for rho, theta in lines[:, 0]:
                if 0 <= theta < np.pi / 4 or 3 * np.pi / 4 <= theta <= np.pi:
                    set1.append((rho, theta))
                elif np.pi / 4 <= theta < 3 * np.pi / 4:
                    set2.append((rho, theta))

            def average_lines(line_set):
                """
                Average the lines in the set
                :param line_set: The set of lines
                :return: The average rho and theta values
                """
                if not line_set:
                    return None, None

                rho_sum = 0
                theta_sum = 0
                count = 0

                for rho, theta in line_set:
                    rho_sum += rho
                    theta_sum += theta
                    count += 1

                avg_rho = rho_sum / count
                avg_theta = theta_sum / count

                return avg_rho, avg_theta

            avg_rho1, avg_theta1 = average_lines(set1)
            avg_rho2, avg_theta2 = average_lines(set2)

            peak_x, peak_y = calculate_intersection(self, avg_rho1, avg_theta1, avg_rho2, avg_theta2)
        else:
            avg_rho1 = avg_theta1 = avg_rho2 = avg_theta2 = None
            peak_x = peak_y = np.nan

        # visualize_linefit(frame, edges, lines, avg_rho1, avg_theta1, avg_rho2, avg_theta2, peak_x, peak_y)

        return peak_x, peak_y

    def calculate_intersection(self, avg_rho1, avg_theta1, avg_rho2, avg_theta2):
        """
        Calculate the intersection point of two lines
        :param self:
        :param avg_rho1:
        :param avg_theta1:
        :param avg_rho2:
        :param avg_theta2:
        :return:
        """
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
        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([x1, x2])
        try:
            intersection = np.linalg.solve(A, B)
            peak_x = intersection[0]
            peak_y = intersection[1]
        except np.linalg.LinAlgError:
            peak_x = np.nan
            peak_y = np.nan

        return peak_x, peak_y

    def visualize_linefit(frame, edges, lines, avg_rho1, avg_theta1, avg_rho2, avg_theta2, peak_x, peak_y):
        """
        Visualize the line fitting results.
        :param frame: The input image frame
        :param edges: The edges detected by Canny edge detection
        :param lines: The lines detected by Hough Line Transform
        :param avg_rho1: The average rho value of the first set of detected lines
        :param avg_theta1: The average theta value of the first set of detected lines
        :param avg_rho2: The average rho value of the second set of detected lines
        :param avg_theta2: The average theta value of the second set of detected lines
        :param peak_x: The calculated x-coordinate of the center
        :param peak_y: The calculated y-coordinate of the center
        """
        # Create a copy of the frame to draw on
        output_frame = frame.copy()

        # Convert to color if the frame is grayscale
        if len(frame.shape) == 2:
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

        # Draw the edges on the frame
        output_frame[edges != 0] = [0, 0, 255]

        # Draw each detected line on the frame
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the calculated average lines on the frame
        if avg_rho1 is not None and avg_theta1 is not None:
            a = np.cos(avg_theta1)
            b = np.sin(avg_theta1)
            x0 = a * avg_rho1
            y0 = b * avg_rho1
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if avg_rho2 is not None and avg_theta2 is not None:
            a = np.cos(avg_theta2)
            b = np.sin(avg_theta2)
            x0 = a * avg_rho2
            y0 = b * avg_rho2
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw the calculated center on the frame
        if not np.isnan(peak_x) and not np.isnan(peak_y):
            center = (int(peak_x), int(peak_y))
            cv2.circle(output_frame, center, 5, (255, 255, 0), -1)

        # Add legend
        cv2.putText(output_frame, 'Edges', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output_frame, 'Detected Lines', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_frame, 'Average Line Set 1', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(output_frame, 'Average Line Set 2', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output_frame, 'Center', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display the frame with the visualizations
        cv2.imshow('Linefit Visualization', output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
